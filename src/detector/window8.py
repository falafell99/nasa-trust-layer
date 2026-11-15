"""Classical 8 s VT/VF features for the exploratory record-level CV."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import butter, find_peaks, peak_widths, resample_poly, sosfilt
from scipy.spatial import cKDTree
from scipy.stats import kurtosis

from src.detector.data import SignalRecord, label_windows
from src.detector.features import WindowFeatures

WINDOW_SECONDS = 8.0
STRIDE_SECONDS = 4.0
FEATURE_NAMES = (
    "qrs_rate_bpm", "rr_cv", "rr_median_s", "spectral_concentration",
    "tci_proxy_ms", "vf_filter_leakage", "sample_entropy",
    "qrs_width_proxy_s", "threshold_crossings_per_s", "spectral_entropy",
)


def _sample_entropy(x: np.ndarray) -> float:
    sd = float(np.std(x))
    if sd < 1e-10:
        return 0.0
    r = 0.2 * sd
    counts = []
    for m in (2, 3):
        templates = np.lib.stride_tricks.sliding_window_view(x, m)[::2]
        pairs = cKDTree(templates).query_pairs(r, p=np.inf, output_type="ndarray")
        counts.append(len(pairs))
    return float(-np.log((counts[1] + 1) / (counts[0] + 1)))


def extract_windows8(record: SignalRecord) -> WindowFeatures:
    fs = record.fs
    n = int(WINDOW_SECONDS * fs)
    hop = int(STRIDE_SECONDS * fs)
    sos = butter(3, (0.5, 40.0), btype="bandpass", fs=fs, output="sos")
    signal = sosfilt(sos, record.signal)
    starts = np.arange(0, max(0, len(signal) - n + 1), hop, dtype=np.int64)
    matrix = np.zeros((len(starts), len(FEATURE_NAMES)), dtype=np.float32)
    freq = np.fft.rfftfreq(n, 1 / fs)
    band = (freq >= 0.5) & (freq <= 40)
    for i, start in enumerate(starts):
        x = signal[start:start+n].astype(float, copy=False)
        x = x - np.mean(x)
        sd = float(np.std(x))
        peaks, _ = find_peaks(np.abs(x), distance=int(0.25*fs), prominence=max(1e-8, 0.35*sd))
        rr = np.diff(peaks) / fs
        rate = len(peaks) * 60 / WINDOW_SECONDS
        rr_cv = float(np.std(rr) / np.mean(rr)) if len(rr) > 1 and np.mean(rr) else 0.0
        rr_med = float(np.median(rr)) if len(rr) else 0.0
        widths = peak_widths(np.abs(x), peaks, rel_height=0.5)[0] / fs if len(peaks) else np.empty(0)
        widths = widths[(widths >= .02) & (widths <= .30)]
        qrs_width = float(np.median(widths)) if len(widths) else 0.0
        threshold = .1 * max(float(np.max(np.abs(x))), 1e-10)
        cross = np.flatnonzero((x[:-1] < threshold) & (x[1:] >= threshold))
        tci = float(np.median(np.diff(cross)) * 1000 / fs) if len(cross) > 1 else 0.0
        period = 2*np.pi*np.sum(np.abs(x)) / max(np.sum(np.abs(np.diff(x))), 1e-10)
        shift = max(1, min(len(x)-1, int(round(period / 2))))
        a, b = x[shift:], x[:-shift]
        vf_leak = float(np.sum(np.abs(a+b)) / max(np.sum(np.abs(a)+np.abs(b)), 1e-10))
        power = np.abs(np.fft.rfft(x)) ** 2
        power[~band] = 0
        total = float(power.sum())
        concentration = float(power.max() / total) if total else 0.0
        p = power[band] / total if total else np.zeros(1)
        p = p[p > 0]
        entropy = float(-np.sum(p*np.log(p)) / np.log(max(2, len(p)))) if len(p) else 0.0
        threshold = sd
        crossings = np.count_nonzero((x[:-1] < threshold) & (x[1:] >= threshold))
        crossings += np.count_nonzero((x[:-1] > -threshold) & (x[1:] <= -threshold))
        matrix[i] = (rate, np.clip(rr_cv,0,5), rr_med, concentration, tci, vf_leak,
                     _sample_entropy(x[::2]), qrs_width, crossings/WINDOW_SECONDS, entropy)
    starts_s = starts / fs
    labels = label_windows(starts_s, WINDOW_SECONDS, record.events, min_positive_overlap_seconds=4.0)
    return WindowFeatures(starts_s, matrix, labels)
