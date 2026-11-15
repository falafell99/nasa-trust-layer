"""Causal, short-window ECG features shared by the rule and logistic model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, peak_widths, sosfilt
from scipy.stats import kurtosis

from src.detector.data import SignalRecord, label_windows


WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
FEATURE_NAMES = (
    "log_rms",
    "log_range",
    "line_length_norm",
    "zero_crossing_rate",
    "kurtosis",
    "qrs_rate_bpm",
    "rr_cv",
    "rr_median_seconds",
    "spectral_entropy",
    "power_below_5hz",
    "power_5_15hz",
    "power_15_40hz",
    "spectral_concentration",
    "threshold_crossings_per_second",
    "qrs_width_proxy_seconds",
)


@dataclass(frozen=True)
class WindowFeatures:
    starts_seconds: np.ndarray
    matrix: np.ndarray
    labels: np.ndarray


def extract_windows(record: SignalRecord) -> WindowFeatures:
    fs = record.fs
    samples = int(round(WINDOW_SECONDS * fs))
    hop = int(round(STRIDE_SECONDS * fs))
    if len(record.signal) < samples:
        return WindowFeatures(np.empty(0), np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int8))

    # A causal filter avoids using samples after an alarm time when computing
    # an offline replay decision.
    sos = butter(3, (0.5, 40.0), btype="bandpass", fs=fs, output="sos")
    filtered = sosfilt(sos, record.signal)
    starts = np.arange(0, len(filtered) - samples + 1, hop, dtype=np.int64)
    matrix = np.zeros((len(starts), len(FEATURE_NAMES)), dtype=np.float32)
    frequencies = np.fft.rfftfreq(samples, d=1.0 / fs)
    band_mask = (frequencies >= 0.5) & (frequencies <= 40.0)

    for i, start in enumerate(starts):
        x = filtered[start : start + samples].astype(np.float64, copy=False)
        centered = x - np.mean(x)
        std = float(np.std(centered))
        rms = float(np.sqrt(np.mean(centered * centered)))
        robust_range = float(np.percentile(x, 95) - np.percentile(x, 5))
        line_length = float(np.mean(np.abs(np.diff(x))))
        zero_crossing = float(np.mean((centered[:-1] >= 0) != (centered[1:] >= 0)))
        shape = float(kurtosis(centered, fisher=True, bias=False)) if std > 1e-10 else 0.0
        if not np.isfinite(shape):
            shape = 0.0

        peak_distance = max(1, int(round(0.25 * fs)))
        peaks, _ = find_peaks(np.abs(centered), distance=peak_distance, prominence=max(1e-8, 0.35 * std))
        rate = len(peaks) * 60.0 / WINDOW_SECONDS
        rr = np.diff(peaks) / fs
        rr_cv = float(np.std(rr) / np.mean(rr)) if len(rr) >= 2 and np.mean(rr) > 0 else 0.0
        rr_median = float(np.median(rr)) if len(rr) else 0.0
        widths = peak_widths(np.abs(centered), peaks, rel_height=0.5)[0] / fs if len(peaks) else np.empty(0)
        plausible_widths = widths[(widths >= 0.02) & (widths <= 0.30)]
        width_proxy = float(np.median(plausible_widths)) if len(plausible_widths) else 0.0
        threshold = 1.0 * std
        threshold_crossings = np.count_nonzero((centered[:-1] < threshold) & (centered[1:] >= threshold))
        threshold_crossings += np.count_nonzero((centered[:-1] > -threshold) & (centered[1:] <= -threshold))
        threshold_crossing_rate = float(threshold_crossings / WINDOW_SECONDS)

        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        spectrum[~band_mask] = 0.0
        total_power = float(np.sum(spectrum))
        if total_power > 0:
            probability = spectrum / total_power
            positive = probability[probability > 0]
            entropy = float(-np.sum(positive * np.log(positive)) / np.log(max(2, len(positive))))
            fractions = []
            for low, high in ((0.5, 5.0), (5.0, 15.0), (15.0, 40.0)):
                mask = (frequencies >= low) & (frequencies < high)
                fractions.append(float(np.sum(spectrum[mask]) / total_power))
            concentration = float(np.max(spectrum[band_mask]) / total_power)
        else:
            entropy, fractions, concentration = 0.0, [0.0, 0.0, 0.0], 0.0

        matrix[i] = (
            np.log1p(rms),
            np.log1p(robust_range),
            line_length / max(std, 1e-8),
            zero_crossing,
            np.clip(shape, -10.0, 50.0),
            rate,
            np.clip(rr_cv, 0.0, 5.0),
            rr_median,
            np.clip(entropy, 0.0, 1.0),
            *fractions,
            concentration,
            threshold_crossing_rate,
            width_proxy,
        )

    starts_seconds = starts.astype(np.float64) / fs
    labels = label_windows(
        starts_seconds,
        WINDOW_SECONDS,
        record.events,
        min_positive_overlap_seconds=0.5,
    )
    return WindowFeatures(starts_seconds, matrix, labels)
