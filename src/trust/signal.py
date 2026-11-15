"""Базовые сигнальные функции (stdlib-free: numpy/scipy). Все частоты в Гц, fs=250 по умолчанию."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

FS = 250


def beats(x, fs=FS):
    """Грубый детектор QRS (полоса 5-15 Гц, энергия производной). Не валидирован на размеченных QRS."""
    x = np.nan_to_num(np.asarray(x, float))
    if len(x) < 3 * fs or np.ptp(x) == 0:
        return np.array([], int)
    b, a = butter(2, [5 / (fs / 2), 15 / (fs / 2)], "band")
    e = np.diff(filtfilt(b, a, x)) ** 2
    w = int(0.12 * fs)
    e = np.convolve(e, np.ones(w) / w, "same")
    thr = 0.3 * np.percentile(e, 99.5)
    pk, _ = find_peaks(e, height=thr, distance=int(0.25 * fs))
    return pk


def hr_from_peaks(pk, fs=FS):
    n = len(pk)
    if n < 2:
        return 0.0
    d = np.diff(pk)
    return float(60 * fs / (np.median(d) if n >= 3 else d[0]))


def flat_frac(x, fs=FS, sec=1):
    n = len(x) // (sec * fs)
    if n == 0:
        return 1.0
    w = np.nan_to_num(np.asarray(x, float)[: n * sec * fs]).reshape(n, -1).std(axis=1)
    ref = np.median(w) if np.median(w) > 0 else 1.0
    return float((w < 0.05 * ref).mean())


def band_frac(x, lo, hi, fs=FS, tot=(0.05, 40.0)):
    x = np.nan_to_num(np.asarray(x, float))
    if len(x) < fs or np.ptp(x) == 0:
        return 0.0
    f, p = welch(x - x.mean(), fs, nperseg=min(len(x), 10 * fs))
    t = p[(f >= tot[0]) & (f <= tot[1])].sum()
    return float(p[(f >= lo) & (f < hi)].sum() / t) if t > 0 else 0.0


def ppg_peaks(x, fs=FS):
    x = np.nan_to_num(np.asarray(x, float))
    if len(x) < 3 * fs or np.ptp(x) == 0:
        return np.array([], int)
    b, a = butter(2, [0.5 / (fs / 2), 5 / (fs / 2)], "band")
    y = filtfilt(b, a, x)
    pk, _ = find_peaks(y, distance=int(0.3 * fs), prominence=0.3 * np.std(y))
    return pk
