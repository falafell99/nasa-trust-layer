"""Признаки блока C (интерпретируемые). Вход: результат cinc.load (первые 300 с, ABP не используется).
Метки сюда не попадают. NaN допустимы (заполняются медианой train в model.py)."""
import numpy as np
from .signal import FS, beats, hr_from_peaks, flat_frac, band_frac, ppg_peaks
from .pipeline import quality, Q_WIN

TYPES = ["Asystole", "Bradycardia", "Tachycardia", "Ventricular_Tachycardia", "Ventricular_Flutter_Fib"]
SIGNAL_FEATS = [
    "ecg1_hr_med", "ecg1_rr_mad", "ecg1_rr_max", "ecg_hr_last10", "ecg_hr_last10_minus_med", "ecg2_hr_last10",
    "ecg_peak_pair_frac", "ppg_hr_last10", "ppg_ibi_mad", "ppg_ibi_max", "ppg_amp_cv", "hr_diff_ecg_ppg",
    "ecg_ppg_pair_frac", "ecg_ppg_delay_mad", "flat_ecg1_last10", "flat_ecg2_last10", "amp_p95_p5_ecg1",
    "lf_frac_ecg1", "hf_frac_ecg1", "q", "ppg_missing", "ecg_n_beats_last10",
]
FEATURES = SIGNAL_FEATS + ["type_" + t for t in TYPES]
L10, L30 = Q_WIN * FS, 30 * FS


def _mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def _pair_frac(a, b, lo, hi):
    """доля пиков a, у которых есть пик b в [lo, hi] отсчётов после (lo<0 допускает до)."""
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    hits, delays = 0, []
    for p in a:
        d = b - p
        m = (d >= lo) & (d <= hi)
        if m.any():
            hits += 1
            delays.append(d[m][0] / FS)
    return hits / len(a), (_mad(delays) if len(delays) >= 3 else np.nan)


def extract(d, alarm_type):
    e1 = np.nan_to_num(d["ecg"][0])
    e2 = np.nan_to_num(d["ecg"][1]) if len(d["ecg"]) > 1 else None
    pk1 = beats(e1)
    rr = np.diff(pk1) / FS
    f = dict.fromkeys(SIGNAL_FEATS, np.nan)
    f["ecg1_hr_med"] = 60 / np.median(rr) if len(rr) >= 3 else 0.0
    f["ecg1_rr_mad"] = _mad(rr) if len(rr) >= 3 else np.nan
    f["ecg1_rr_max"] = float(rr.max()) if len(rr) else np.nan
    last1 = pk1[pk1 >= len(e1) - L10]
    pk_l10 = [last1]
    f["ecg_n_beats_last10"] = float(len(last1))
    hr1 = hr_from_peaks(beats(e1[-L10:]))
    hr2 = np.nan
    if e2 is not None:
        hr2 = hr_from_peaks(beats(e2[-L10:]))
    f["ecg2_hr_last10"] = hr2
    f["ecg_hr_last10"] = max(hr1, hr2) if not np.isnan(hr2) else hr1  # без меток: берём лучший канал
    f["ecg_hr_last10_minus_med"] = f["ecg_hr_last10"] - f["ecg1_hr_med"]
    if e2 is not None:
        pk2 = beats(e2)
        f["ecg_peak_pair_frac"], _ = _pair_frac(pk1[pk1 >= len(e1) - L30], pk2, -int(0.1 * FS), int(0.1 * FS))
    f["flat_ecg1_last10"] = flat_frac(e1[-L10:])
    f["flat_ecg2_last10"] = flat_frac(e2[-L10:]) if e2 is not None else np.nan
    f["amp_p95_p5_ecg1"] = float(np.percentile(e1[-L10:], 95) - np.percentile(e1[-L10:], 5))
    f["lf_frac_ecg1"] = band_frac(e1[-L10:], 0.0, 0.5)
    f["hf_frac_ecg1"] = band_frac(e1[-L10:], 20.0, 40.0)
    f["q"] = quality(np.array([x[-L10:] for x in d["ecg"]]))
    ppg = d["ppg"]
    f["ppg_missing"] = float(ppg is None)
    if ppg is not None:
        ppg = np.nan_to_num(ppg)
        pp = ppg_peaks(ppg)
        pp10 = ppg_peaks(ppg[-L10:])
        f["ppg_hr_last10"] = hr_from_peaks(pp10) if len(pp10) >= 3 else 0.0
        ibi = np.diff(pp[pp >= len(ppg) - L30]) / FS
        f["ppg_ibi_mad"] = _mad(ibi) if len(ibi) >= 3 else np.nan
        f["ppg_ibi_max"] = float(ibi.max()) if len(ibi) else np.nan
        amps = ppg[pp[pp >= len(ppg) - L30]]
        f["ppg_amp_cv"] = float(np.std(amps) / abs(np.mean(amps))) if len(amps) >= 3 and np.mean(amps) != 0 else np.nan
        f["hr_diff_ecg_ppg"] = abs(f["ecg_hr_last10"] - f["ppg_hr_last10"])
        f["ecg_ppg_pair_frac"], f["ecg_ppg_delay_mad"] = _pair_frac(pk1[pk1 >= len(e1) - L30], pp, int(0.15 * FS), int(0.6 * FS))
    for t in TYPES:
        f["type_" + t] = float(alarm_type == t)
    return f
