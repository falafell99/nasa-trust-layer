"""Признаки качества окна ЭКГ (10 с, 250 Гц). Все признаки безразмерные (доли, отношения), чтобы не зависеть от усиления."""
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
from .signal import FS, beats, flat_frac

QF = ["flat", "lf", "qrs_band", "hf20_40", "vhf40_100", "kurt", "sat", "beats", "rr_plaus", "rr_cv"]


def window_features(x, fs=FS):
    x = np.nan_to_num(np.asarray(x, float))
    f = dict.fromkeys(QF, 0.0)
    f["flat"] = flat_frac(x, fs)
    if np.ptp(x) == 0:
        f.update(kurt=0.0, rr_plaus=0.0, rr_cv=1.0)
        return f
    fr, p = welch(x - x.mean(), fs, nperseg=min(len(x), 10 * fs))
    tot = p[(fr >= 0.05) & (fr <= 100)].sum()
    band = lambda lo, hi: float(p[(fr >= lo) & (fr < hi)].sum() / tot) if tot > 0 else 0.0
    f["lf"], f["qrs_band"], f["hf20_40"], f["vhf40_100"] = band(0.05, 0.5), band(5, 15), band(20, 40), band(40, 100)
    f["kurt"] = float(kurtosis(x))
    f["sat"] = float(np.mean((x == x.max()) | (x == x.min())))
    pk = beats(x, fs)
    f["beats"] = float(len(pk))
    rr = np.diff(pk) / fs
    f["rr_plaus"] = float(np.mean((rr > 0.3) & (rr < 2.0))) if len(rr) else 0.0
    f["rr_cv"] = float(np.std(rr) / np.mean(rr)) if len(rr) >= 3 else 1.0
    return f
