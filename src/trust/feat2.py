"""Признаки v2 (интерпретируемые), общие для CinC 2015 и VTaC. Вход: dict(ecg=[...], ecg_names, ppg|None), 250 Гц, первые 300 с.
Метки сюда не попадают. Тип тревоги: у CinC известен, у VTaC в файлах нет -> все one-hot = 0 ("unknown")."""
import warnings
import numpy as np
from scipy.signal import welch, resample_poly
import neurokit2 as nk
from .signal import FS, flat_frac, ppg_peaks, beats
from .qual import window_features

warnings.filterwarnings("ignore")
TYPES = ["Asystole", "Bradycardia", "Tachycardia", "Ventricular_Tachycardia", "Ventricular_Flutter_Fib"]
L10, L15, L30 = 10 * FS, 15 * FS, 30 * FS
N = 300 * FS
NAN = np.nan


def _mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else NAN


def qrs(x):
    """R-пики neurokit2 (метод neurokit); при сбое пустой массив. Независим от грубого beats()."""
    x = np.nan_to_num(np.asarray(x, float))
    if np.ptp(x) == 0:
        return np.array([], int)
    try:
        _, info = nk.ecg_peaks(x, sampling_rate=FS, method="neurokit")
        return np.asarray(info["ECG_R_Peaks"], int)
    except Exception:
        return np.array([], int)


def tcorr(x, pk, half=int(0.25 * FS)):
    """SQI по шаблону: средняя корреляция комплексов (±0.25 с) с медианным шаблоном. NaN если комплексов < 4."""
    x = np.nan_to_num(np.asarray(x, float))
    pk = [p for p in pk if p - half >= 0 and p + half < len(x)]
    if len(pk) < 4:
        return NAN
    S = np.stack([x[p - half:p + half] for p in pk])
    S = S - S.mean(1, keepdims=True)
    t = np.median(S, 0)
    if t.std() == 0:
        return 0.0
    c = [np.corrcoef(s, t)[0, 1] if s.std() > 0 else 0.0 for s in S]
    return float(np.nanmean(c))


def _runs(mask):
    """длина самой длинной серии True подряд."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _roll_hr(rr, k):
    """средняя ЧСС по k подряд идущим интервалам (уд/мин): min и max."""
    if len(rr) < k:
        return NAN, NAN
    m = np.convolve(rr, np.ones(k) / k, "valid")
    h = 60 / m
    return float(h.min()), float(h.max())


def _pair(a, b, lo, hi):
    if len(a) == 0 or len(b) == 0:
        return NAN, NAN
    hits, dl = 0, []
    for p in a:
        d = b - p
        m = (d >= lo) & (d <= hi)
        if m.any():
            hits += 1
            dl.append(d[m][np.argmin(np.abs(d[m]))] / FS)
    return hits / len(a), (_mad(dl) if len(dl) >= 3 else NAN)


def _vf(x):
    """признаки фибрилляции/флаттера в последних 10 с одного отведения."""
    x = np.nan_to_num(np.asarray(x[-L10:], float))
    out = dict(dom_f=NAN, p3_7=NAN, spec_ent=NAN, zc=NAN, amp_ratio=NAN)
    if np.ptp(x) == 0:
        return out
    fr, p = welch(x - x.mean(), FS, nperseg=len(x))
    m = (fr >= 0.5) & (fr <= 40)
    pm = p[m]; fm = fr[m]
    out["dom_f"] = float(fm[np.argmax(pm)])
    out["p3_7"] = float(p[(fr >= 3) & (fr <= 7)].sum() / pm.sum())
    pn = pm / pm.sum()
    out["spec_ent"] = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    out["zc"] = float(np.mean(np.diff(np.sign(x - np.median(x))) != 0))
    return out


def extract(d, alarm_type=None):
    ecg = [np.nan_to_num(np.asarray(e, float)) for e in d["ecg"]][:3]
    f = {}
    # --- пики и качество по отведениям
    per = []
    for x in ecg:
        pk = qrs(x)
        pk0 = beats(x)
        w = x[-L10:]
        qf = window_features(w)
        pk10 = pk[pk >= N - L10]
        per.append(dict(x=x, pk=pk, pk0=pk0, tc=tcorr(w, pk10 - (N - L10)), flat=qf["flat"], nb=len(pk10), qrs_band=qf["qrs_band"], kurt=qf["kurt"],
                        hf=qf["hf20_40"], rr_plaus=qf["rr_plaus"]))
    # качество отведения: шаблонная корреляция (если нет, 0) * доля неплоских секунд; лучшее отведение = L1
    for L in per:
        L["score"] = (0 if np.isnan(L["tc"]) else L["tc"]) * (1 - L["flat"])
    per.sort(key=lambda L: -L["score"])
    for i in range(3):
        L = per[i] if i < len(per) else None
        f["lead%d_tc" % i] = L["tc"] if L else NAN
        f["lead%d_flat" % i] = L["flat"] if L else NAN
        f["lead%d_nb10" % i] = float(L["nb"]) if L else NAN
        f["lead%d_qrsband" % i] = L["qrs_band"] if L else NAN
    f["n_leads"] = float(len(per))
    tcs = [L["tc"] for L in per if not np.isnan(L["tc"])]
    f["tc_max"], f["tc_min"] = (max(tcs), min(tcs)) if tcs else (NAN, NAN)
    f["all_flat10"] = float(np.mean([L["flat"] for L in per])) if per else NAN
    if not per:
        return {**f, **_ppg_block(d, None, None), **_types(alarm_type)}
    L1 = per[0]
    x1, pk1 = L1["x"], L1["pk"]
    # --- ритм по лучшему отведению
    rr30 = np.diff(pk1[pk1 >= N - L30]) / FS
    rr15 = np.diff(pk1[pk1 >= N - L15]) / FS
    rr10 = np.diff(pk1[pk1 >= N - L10]) / FS
    f["hr_med30"] = 60 / np.median(rr30) if len(rr30) >= 3 else 0.0
    f["rr_mad30"] = _mad(rr30) if len(rr30) >= 3 else NAN
    f["rr_max30"] = float(rr30.max()) if len(rr30) else NAN
    f["hr_last10"] = 60 / np.median(rr10) if len(rr10) >= 2 else 0.0
    f["hr_last10_minus_med"] = f["hr_last10"] - f["hr_med30"]
    f["rr_max10"] = float(rr10.max()) if len(rr10) else NAN
    f["rr_min10"] = float(rr10.min()) if len(rr10) else NAN
    f["rr_cv10"] = float(rr10.std() / rr10.mean()) if len(rr10) >= 3 else NAN
    f["nb10_l1"] = float(len(rr10) + 1) if len(rr10) else 0.0
    f["nb10_l1_old"] = float(np.sum(L1["pk0"] >= N - L10))
    # --- определения тревог CinC (последние 15 с)
    f["hr5_min"], f["hr5_max"] = _roll_hr(rr15, 5)
    f["hr17_min"], f["hr17_max"] = _roll_hr(rr15, 17) if len(rr15) >= 17 else (NAN, NAN)
    f["run_brady40"] = float(_runs(rr15 > 1.5))        # серия интервалов > 1.5 с (ЧСС < 40)
    f["run_tachy140"] = float(_runs((rr15 < 60 / 140) & (rr15 > 0.15)))
    f["run_vt100"] = float(_runs((rr15 < 0.6) & (rr15 > 0.15)))
    # асистолия: наибольший промежуток без R-пика в последние 15 с, по лучшему отведению и по всем отведениям (min по отведениям)
    def maxgap(pk, lo):
        q = np.concatenate([[lo], pk[pk >= lo], [N]]); return float(np.diff(q).max() / FS)
    gaps = [maxgap(L["pk"], N - L15) for L in per]
    f["gap15_l1"], f["gap15_min_leads"] = gaps[0], min(gaps)
    pu = np.sort(np.concatenate([L["pk"] for L in per]))
    f["gap15_union"] = maxgap(pu, N - L15)
    # --- фибрилляция
    v = _vf(x1)
    for k, val in v.items():
        f["vf_" + k] = val
    p5 = np.percentile(x1[-L10:], 95) - np.percentile(x1[-L10:], 5)
    pref = np.percentile(x1[:-L30], 95) - np.percentile(x1[:-L30], 5) if len(x1) > L30 else NAN
    f["vf_amp_ratio"] = float(p5 / pref) if pref and not np.isnan(pref) and pref > 0 else NAN
    # --- согласие между отведениями
    if len(per) > 1:
        p2 = per[1]["pk"]
        f["lead_pair_frac"], _ = _pair(pk1[pk1 >= N - L30], p2, -int(0.1 * FS), int(0.1 * FS))
        r2 = np.diff(p2[p2 >= N - L10]) / FS
        f["hr_lead_diff10"] = abs(f["hr_last10"] - (60 / np.median(r2) if len(r2) >= 2 else 0.0))
    else:
        f["lead_pair_frac"] = f["hr_lead_diff10"] = NAN
    f["qrs_old_vs_nk10"] = float(f["nb10_l1_old"] - f["nb10_l1"])
    # --- PPG и согласие ECG/PPG
    f.update(_ppg_block(d, pk1, f))
    f.update(_types(alarm_type))
    return f


def _types(t):
    return {"type_" + k: float(t == k) for k in TYPES}


def _ppg_block(d, pk1, f):
    out = dict(ppg_missing=1.0, ppg_hr10=NAN, ppg_ibi_mad30=NAN, ppg_ibi_max10=NAN, ppg_amp_cv=NAN, ppg_tc=NAN, ppg_flat10=NAN,
               ppg_amp_ratio=NAN, ppg_gap15=NAN, hr_diff_ecg_ppg=NAN, ecg_ppg_pair=NAN, ecg_ppg_delay_mad=NAN, ppg_n10=NAN)
    ppg = d.get("ppg")
    if ppg is None:
        return out
    x = np.nan_to_num(np.asarray(ppg, float))
    out["ppg_missing"] = 0.0
    pp = ppg_peaks(x)
    p10 = pp[pp >= N - L10]
    out["ppg_n10"] = float(len(p10))
    ib10 = np.diff(p10) / FS
    out["ppg_hr10"] = 60 / np.median(ib10) if len(ib10) >= 2 else 0.0
    ib30 = np.diff(pp[pp >= N - L30]) / FS
    out["ppg_ibi_mad30"] = _mad(ib30) if len(ib30) >= 3 else NAN
    out["ppg_ibi_max10"] = float(ib10.max()) if len(ib10) else NAN
    a30 = x[pp[pp >= N - L30]]
    out["ppg_amp_cv"] = float(a30.std() / abs(a30.mean())) if len(a30) >= 3 and a30.mean() != 0 else NAN
    out["ppg_tc"] = tcorr(x[-L10:], p10 - (N - L10), half=int(0.4 * FS))
    out["ppg_flat10"] = flat_frac(x[-L10:])
    a10 = np.ptp(x[-L10:]); aref = np.ptp(x[N - 70 * FS:N - 10 * FS]) if N > 70 * FS else NAN
    out["ppg_amp_ratio"] = float(a10 / aref) if aref and aref > 0 else NAN
    q = np.concatenate([[N - L15], pp[pp >= N - L15], [N]])
    out["ppg_gap15"] = float(np.diff(q).max() / FS)
    if f is not None:
        out["hr_diff_ecg_ppg"] = abs(f["hr_last10"] - out["ppg_hr10"])
        out["ecg_ppg_pair"], out["ecg_ppg_delay_mad"] = _pair(pk1[pk1 >= N - L30], pp, int(0.15 * FS), int(0.6 * FS))
    return out


def cnn_input(d, sec=20, ds=2):
    """Массив [3, sec*FS/ds] (ЭКГ лучшие два отведения по порядку имён + PPG), последние sec с, z-нормировка по каналу; нет канала -> нули."""
    out = np.zeros((3, sec * FS // ds), np.float32)
    chans = [np.asarray(e, float) for e in d["ecg"][:2]] + ([np.asarray(d["ppg"], float)] if d.get("ppg") is not None else [])
    idx = [0, 1] if len(d["ecg"]) >= 2 else [0]
    slots = idx + ([2] if d.get("ppg") is not None else [])
    for s, x in zip(slots, chans):
        x = np.nan_to_num(x[-sec * FS:])
        x = resample_poly(x, 1, ds)
        sd = x.std()
        out[s] = ((x - x.mean()) / sd) if sd > 0 else 0
    return out


FEATURES_ALL = None
