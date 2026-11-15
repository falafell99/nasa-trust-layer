"""Контракт интерфейсов (docs/03 §6). verify сейчас = базовая линия V0 (ECG-only правило), не модель.
Константы заданы ДО просмотра демо-данных; q это эвристический балл качества, не вероятность и не обучен."""
import json, os
import numpy as np
from .signal import FS, beats, hr_from_peaks, flat_frac, band_frac, ppg_peaks
from .qual import window_features

SUPPORTED = {"Asystole", "Bradycardia", "Tachycardia", "Ventricular_Tachycardia", "Ventricular_Flutter_Fib"}
Q_GOOD = 0.6   # ниже: жёлтый (качество недостаточно для серого/красного)
Q_WIN = 10     # с, окно качества и проверки


def quality_v1(window, accel=None):
    """Старая эвристика (оставлена для сравнения). Не использовать в продукте: см. progress/logs/Rafael.md."""
    w = np.atleast_2d(window)
    best = 0.0
    for x in w:
        flat = flat_frac(x)
        lf = band_frac(x, 0.0, 0.5)
        hf = band_frac(x, 20.0, 40.0)
        q = (1 - flat) * (1 - min(lf / 0.9, 1.0)) * (1 - min(hf / 0.3, 1.0))
        best = max(best, q)
    return float(best)


_QP = {}


def _qparams():
    if not _QP:
        _QP.update(json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "models", "quality_v2.json"))))
    return _QP


def quality(window, accel=None):
    """window: ndarray[ch, n] (ЭКГ, 250 Гц, 10 с). 0..1, лучший канал. q = 1 - P(класс 2 или 3 BUT QDB) по 9 безразмерным признакам;
    логистическая регрессия обучена на BUT 100001+105001 (models/quality_v2.json). Это оценка качества, не вероятность аритмии.
    accel пока не используется (док. v1: отдельный эксперимент). Перенос на CinC не проверен."""
    P = _qparams()
    best = 0.0
    for x in np.atleast_2d(window):
        f = window_features(x)
        z = (np.array([f[k] for k in P["features"]]) - np.array(P["mean"])) / np.array(P["scale"])
        q = 1.0 - 1.0 / (1.0 + np.exp(-(z @ np.array(P["coef"]) + P["intercept"])))
        best = max(best, float(q))
    return best


def verify_detail(segment):
    """V0: ECG-only правило по последним 10 с. Возвращает (p_true in {0.0,1.0}, info). Не вероятность."""
    fs = segment.get("fs", FS)
    typ = segment["alarm_type"]
    cands = [e for e in (segment.get("ecg1"), segment.get("ecg2")) if e is not None]
    best = None
    for x in cands:
        seg = np.asarray(x)[-Q_WIN * fs:]
        pk = beats(seg, fs)
        if best is None or len(pk) > len(best[1]):
            best = (seg, pk, x)
    if best is None or typ not in SUPPORTED:
        return None, {"reason": "no_ecg" if best is None else "unsupported_type"}
    seg, pk, _ = best
    n = len(pk)
    hr = hr_from_peaks(pk, fs)
    ff = flat_frac(seg, fs)
    if typ == "Asystole":
        p, cond = int(n <= 2 or ff > 0.5), "no QRS or flat line"
    elif typ == "Bradycardia":
        p, cond = int(hr < 40), "HR < 40"
    elif typ == "Tachycardia":
        p, cond = int(hr > 140), "HR > 140"
    elif typ == "Ventricular_Tachycardia":
        p, cond = int(hr > 100), "HR > 100"
    else:
        p, cond = int(hr > 150), "HR > 150"
    info = {"hr": round(hr, 1), "n_beats": int(n), "flat_frac": round(ff, 2), "cond": cond, "met": bool(p)}
    ppg = segment.get("ppg")
    if ppg is not None:
        pp = ppg_peaks(np.asarray(ppg)[-Q_WIN * fs:], fs)
        info["ppg_hr"] = round(hr_from_peaks(pp, fs), 1) if len(pp) >= 3 else None  # справочно, V0 не использует
    return float(p), info


def verify(segment):
    return verify_detail(segment)[0]


def decide(q, p_true, alarm_type):
    """Два статуса (серый отключён, решение лида): red только при q>=Q_GOOD и p>=0.5; иначе yellow (при сомнении жёлтый)."""
    if p_true is None:
        return {"state": "yellow", "reason_code": "no_ecg_or_unsupported_type"}
    if q < Q_GOOD:
        return {"state": "yellow", "reason_code": "signal_quality_low"}
    if p_true >= 0.5:
        return {"state": "red", "reason_code": "condition_met_good_signal"}
    return {"state": "yellow", "reason_code": "condition_not_met_good_signal"}


class VirtualSensor:
    """Отдаёт пакеты {t, channel, samples}; потребитель не знает, что запись."""
    def __init__(self, channels: dict, fs=FS, packet_s=1.0):
        self.ch, self.fs, self.n = channels, fs, int(packet_s * fs)

    def __iter__(self):
        total = min(len(v) for v in self.ch.values())
        for i in range(0, total - self.n + 1, self.n):
            for name, v in self.ch.items():
                yield {"t": i / self.fs, "channel": name, "samples": v[i:i + self.n]}


def run_alarm(rec, alarm_type, d):
    """d: результат cinc_common.load. Возвращает решение + пояснения. Метка записи сюда НЕ передаётся."""
    e1 = d["ecg"][0] if d["ecg"] else None
    e2 = d["ecg"][1] if len(d["ecg"]) > 1 else None
    seg = {"ecg1": e1, "ecg2": e2, "ppg": d["ppg"], "alarm_type": alarm_type, "fs": FS}
    win = np.array([x[-Q_WIN * FS:] for x in d["ecg"]])
    q = quality(win) if len(win) else 0.0
    p, info = verify_detail(seg)
    return {"q": round(q, 3), "p_true": p, "info": info, **decide(q, p, alarm_type)}
