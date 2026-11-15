"""Сборка сцен демо ИЗ ДАННЫХ. Ничего не подставляется: нет файла/записи -> исключение, экран покажет ошибку."""
import csv, json, os
import numpy as np
from scipy.signal import resample_poly
from .cinc import load, read_alarms
from .pipeline import run_alarm, quality, Q_GOOD, Q_WIN
from .signal import FS, flat_frac

SPLIT = "data/split.json"
BUT = "data/butqdb/111001"
BUT_T0 = 67740  # с от начала записи 111001 (фрагмент 180 с)
BUT_DUR = 180
ACC_CAL = [(6.7297, -17228), (9.4294, 8015), (7.124, 4634)]  # из 111001_ACC.hea: gain, baseline (мг)

TYPE_RU = {"Asystole": "Asystole", "Bradycardia": "Bradycardia", "Tachycardia": "Tachycardia",
           "Ventricular_Tachycardia": "Ventricular tachycardia", "Ventricular_Flutter_Fib": "Ventricular flutter/fibrillation"}
ACTION = {
    "red": "Looks real. Check the person now",
    "yellow": "Event not excluded. Check",
    "gray": "Not confirmed by available channels",
}


def _r(x, n=4):
    return np.round(np.nan_to_num(np.asarray(x, float)), n).tolist()  # NaN в записи рисуется нулём


def pick_scene_b():
    """Правило выбора (фиксировано, без просмотра сигналов): validation, есть PLETH, первая по отсортированным ID запись
    для каждого сценария: (1) метка true и решение red; (2) метка false, жёлтый и q<0.3; (3) метка false и серый."""
    sp = json.load(open(SPLIT))
    al = read_alarms()
    want = {"true_red": None, "false_bad": None, "false_good": None}
    for r in sorted(sp["validation"]):  # ключ holdout не используется
        if all(want.values()):
            break
        t, y = al[r]
        d = load(r)
        if d["ppg"] is None or len(d["ecg"]) < 1:
            continue
        o = run_alarm(r, t, d)
        if want["true_red"] is None and y == 1 and o["state"] == "red":
            want["true_red"] = r
        elif want["false_bad"] is None and y == 0 and o["state"] == "yellow" and o["q"] < 0.3:
            want["false_bad"] = r
        elif want["false_good"] is None and y == 0 and o["state"] == "gray":
            want["false_good"] = r
    return want


def build_b():
    al = read_alarms()
    scenes = []
    titles = {"true_red": "B1", "false_bad": "B2", "false_good": "B3"}
    for key, rec in pick_scene_b().items():
        if rec is None:
            raise RuntimeError("no record found for scene " + key)
        t, y = al[rec]
        d = load(rec)
        o = run_alarm(rec, t, d)
        n30 = 30 * FS
        ch = [{"name": n, "x": _r(x[-n30:])} for n, x in zip(d["ecg_names"][:2], d["ecg"][:2])]
        ch.append({"name": "PLETH", "x": _r(d["ppg"][-n30:])})
        flats = {n: round(flat_frac(x[-Q_WIN * FS:]), 2) for n, x in zip(d["ecg_names"][:2], d["ecg"][:2])}
        info = o["info"]
        if o["p_true"] is None:
            check = "Check not run: no ECG, or alarm type not supported."
        else:
            check = ("Simple ECG rule (baseline, not a trained model): over the last 10 s, heart rate %.0f bpm from %d beats; "
                     "condition \"%s\" %s." % (info["hr"], info["n_beats"], info["cond"], "met" if info["met"] else "not met"))
            if info.get("ppg_hr"):
                check += " PPG pulse %.0f bpm (shown for reference, not used)." % info["ppg_hr"]
            if o["state"] == "yellow":
                check += " ECG quality is low, so the rule result is not used to set the alarm aside."
        scenes.append({
            "id": key, "title": titles[key], "record": rec, "alarm_type": TYPE_RU[t], "t_alarm": 300,
            "state": o["state"], "action": ACTION[o["state"]], "reason_code": o["reason_code"],
            "quality_line": "ECG (%s) quality score %.2f of 1 (heuristic score, not a probability; sufficient from %.2f). Flat seconds in last 10 s: %s"
                            % ("/".join(d["ecg_names"][:2]), o["q"], Q_GOOD, ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in flats.items())),
            "check_line": check, "channels": ch, "fs": FS, "highlight_s": Q_WIN,
            "label_reveal": "CinC 2015 expert label: alarm %s." % ("confirmed" if y == 1 else "not confirmed"),
        })
    return scenes


def _but_ann_classes(n_s):
    """Консенсусный класс на каждую секунду фрагмента (из _ANN.csv, группа 4, отсчёты 1-based 1000 Гц)."""
    rows = list(csv.reader(open(os.path.join(BUT, "111001_ANN.csv"))))
    seg = [(int(r[9]), int(r[10]), int(r[11])) for r in rows if r[11].strip() != ""]
    cls = np.zeros(n_s * 1000, int)
    lo, hi = BUT_T0 * 1000 + 1, (BUT_T0 + n_s) * 1000
    for s, e, c in seg:
        a, b = max(s, lo), min(e, hi)
        if a <= b:
            cls[a - lo:b - lo + 1] = c
    return cls


def build_a():
    ecg = np.fromfile(os.path.join(BUT, "frag_ECG_t%d_%ds.bin" % (BUT_T0, BUT_DUR)), "<i2").astype(float)
    acc = np.fromfile(os.path.join(BUT, "frag_ACC_t%d_%ds.bin" % (BUT_T0, BUT_DUR)), "<i2").reshape(-1, 3).astype(float)
    if len(ecg) != BUT_DUR * 1000 or len(acc) != BUT_DUR * 100:
        raise RuntimeError("BUT QDB fragment is incomplete")
    ecg250 = resample_poly(ecg, 1, 4)  # антиалиасинг, не срез [::4]
    accp = np.stack([(acc[:, i] - b) / g for i, (g, b) in enumerate(ACC_CAL)], 1)
    amag = np.linalg.norm(accp, axis=1)
    cls = _but_ann_classes(BUT_DUR)
    wins = []
    for t0 in range(0, BUT_DUR - Q_WIN + 1, 2):
        q = quality(ecg250[t0 * FS:(t0 + Q_WIN) * FS][None, :])
        seg = cls[t0 * 1000:(t0 + Q_WIN) * 1000]
        c = int(np.bincount(seg, minlength=4).argmax())
        mot = float(np.std(amag[t0 * 100:(t0 + Q_WIN) * 100]))
        wins.append({"t": t0 + Q_WIN, "q": round(q, 3), "cls": c, "motion_mg": round(mot, 1)})
    by = {}
    for c in (1, 2, 3):
        qs = [w["q"] for w in wins if w["cls"] == c]
        by[str(c)] = {"n_windows": len(qs), "mean_q": round(float(np.mean(qs)), 3) if qs else None,
                      "share_below_gate": round(float(np.mean([x < Q_GOOD for x in qs])), 3) if qs else None}
    return {
        "record": "BUT QDB 111001", "t0_s": BUT_T0, "duration_s": BUT_DUR, "fs": FS,
        "ecg": _r(ecg250 / np.std(ecg250), 3), "acc_mag_mg": _r(amag, 1)[::2], "acc_fs": 50,
        "class_per_s": cls[::1000].tolist(), "windows": wins, "q_gate": Q_GOOD, "win_s": Q_WIN, "by_class": by,
        "text_low": "Signal quality low. Check the sensor.",
        "text_ok": "Signal quality sufficient",
    }


def build_all():
    out = {"b": None, "a": None, "errors": {}}
    for k, f in (("b", build_b), ("a", build_a)):
        try:
            out[k] = f()
        except Exception as e:  # честно показать, не подставлять заглушку
            out["errors"][k] = "%s: %s" % (type(e).__name__, e)
    return out
