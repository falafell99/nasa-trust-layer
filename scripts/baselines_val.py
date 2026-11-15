"""Базовые линии на validation (hold-out НЕ читается). Пороги правила (в) заданы заранее по определениям
тревог CinC 2015 и не подбирались на validation."""
import json, collections
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from cinc_common import FS, ROOT, read_alarms, load

sp = json.load(open("data/split.json"))
alarms = read_alarms()
TRAIN, VAL = sp["train"], sp["validation"]  # ключ holdout не используется


def score(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    return tp, fn, tn, fp, 100 * (tp + tn) / (tp + tn + fp + 5 * fn)


def beats(x):
    x = np.nan_to_num(x)
    b, a = butter(2, [5 / (FS / 2), 15 / (FS / 2)], "band")
    e = np.diff(filtfilt(b, a, x)) ** 2
    e = np.convolve(e, np.ones(int(0.12 * FS)) / int(0.12 * FS), "same")
    thr = 0.3 * np.percentile(e, 99.5) if e.max() > 0 else 1
    pk, _ = find_peaks(e, height=thr, distance=int(0.25 * FS))
    return pk


def flat_frac(x, sec=1):
    n = len(x) // (sec * FS)
    w = np.nan_to_num(x[: n * sec * FS]).reshape(n, -1).std(axis=1)
    ref = np.median(w) if np.median(w) > 0 else 1.0
    return float((w < 0.05 * ref).mean())


def rule(rec, typ):
    """ЭКГ-only, последние 10 с до тревоги, лучшая из двух ЭКГ по числу найденных ударов (без меток)."""
    d = load(rec)
    best = None
    for x in d["ecg"]:
        seg = x[-10 * FS:]
        pk = beats(seg)
        if best is None or len(pk) > len(best[1]):
            best = (seg, pk)
    seg, pk = best
    n = len(pk)
    hr = 60 * FS / np.median(np.diff(pk)) if n >= 3 else (0.0 if n < 2 else 60 * FS / np.diff(pk)[0])
    ff = flat_frac(seg)
    # пороги по определениям CinC 2015 (заданы заранее)
    if typ == "Asystole":
        return int(n <= 2 or ff > 0.5)
    if typ == "Bradycardia":
        return int(hr < 40)
    if typ == "Tachycardia":
        return int(hr > 140)
    if typ == "Ventricular_Tachycardia":
        return int(hr > 100)
    return int(hr > 150)  # VF/Flutter: простое правило по частоте


y_tr = [alarms[r][1] for r in TRAIN]
y = [alarms[r][1] for r in VAL]
typ = [alarms[r][0] for r in VAL]
print("train n=%d true=%d | validation n=%d true=%d false=%d" % (len(TRAIN), sum(y_tr), len(VAL), sum(y), len(VAL) - sum(y)))
rate = collections.defaultdict(lambda: [0, 0])
for r in TRAIN:
    t, l = alarms[r]; rate[t][0] += l; rate[t][1] += 1
print("train true-rate by type:", {t: "%d/%d" % tuple(v) for t, v in sorted(rate.items())})
res = {}
res["(a) keep all"] = score(y, [1] * len(y))
res["(a') suppress all"] = score(y, [0] * len(y))
res["(b) type-only (train rate>=0.5)"] = score(y, [int(rate[t][0] / rate[t][1] >= 0.5) for t in typ])
pr = [rule(r, t) for r, t in zip(VAL, typ)]
res["(c) ECG-only rule"] = score(y, pr)
print("%-34s %4s %4s %4s %4s %7s" % ("baseline", "TP", "FN", "TN", "FP", "score"))
for k, v in res.items():
    print("%-34s %4d %4d %4d %4d %7.2f" % ((k,) + v))
# bootstrap CI по записям (validation), 2000 повторов, seed 0
rng = np.random.RandomState(0)
yv = np.array(y)
preds = {"(a) keep all": np.ones(len(y), int), "(a') suppress all": np.zeros(len(y), int),
         "(b) type-only (train rate>=0.5)": np.array([int(rate[t][0] / rate[t][1] >= 0.5) for t in typ]),
         "(c) ECG-only rule": np.array(pr)}
print("\nбутстрап 95%% ДИ score (по записям, n=%d, B=2000):" % len(y))
for k, p in preds.items():
    s = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y)); s.append(score(yv[i], p[i])[4])
    print("%-34s [%.1f, %.1f]" % (k, *np.percentile(s, [2.5, 97.5])))
print("\n(c) по типам (TP FN TN FP), знаменатель — validation:")
for t in sorted(set(typ)):
    idx = [i for i, tt in enumerate(typ) if tt == t]
    print("  %-26s n=%3d" % (t, len(idx)), score(yv[idx], np.array(pr)[idx])[:4])
