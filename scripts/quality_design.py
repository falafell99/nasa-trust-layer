"""Правило качества v2. Проектируется ТОЛЬКО на BUT 100001 и 105001; 111001 используется только как тест.
Цель (вариант B, выбран по leave-one-record-out между 100001 и 105001, scripts/quality_loro.py): классы 2+3 против класса 1. q = 1 - P(класс 2 или 3). Порог 0.60 не меняется.
Окна: непересекающиеся 10 с, целиком внутри сегмента одного консенсусного класса, до 400 окон на класс и запись (seed 0)."""
import csv, json, sys
import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from src.trust.qual import window_features
from src.trust.pipeline import quality

BASE = "data/butqdb"
FEATS = ["flat", "lf", "qrs_band", "hf20_40", "vhf40_100", "kurt", "beats", "rr_plaus", "rr_cv"]
W = 10000  # отсчётов 1000 Гц


def segs(rec):
    rows = list(csv.reader(open("%s/%s/%s_ANN.csv" % (BASE, rec, rec))))
    return [(int(r[9]), int(r[10]), int(r[11])) for r in rows if r[11].strip() != ""]


def windows(rec, cap=400, seed=0):
    ecg = np.memmap("%s/%s/%s_ECG.dat" % (BASE, rec, rec), dtype="<i2", mode="r")
    rng = np.random.RandomState(seed)
    byc = {1: [], 2: [], 3: []}
    for s, e, c in segs(rec):
        if c in byc:
            for a in range(s - 1, e - W + 1, W):  # 0-based начала, окно целиком в сегменте
                byc[c].append(a)
    out = []
    for c, starts in byc.items():
        if len(starts) > cap:
            starts = list(rng.choice(starts, cap, replace=False))
        for a in starts:
            x = resample_poly(np.asarray(ecg[a:a + W], float), 1, 4)
            out.append((c, a, x))
    return out


def feats(x):
    f = window_features(x)
    return [f[k] for k in FEATS]


def build(recs):
    X, y, q_old, meta = [], [], [], []
    for r in recs:
        for c, a, x in windows(r):
            X.append(feats(x)); y.append(c); q_old.append(quality(x[None, :])); meta.append((r, a / 1000.0))
    return np.array(X), np.array(y), np.array(q_old), meta


if __name__ == "__main__":
    Xtr, ytr, qtr_old, _ = build(["100001", "105001"])
    Xte, yte, qte_old, mte = build(["111001"])
    print("train windows (100001+105001) by class:", {c: int((ytr == c).sum()) for c in (1, 2, 3)})
    print("test windows (111001) by class:", {c: int((yte == c).sum()) for c in (1, 2, 3)})
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit((Xtr - mu) / sd, (ytr >= 2).astype(int))
    qn = lambda X: 1 - m.predict_proba((X - mu) / sd)[:, 1]
    qtr_new, qte_new = qn(Xtr), qn(Xte)
    G = 0.60
    def rep(name, y, qo, qn_, rec):
        print("\n%s: windows below 0.60 by consensus class (old q | new q); AUROC class3-vs-class1 (higher score = worse quality)" % name)
        for c in (1, 2, 3):
            k = y == c
            if k.any():
                print("  class %d n=%4d  old %5.1f%% mean %.3f | new %5.1f%% mean %.3f" % (c, k.sum(), 100 * (qo[k] < G).mean(), qo[k].mean(), 100 * (qn_[k] < G).mean(), qn_[k].mean()))
        k = (y == 1) | (y == 3)
        if (y == 3).any() and (y == 1).any():
            print("  AUROC (class3 vs 1): old %.3f | new %.3f" % (roc_auc_score(y[k] == 3, -qo[k]), roc_auc_score(y[k] == 3, -qn_[k])))
    rr = np.array([m_[0] for m_ in []])
    for r in ("100001", "105001"):
        k = np.array([False] * len(ytr))
    # по записям внутри train
    n100 = int((ytr[: len(ytr)] > 0).sum())
    rep("TRAIN pooled (100001+105001, in-sample)", ytr, qtr_old, qtr_new, "train")
    rep("TEST 111001 (not used for design)", yte, qte_old, qte_new, "111001")
    print("\ncoefficients (standardized features):", {k: round(float(v), 2) for k, v in zip(FEATS, m.coef_[0])})
    json.dump({"features": FEATS, "mean": mu.tolist(), "scale": sd.tolist(), "coef": m.coef_[0].tolist(),
               "intercept": float(m.intercept_[0]), "target": "P(consensus class 2 or 3)", "q": "1-P", "gate": G,
               "trained_on": "BUT QDB 100001+105001 pure 10 s windows, <=400/class/record, seed 0",
               "test_only": "BUT QDB 111001"}, open("models/quality_v2.json", "w"), indent=1)
    np.savez("data/quality_design_windows.npz", Xte=Xte, yte=yte, qte_old=qte_old, qte_new=qte_new, tt=np.array([t for _, t in mte]))
