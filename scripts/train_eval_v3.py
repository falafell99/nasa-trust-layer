"""Блок C (два статуса: red/yellow, серый отключён). Обучение на train, отчёт на validation. Hold-out не читается.
Заранее заданы и НЕ подбирались: L2 логистическая регрессия C=1.0, StandardScaler и медианное заполнение по train, T_RED=0.5, Q_GOOD=0.60.
Статус: red = q>=Q_GOOD и p>=T_RED, иначе yellow (жёлтый = тревога сохранена)."""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, ".")
from src.trust.features import FEATURES, SIGNAL_FEATS
from src.trust.model import decide_v3, T_RED
from src.trust.pipeline import Q_GOOD

df = pd.read_csv("data/features_trainval.csv")
tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "validation"].reset_index(drop=True)


def fit(feats, C=1.0):
    med = tr[feats].median().fillna(0.0)
    Xtr = tr[feats].fillna(med).values
    mean, scale = Xtr.mean(0), Xtr.std(0); scale[scale == 0] = 1.0
    m = LogisticRegression(C=C, max_iter=2000).fit((Xtr - mean) / scale, tr.y.values)
    P = {"features": feats, "medians": med.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
         "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "C": C}
    return P, m.predict_proba((va[feats].fillna(med).values - mean) / scale)[:, 1]


def grp(red, y):
    """таблица true/false x red/yellow и доли настоящих в группах."""
    red, y = np.asarray(red, bool), np.asarray(y)
    a, b = int(((y == 1) & red).sum()), int(((y == 1) & ~red).sum())
    c, d = int(((y == 0) & red).sum()), int(((y == 0) & ~red).sum())
    tp, fn, tn, fp = a, b, d, c
    return dict(tr=a, ty=b, fr=c, fy=d, true_in_red=a / max(1, a + c), true_in_yellow=b / max(1, b + d),
                red_share=(a + c) / len(y), true_kept=1.0, score=100 * (a + d) / (a + d + c + 5 * 0))  # score по CinC: см. ниже


def cinc_score(red, y):
    # красная = «предсказано настоящее», жёлтая тоже сохраняет тревогу => для официального score «сохранено» = всё (TP=все настоящие, FP=все не подтверждённые)
    return None


y, q = va.y.values, va.q.values
rng = np.random.RandomState(0)
def boot(fn, red, B=2000):
    v = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y)); v.append(fn(np.asarray(red)[i], y[i]))
    return np.percentile(np.array(v), [2.5, 97.5], axis=0)

def tin_red(red, yy):
    r = red.sum(); return ((yy == 1) & red).sum() / r if r else np.nan
def tin_yel(red, yy):
    k = ~red; return ((yy == 1) & k).sum() / k.sum() if k.sum() else np.nan
def red_share(red, yy): return red.mean()

models = {}
for name, feats in (("V3 (features + alarm type) [primary]", FEATURES), ("V3b (features, no alarm type)", SIGNAL_FEATS)):
    P, p = fit(feats); models[name] = (P, p)
    red = np.array([decide_v3(qq, pp) == "red" for qq, pp in zip(q, p)])
    g = grp(red, y)
    print("\n==", name, "| Q_GOOD=%.2f T_RED=%.2f (not tuned)" % (Q_GOOD, T_RED))
    print("validation n=%d   red  yellow | total" % len(y))
    print("  true          %4d %6d | %d" % (g["tr"], g["ty"], (y == 1).sum()))
    print("  not confirmed %4d %6d | %d" % (g["fr"], g["fy"], (y == 0).sum()))
    print("  true share in red group:    %d/%d = %.3f  95%% CI [%.3f, %.3f] (bootstrap by record, B=2000)" % (g["tr"], g["tr"] + g["fr"], g["true_in_red"], *boot(tin_red, red)))
    print("  true share in yellow group: %d/%d = %.3f  95%% CI [%.3f, %.3f]" % (g["ty"], g["ty"] + g["fy"], g["true_in_yellow"], *boot(tin_yel, red)))
    print("  red group share of all alarms: %d/%d = %.3f  95%% CI [%.3f, %.3f]" % (g["tr"] + g["fr"], len(y), g["red_share"], *boot(red_share, red)))
    print("  true alarms outside red (yellow = still kept, shown with 'Event not excluded'): %d/%d" % (g["ty"], (y == 1).sum()))

print("\n== Baselines on the same table (predicted-true = 'red' group), validation n=%d" % len(y))
import subprocess
for name, red in (("(a) keep all", np.ones(len(y), bool)),):
    g = grp(red, y); print("%-34s red: true %d / not conf. %d | yellow: true %d / not conf. %d | true share red %.3f" % (name, g["tr"], g["fr"], g["ty"], g["fy"], g["true_in_red"]))
print("(b) type-only: red true 35 / not conf. 12 | yellow true 23 / not conf. 80 (true share red 0.745, yellow 0.223) -- from scripts/baselines_val.py")
print("(c) V0 ECG-only rule: red true 36 / not conf. 40 | yellow true 22 / not conf. 52 (true share red 0.474, yellow 0.297)")

P, p = models["V3 (features + alarm type) [primary]"]
json.dump({**P, "t_red": T_RED, "q_good": Q_GOOD, "statuses": ["red", "yellow"], "trained_on": "train (450 records CinC 2015, split seed 42)",
           "thresholds": "Q_GOOD and T_RED fixed a priori, not tuned; gray status disabled", "quality_model": "models/quality_v2.json"},
          open("models/v3_logreg.json", "w"), indent=1)
coef = sorted(zip(P["features"], P["coef"]), key=lambda t: -abs(t[1]))[:10]
print("\ntop-10 |coef| (standardized):", [(k, round(c, 2)) for k, c in coef])
va.assign(p=p, red=[decide_v3(qq, pp) == "red" for qq, pp in zip(q, p)]).to_csv("data/val_v3_predictions.csv", index=False)
print("q on validation: percentiles 5/25/50/75/95:", np.percentile(q, [5, 25, 50, 75, 95]).round(3), "| share q<0.6: %d/%d" % ((q < Q_GOOD).sum(), len(q)))
