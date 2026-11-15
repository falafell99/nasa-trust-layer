"""Заморозка V4 по результатам CV (тесты не читаются). Модель: <MODEL> на пуле CinC train+val (600) + VTaC train+val (4555).
Порог T: наибольший p, при котором красная группа OOF на CinC не меньше, чем у «только тип» (186 из 600 по OOF).
Сохраняет models/v4_lgbm.txt и models/v4_config.json."""
import sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, ".")
from scripts.cv_gbm import load_all, models, oof

MODEL = "lgbm"  # GBM-часть; финальная оценка = 0.5*GBM + 0.5*CNN (выбор по CV: blend AUROC 0.958 CinC / 0.956 VTaC против 0.944/0.921 и 0.908/0.939)
df, feats = load_all(True)
c = (df.ds == "cinc").values
y = df.y.values
P = 0.5 * oof(df, feats, models()[MODEL], seed=0) + 0.5 * np.load("data/oof_cnn.npy")  # те же фолды (seed 0), что у CNN-CV
# размер красной группы «только тип» на CinC OOF
cc = df[c].reset_index(drop=True); tr_red = np.zeros(len(cc), bool)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(cc, cc.y, cc.group):
    r = cc.iloc[tr].groupby("alarm_type").y.mean(); tr_red[te] = cc.iloc[te].alarm_type.map(r).values >= 0.5
n_t = int(tr_red.sum())
pc = np.sort(P[c])[::-1]
T = float(pc[n_t - 1])
rep = {}
for name, m in (("cinc", c), ("vtac", ~c)):
    red = P[m] >= T
    rep[name] = dict(n=int(m.sum()), red=int(red.sum()), true_red=int(y[m][red].sum()), share=float(y[m][red].mean()), keepall=float(y[m].mean()))
print("OOF (blend, seed 0), T=%.4f, type-only CinC red=%d (%d true)" % (T, n_t, int(cc.y[tr_red].sum())))
print(json.dumps(rep, indent=1))
m = models()[MODEL]().fit(df[feats], y)
m.booster_.save_model("models/v4_lgbm.txt")
json.dump(dict(model="lgbm+cnn blend 0.5/0.5", blend_cnn=True, features=feats, T=T, oof_report=rep,
               trained_on="CinC train+val 600 + VTaC train+val 4555 (pooled, VTaC alarm type unknown)",
               rule="T = largest p such that CinC OOF red group >= type-only OOF red group size (blend OOF, seed 0)"), open("models/v4_config.json", "w"), indent=1)
