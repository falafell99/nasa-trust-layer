"""5-fold CV по группам (запись CinC / пациент VTaC). Только train+val CinC и train+val VTaC; тестовые части не читаются."""
import sys, glob, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, ".")
from src.trust.feat2 import TYPES

META = ["split", "record", "group", "y", "alarm_type", "ds"]


def load_all(use_vtac):
    fr = [pd.read_csv("data/feat2_cinc_train-validation.csv").assign(ds="cinc")]
    if use_vtac:
        fr.append(pd.read_csv("data/feat2_vtac_train-val.csv").assign(ds="vtac"))
    df = pd.concat(fr, ignore_index=True)
    return df, [c for c in df.columns if c not in META and df[c].notna().sum() > 0.02 * len(df)]  # HGB падает на почти пустых столбцах


def models():
    return {
        "lgbm": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=8, random_state=0),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=5.0, random_state=0),
    }


def oof(df, feats, make, seed=0, k=5):
    y, g = df.y.values, df.group.values
    p = np.zeros(len(df))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(df, y, g):
        m = make().fit(df.iloc[tr][feats], y[tr]); p[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    return p


def red_at(p, y, n):
    """лучшие n по p -> (настоящих, всего)"""
    o = np.argsort(-p)[:n]; return int(y[o].sum()), n


if __name__ == "__main__":
    use_vtac = len(sys.argv) > 1 and sys.argv[1] == "vtac"
    df, feats = load_all(use_vtac)
    print("n=%d true=%d features=%d" % (len(df), df.y.sum(), len(feats)))
    # тип-only OOF на CinC (доля настоящих по типу в обучающих фолдах)
    c = df[df.ds == "cinc"].reset_index(drop=True)
    typ_red = np.zeros(len(c), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(c, c.y, c.group):
        r = c.iloc[tr].groupby("alarm_type").y.mean(); typ_red[te] = c.iloc[te].alarm_type.map(r).values >= 0.5
    n_t = int(typ_red.sum()); print("type-only OOF on CinC: red %d/%d true share %.3f" % (c.y[typ_red].sum(), n_t, c.y[typ_red].mean()))
    for name, mk in models().items():
        p = oof(df, feats, mk)
        for ds in ["cinc"] + (["vtac"] if use_vtac else []):
            m = (df.ds == ds).values; y = df.y.values[m]
            s = "%s %s AUROC %.3f" % (name, ds, roc_auc_score(y, p[m]))
            if ds == "cinc":
                t, n = red_at(p[m], y, n_t); s += " | matched size %d: true %d (%.3f) vs type-only %d" % (n, t, t / n, c.y[typ_red].sum())
            print(s)
