"""Выбор варианта правила качества ТОЛЬКО по leave-one-record-out между 100001 и 105001 (111001 не используется)."""
import sys, numpy as np
sys.path.insert(0, ".")
from sklearn.linear_model import LogisticRegression
from quality_design import build, FEATS
D = {r: build([r])[:2] for r in ("100001", "105001")}
for var, tgt in (("A: class3 vs rest", lambda y: (y == 3)), ("B: class2+3 vs class1", lambda y: (y >= 2))):
    for trn, tst in (("105001", "100001"), ("100001", "105001")):
        Xa, ya = D[trn]; Xb, yb = D[tst]
        mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit((Xa - mu) / sd, tgt(ya).astype(int))
        q = 1 - m.predict_proba((Xb - mu) / sd)[:, 1]
        s = "  ".join("class%d n=%d flagged %.0f%%" % (c, (yb == c).sum(), 100 * (q[yb == c] < 0.6).mean()) for c in (1, 2, 3) if (yb == c).any())
        print("%-24s train %s -> test %s: %s" % (var, trn, tst, s))
