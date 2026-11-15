"""Таблицы качества: по трём записям BUT и по окнам 111001 (72-96 с фрагмента). Старый q (v1) против нового (v2), порог 0.60."""
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from scipy.signal import resample_poly
from quality_design import windows
from src.trust.pipeline import quality, quality_v1
G = 0.60
print("per record, pure 10 s windows (<=400/class), share below %.2f: old q | new q" % G)
for r, role in (("100001", "design"), ("105001", "design"), ("111001", "TEST")):
    for c in (1, 2, 3):
        ws = [x for cc, a, x in windows(r) if cc == c]
        if not ws: continue
        qo = np.array([quality_v1(x[None, :]) for x in ws]); qn = np.array([quality(x[None, :]) for x in ws])
        print("  %s (%s) class %d n=%3d: old %5.1f%% | new %5.1f%%   (mean q old %.2f new %.2f)" % (r, role, c, len(ws), 100 * (qo < G).mean(), 100 * (qn < G).mean(), qo.mean(), qn.mean()))
e = np.fromfile("data/butqdb/111001/frag_ECG_t67740_180s.bin", "<i2").astype(float); x = resample_poly(e, 1, 4)
import json
from src.trust.scenes import _but_ann_classes
cls = _but_ann_classes(180)
print("\n111001 fragment windows (end s: old q, new q, share of class 3 in window):")
for t in range(70, 106, 2):
    w = x[(t - 10) * 250:t * 250]; sh = float((cls[(t - 10) * 1000:t * 1000] == 3).mean())
    qo, qn = quality_v1(w[None, :]), quality(w[None, :])
    print("  %3d-%3d  old %.3f %-5s | new %.3f %-5s | class3 share %.2f" % (t - 10, t, qo, "LOW" if qo < G else "ok", qn, "LOW" if qn < G else "ok", sh))
