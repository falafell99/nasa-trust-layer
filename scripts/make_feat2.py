"""Признаки v2 + массивы для CNN. Использование: make_feat2.py cinc <part,...> | vtac <part,...>   (part: train, validation, holdout / train, val, test)
Тестовые части (CinC holdout, VTaC test) считаются только отдельным вызовом на финальном шаге."""
import json, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
from src.trust import feat2, vtac
from src.trust.cinc import load, read_alarms

AL = read_alarms()


def job(a):
    ds, key = a
    if ds == "cinc":
        d = load(key); t = AL[key][0]; y = AL[key][1]; grp = key
    else:
        rec, ev = key; d = vtac.load(rec, ev); t = "Ventricular_Tachycardia"; y = None; grp = rec
    f = feat2.extract(d, t)
    return f, feat2.cnn_input(d)


if __name__ == "__main__":
    ds, parts = sys.argv[1], sys.argv[2].split(",")
    rows, keys = [], []
    if ds == "cinc":
        sp = json.load(open("data/split.json"))
        for p in parts:
            for r in sp[p]:
                keys.append(r); rows.append(dict(split=p, record=r, group=r, y=AL[r][1], alarm_type=AL[r][0]))
    else:
        ev = vtac.events()
        for p in parts:
            for _, e in ev[ev.split == p].iterrows():
                keys.append((e.record, e.event)); rows.append(dict(split=p, record=e.event, group=e.record, y=int(e.y), alarm_type="Ventricular_Tachycardia"))
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(job, [(ds, k) for k in keys], chunksize=8))
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame([r[0] for r in res])], axis=1)
    tag = "%s_%s" % (ds, "-".join(parts))
    df.to_csv("data/feat2_%s.csv" % tag, index=False)
    np.save("data/cnn_%s.npy" % tag, np.stack([r[1] for r in res]).astype(np.float16))
    print(tag, df.shape, df.y.value_counts().to_dict())
