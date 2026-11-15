"""Фиксированное разбиение 750 записей CinC 2015 по записи: seed 42, 60/20/20. Сигналы не читаются."""
import json, os
import numpy as np
from cinc_common import ROOT, read_alarms, channel_names, pick_channels

alarms = read_alarms()
recs = sorted(alarms)
rng = np.random.RandomState(42)
perm = rng.permutation(len(recs))
n_tr, n_va = int(0.6 * len(recs)), int(0.2 * len(recs))
sp = {
    "train": sorted(recs[i] for i in perm[:n_tr]),
    "validation": sorted(recs[i] for i in perm[n_tr:n_tr + n_va]),
    "holdout": sorted(recs[i] for i in perm[n_tr + n_va:]),
}
n_ecg_ppg = sum(1 for r in recs if pick_channels(channel_names(ROOT, r))[0] and pick_channels(channel_names(ROOT, r))[1])
print("ECG>=1 + PLETH records:", n_ecg_ppg, "/", len(recs))
for k, v in sp.items():
    print(k, len(v))
os.makedirs("data", exist_ok=True)
json.dump({"seed": 42, "fractions": [0.6, 0.2, 0.2], "unit": "record", **sp}, open("data/split.json", "w"), indent=1)
