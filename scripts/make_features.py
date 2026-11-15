"""Считает признаки для train+validation (ключ holdout НЕ читается) -> data/features_trainval.csv (не в git)."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
sys.path.insert(0, ".")
from src.trust.cinc import load, read_alarms
from src.trust.features import extract

AL = read_alarms()


def one(rec):
    d = load(rec)
    f = extract(d, AL[rec][0])
    f.update(record=rec, alarm_type=AL[rec][0], y=AL[rec][1])  # y только для оценки
    return f


if __name__ == "__main__":
    sp = json.load(open("data/split.json"))
    rows = []
    with ProcessPoolExecutor() as ex:
        for part in ("train", "validation"):
            for r in ex.map(one, sp[part], chunksize=8):
                r["split"] = part
                rows.append(r)
    pd.DataFrame(rows).to_csv("data/features_trainval.csv", index=False)
    print("rows", len(rows), pd.DataFrame(rows).split.value_counts().to_dict())
