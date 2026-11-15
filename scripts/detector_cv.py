"""Nested record-level CV for 8 s VT/VF windows. Prior holdout waveforms are excluded."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.detector.data import DATA_ROOT, load_record
from src.detector.features import WindowFeatures
from src.detector.window8 import FEATURE_NAMES, extract_windows8

DATASETS = ("vfdb", "cudb", "mitdb")
SPLIT_FILE = Path("results/detector_split.json")
OUT_FILE = Path("results/detector_cv.json")
OUTER_FOLDS, INNER_FOLDS, REARM_SECONDS = 5, 4, 30.0
BOOTSTRAPS, SEED = 2000, 42


def balanced_folds(records, indices, n_folds, seed):
    rng = np.random.RandomState(seed)
    out = [[] for _ in range(n_folds)]
    for ds in DATASETS:
        for has_event in (False, True):
            group = [i for i in indices if records[i]["dataset"] == ds and bool(records[i]["signal"].events) == has_event]
            rng.shuffle(group)
            for j, idx in enumerate(group):
                out[j % n_folds].append(idx)
    return [sorted(x) for x in out]


def fit_predict(records, windows, train_ids, test_ids):
    xs, ys = [], []
    rng = np.random.RandomState(SEED)
    for i in train_ids:
        w = windows[i]
        for cls in (0, 1):
            ids = np.flatnonzero((w.labels == cls))
            if len(ids) > 1500:
                ids = rng.choice(ids, 1500, replace=False)
            if len(ids):
                xs.append(w.matrix[ids]); ys.append(np.full(len(ids), cls))
    x = np.concatenate(xs); y = np.concatenate(ys)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.1, class_weight="balanced", max_iter=300, random_state=SEED)
    model.fit(scaler.transform(x), y)
    return {i: model.predict_proba(scaler.transform(windows[i].matrix))[:, 1] for i in test_ids}


def window_stats(record, win, prob, threshold):
    keep = win.labels >= 0
    y, p = win.labels[keep], prob[keep]
    pred = p >= threshold
    sens = float(np.mean(pred[y == 1])) if np.any(y == 1) else None
    spec = float(np.mean(~pred[y == 0])) if np.any(y == 0) else None
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return {"auroc": auc, "sensitivity": sens, "specificity": spec,
            "positive_windows": int(np.sum(y == 1)), "negative_windows": int(np.sum(y == 0))}


def detections(win, prob, threshold):
    ids = np.flatnonzero(prob >= threshold)
    groups = []
    for idx in ids:
        t = float(win.starts_seconds[idx])
        if not groups or t - groups[-1]["last_positive"] >= REARM_SECONDS + 8.0:
            groups.append({"alarm_time": t + 8.0, "last_positive": t})
        else:
            groups[-1]["last_positive"] = t
    return groups


def episode_stats(signal, win, prob, threshold):
    alarms = detections(win, prob, threshold)
    used, delays, false_alarms, duplicate_event_alarms = set(), [], 0, 0
    for alarm in alarms:
        in_event = [(j,e) for j,e in enumerate(signal.events) if e.start <= alarm["alarm_time"] < e.end]
        if not in_event:
            false_alarms += 1
        else:
            unmatched = [(j,e) for j,e in in_event if j not in used]
            if unmatched:
                j,e = min(unmatched,key=lambda x:x[1].start)
                used.add(j); delays.append(alarm["alarm_time"]-e.start)
            else:
                duplicate_event_alarms += 1
    event_seconds = sum(max(0.0, e.end-e.start) for e in signal.events)
    hours = max(0.0, signal.duration-event_seconds)/3600
    return {"record": signal.key, "episodes": len(signal.events), "hits": len(used),
            "false_alarms": false_alarms, "duplicate_event_alarms": duplicate_event_alarms,
            "background_hours": hours, "delays": delays}


def bootstrap_summary(rows, seed=SEED):
    rng = np.random.RandomState(seed)
    def aggregate(sample):
        npos = sum(r["positive_windows"] for r in sample)
        nneg = sum(r["negative_windows"] for r in sample)
        wins = [r for r in sample if r["auroc"] is not None]
        auroc = float(np.mean([r["auroc"] for r in wins])) if wins else None
        sensrows = [r["sensitivity"] for r in sample if r["sensitivity"] is not None]
        specrows = [r["specificity"] for r in sample if r["specificity"] is not None]
        return {"auroc_record_macro": auroc,
                "sensitivity_record_macro": float(np.mean(sensrows)) if sensrows else None,
                "specificity_record_macro": float(np.mean(specrows)) if specrows else None,
                "window_denominators": {"positive": npos, "negative": nneg}}
    point = aggregate(rows)
    ci = {}
    for key in ("auroc_record_macro", "sensitivity_record_macro", "specificity_record_macro"):
        vals = []
        for _ in range(BOOTSTRAPS):
            sample = [rows[i] for i in rng.randint(0, len(rows), len(rows))]
            value = aggregate(sample)[key]
            if value is not None: vals.append(value)
        ci[key] = [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else None
    point["ci95_record_bootstrap"] = ci
    point["records_with_metrics"] = len(rows)
    return point


def episode_summary(rows):
    rng = np.random.RandomState(SEED)
    def agg(sample):
        episodes = sum(r["episodes"] for r in sample); hits = sum(r["hits"] for r in sample)
        hours = sum(r["background_hours"] for r in sample); fp = sum(r["false_alarms"] for r in sample)
        delays = [d for r in sample for d in r["delays"]]
        return {"episodes": episodes, "detected": hits, "missed": episodes-hits,
                "sensitivity": hits/episodes if episodes else None,
                "false_alarms": fp, "background_hours": hours,
                "duplicate_alarms_during_events_ignored": sum(r["duplicate_event_alarms"] for r in sample),
                "false_alarms_per_hour": fp/hours if hours else None,
                "median_delay_seconds": float(np.median(delays)) if delays else None,
                "delay_hits_denominator": len(delays)}
    point = agg(rows); cis = {}
    for key in ("sensitivity", "false_alarms_per_hour", "median_delay_seconds"):
        vals=[]
        for _ in range(BOOTSTRAPS):
            sample=[rows[i] for i in rng.randint(0,len(rows),len(rows))]
            value=agg(sample)[key]
            if value is not None and np.isfinite(value): vals.append(value)
        cis[key]=[float(x) for x in np.percentile(vals,[2.5,97.5])] if vals else None
    point["ci95_record_bootstrap"] = cis
    point["records"] = len(rows)
    return point


def main():
    split=json.loads(SPLIT_FILE.read_text())
    records=[]; windows=[]; source_fs={}
    for ds in DATASETS:
        ids=split["datasets"][ds]["train"]+split["datasets"][ds]["validation"]
        for rid in ids:
            path=DATA_ROOT/ds/rid
            header=wfdb.rdheader(str(path)); fs=float(header.fs)
            source_fs.setdefault(ds,sorted(set()))
            if fs not in source_fs[ds]: source_fs[ds].append(fs)
            signal=load_record(ds,rid)
            if signal.fs != 250: raise RuntimeError(f"resampling failed for {ds}/{rid}: {signal.fs}")
            windows.append(extract_windows8(signal))
            records.append({"dataset":ds,"id":rid,"signal":signal})
            print(f"features {len(records)} {ds}/{rid} fs={fs}->250",flush=True)

    all_ids=list(range(len(records))); outer=balanced_folds(records,all_ids,OUTER_FOLDS,SEED)
    oof={}; fold_thresholds=[]
    for fold, test_ids in enumerate(outer):
        train_ids=sorted(set(all_ids)-set(test_ids))
        inner=balanced_folds(records,train_ids,INNER_FOLDS,SEED+fold+1)
        inner_oof={}
        for inner_test in inner:
            inner_train=sorted(set(train_ids)-set(inner_test))
            inner_oof.update(fit_predict(records,windows,inner_train,inner_test))
        candidates=np.unique(np.quantile(np.concatenate([inner_oof[i][windows[i].labels>=0] for i in train_ids]),np.linspace(.01,.99,99)))
        def youden(t):
            vals=[]
            for i in train_ids:
                y=windows[i].labels; p=inner_oof[i]; mask=y>=0; y=y[mask]; pred=p[mask]>=t
                if np.any(y==0) and np.any(y==1):
                    vals.append((np.mean(pred[y==1])+np.mean(~pred[y==0]))/2)
            return float(np.mean(vals)) if vals else -1
        threshold=max(candidates,key=lambda t:(youden(t),float(t))); fold_thresholds.append(float(threshold))
        oof.update(fit_predict(records,windows,train_ids,test_ids))
        print(f"outer {fold+1}/{OUTER_FOLDS}: {len(test_ids)} records, inner threshold={threshold:.6g}",flush=True)

    window_rows={}; episode_rows={}
    for i,rec in enumerate(records):
        threshold=fold_thresholds[next(k for k,ids in enumerate(outer) if i in ids)]
        window_rows.setdefault(rec["dataset"],[]).append(window_stats(rec,windows[i],oof[i],threshold))
        episode_rows.setdefault(rec["dataset"],[]).append(episode_stats(rec["signal"],windows[i],oof[i],threshold))
    result={"label":"exploratory, cross-validated, not held-out",
            "protocol":"Nested 5-fold record-level outer CV; 4-fold record-level inner CV chooses one Youden-J threshold per outer fold; 8 s windows, 4 s stride; old holdout records excluded",
            "class_rule":"positive rhythm annotations VT, VF, VFIB; ventricular flutter excluded; windows positive at >=4 s event overlap, boundary overlap ignored",
            "rearm_rule":"merge predicted-positive windows until 30 consecutive seconds without a positive; alarm timestamp is end of first positive 8 s window",
            "feature_names":list(FEATURE_NAMES),"source_fs_hz":source_fs,"target_fs_hz":250,
            "thresholds_by_outer_fold":fold_thresholds,"threshold_selection":"maximize mean per-record Youden J on inner OOF windows only",
            "records":sum(len(v) for v in window_rows.values()),"window_metrics_by_dataset":{},"episode_metrics_by_dataset":{}}
    for ds in DATASETS:
        eps=episode_rows.get(ds,[]); wins=window_rows.get(ds,[])
        result["window_metrics_by_dataset"][ds]=bootstrap_summary(wins)
        result["episode_metrics_by_dataset"][ds]=episode_summary(eps)
    result["episode_metrics_all_datasets"]=episode_summary([r for group in episode_rows.values() for r in group])
    OUT_FILE.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"wrote {OUT_FILE}",flush=True)


if __name__=="__main__": main()
