"""Exploratory leave-one-record-out VT/VF detector CV; never loads old holdout records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.detector.data import DATA_ROOT, load_record
from src.detector.evaluation import evaluate, score_record
from src.detector.features import FEATURE_NAMES, WindowFeatures, extract_windows
from src.detector.model import fit_model, predict_probability


SPLIT_PATH = Path("results/detector_split.json")
OUT = Path("results/detector_cv2_loco.json")
DATASETS = ("vfdb", "cudb", "mitdb")
TARGET_FP_H = (1.0, 10.0)
windows_by_record: list[WindowFeatures] = []


def rule_score(windows: WindowFeatures) -> np.ndarray:
    f = {name: windows.matrix[:, i] for i, name in enumerate(FEATURE_NAMES)}
    # A compact classical score: tachycardia, irregular RR, broadband VF,
    # or frequent threshold crossings can each raise suspicion.
    return np.maximum.reduce((
        (f["qrs_rate_bpm"] - 100.0) / 80.0,
        f["rr_cv"] / 0.30,
        (0.15 - f["spectral_concentration"]) / 0.10,
        f["threshold_crossings_per_second"] / 25.0,
    ))


def select_thresholds(records, windows_by_record, predictions):
    valid = np.concatenate([w.labels >= 0 for w in windows_by_record])
    values = np.concatenate(predictions)
    # A finite, deterministic sweep over out-of-fold scores. Selection is on
    # CV itself, so reported operating points are exploratory, not held out.
    thresholds = np.unique(np.quantile(values[valid], np.linspace(0.0, 1.0, 51)))
    evaluated = []
    for threshold in thresholds:
        point = []
        for record, windows, scores in zip(records, windows_by_record, predictions):
            point.append(score_record(record, windows, scores >= threshold))
        metrics = evaluate(point, bootstrap=0)
        evaluated.append((float(threshold), metrics))
    return {
        str(target): min(evaluated, key=lambda row: abs(row[1]["false_alarms_per_hour"] - target))
        for target in TARGET_FP_H
    }


def main():
    global windows_by_record
    split = json.loads(SPLIT_PATH.read_text())
    records = []
    folds = []
    for dataset in DATASETS:
        # Restrict this exploratory CV to the previous run's train+validation
        # records. Prior holdout records are excluded before any signal read.
        allowed = split["datasets"][dataset]["train"] + split["datasets"][dataset]["validation"]
        for record_id in allowed:
            record = load_record(dataset, record_id)
            windows = extract_windows(record)
            records.append(record)
            folds.append(windows)
            print(f"features {len(records)}/{sum(len(split['datasets'][d]['train']) + len(split['datasets'][d]['validation']) for d in DATASETS)} {record.key} windows={len(windows.labels)}", flush=True)

    predictions = [np.zeros(len(w.labels), dtype=float) for w in folds]
    for index, (held_record, held_windows) in enumerate(zip(records, folds)):
        training = [(r, w) for i, (r, w) in enumerate(zip(records, folds)) if i != index]
        model = fit_model(training)
        predictions[index] = predict_probability(model, held_windows.matrix)
        print(f"fold {index + 1}/{len(records)} held={held_record.key}", flush=True)

    rules = [rule_score(windows) for windows in folds]
    output = {
        "protocol": "exploratory, cross-validated, not held-out; leave-one-record-out on previous train+validation records only; prior holdout records excluded before signal loading",
        "datasets": list(DATASETS),
        "records": len(records),
        "episodes": sum(len(record.events) for record in records),
        "feature_names": list(FEATURE_NAMES),
        "threshold_selection": "two CV operating points chosen nearest to 1 and 10 false alarms/hour from 51 deterministic score quantiles; thresholds and metrics use the same OOF predictions, so optimistic selection bias is expected",
        "old_holdout": "excluded; no waveform or prediction from the previous holdout was read",
        "models": {},
    }
    for name, values in (("classical_rule", rules), ("logistic", predictions)):
        chosen = select_thresholds(records, folds, values)
        output["models"][name] = {
            "operating_points": {
                target: {
                    "threshold": threshold,
                    "metrics": evaluate([
                        score_record(record, windows, scores >= threshold)
                        for record, windows, scores in zip(records, folds, values)
                    ], bootstrap=2000),
                }
                for target, (threshold, metrics) in chosen.items()
            }
        }
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
