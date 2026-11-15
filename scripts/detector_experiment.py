"""Run validation first, then (once) evaluate the frozen detector on hold-out."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector.data import DATA_ROOT, SignalRecord, load_record
from src.detector.evaluation import RecordScore, evaluate, evaluate_by_dataset, score_record
from src.detector.features import (
    FEATURE_NAMES,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
    WindowFeatures,
    extract_windows,
)
from src.detector.model import fit_model, predict_probability


SPLIT_PATH = Path("results/detector_split.json")
VALIDATION_PATH = Path("results/detector_validation.json")
HOLDOUT_PATH = Path("results/detector_holdout.json")
HOLDOUT_ATTEMPT_PATH = Path("results/detector_holdout_attempted.json")
MIN_POSITIVE_OVERLAP_SECONDS = 0.5
MAX_VALIDATION_FALSE_ALARMS_PER_HOUR = 1.0
THRESHOLD_CANDIDATE_QUANTILES = 101

# Fixed before validation: a simple fast/regular QRS rule for VT, plus a
# high-entropy, high-frequency rule for VF-like activity. These are a baseline,
# not a clinical operating point.
RULE_CONFIG = {
    "vt_rate_bpm_min": 150.0,
    "vt_rr_cv_max": 0.35,
    "vf_spectral_entropy_min": 0.82,
    "vf_power_15_40hz_min": 0.35,
    "vf_qrs_rate_bpm_max": 150.0,
}


def load_split() -> dict[str, Any]:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def split_digest() -> str:
    return hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()


def load_records(split: dict[str, Any], part: str) -> list[tuple[SignalRecord, WindowFeatures]]:
    loaded: list[tuple[SignalRecord, WindowFeatures]] = []
    for dataset in ("vfdb", "cudb", "mitdb"):
        names = split["datasets"][dataset][part]
        print(f"{part}: {dataset} {len(names)} records", flush=True)
        for index, name in enumerate(names, start=1):
            record = load_record(dataset, name, DATA_ROOT)
            windows = extract_windows(record)
            loaded.append((record, windows))
            if index % 5 == 0 or index == len(names):
                print(f"  {dataset}: {index}/{len(names)}", flush=True)
    return loaded


def rule_predictions(windows: WindowFeatures) -> np.ndarray:
    column = {name: i for i, name in enumerate(FEATURE_NAMES)}
    x = windows.matrix
    vt = (
        (x[:, column["qrs_rate_bpm"]] >= RULE_CONFIG["vt_rate_bpm_min"])
        & (x[:, column["rr_cv"]] <= RULE_CONFIG["vt_rr_cv_max"])
        & (x[:, column["rr_median_seconds"]] > 0)
    )
    vf = (
        (x[:, column["spectral_entropy"]] >= RULE_CONFIG["vf_spectral_entropy_min"])
        & (x[:, column["power_15_40hz"]] >= RULE_CONFIG["vf_power_15_40hz_min"])
        & (x[:, column["qrs_rate_bpm"]] < RULE_CONFIG["vf_qrs_rate_bpm_max"])
    )
    return vt | vf


def scores_for_probability(
    records: list[tuple[SignalRecord, WindowFeatures]],
    model: dict[str, Any],
    threshold: float,
) -> list[RecordScore]:
    output = []
    for record, windows in records:
        probabilities = predict_probability(model, windows.matrix)
        output.append(score_record(record, windows, probabilities >= threshold))
    return output


def scores_for_rule(records: list[tuple[SignalRecord, WindowFeatures]]) -> list[RecordScore]:
    return [score_record(record, windows, rule_predictions(windows)) for record, windows in records]


def tune_threshold(
    records: list[tuple[SignalRecord, WindowFeatures]],
    model: dict[str, Any],
) -> tuple[float, dict[str, object]]:
    probabilities = [predict_probability(model, windows.matrix) for _, windows in records]
    nonempty = [values for values in probabilities if len(values)]
    if not nonempty:
        raise ValueError("Validation split has no windows")
    all_probability = np.concatenate(nonempty)
    candidates = np.unique(
        np.concatenate(
            (
                np.quantile(all_probability, np.linspace(0.0, 1.0, THRESHOLD_CANDIDATE_QUANTILES)),
                np.linspace(0.0, 1.0, 101),
            )
        )
    )
    grid: list[tuple[float, dict[str, object]]] = []
    for threshold in candidates:
        current: list[RecordScore] = []
        for (record, windows), values in zip(records, probabilities):
            current.append(score_record(record, windows, values >= threshold))
        metrics = evaluate(current, bootstrap=0)
        grid.append((float(threshold), metrics))

    feasible = [
        item
        for item in grid
        if item[1]["false_alarms_per_hour"] is not None
        and float(item[1]["false_alarms_per_hour"]) <= MAX_VALIDATION_FALSE_ALARMS_PER_HOUR
    ]
    if feasible:
        selected, metrics = max(
            feasible,
            key=lambda item: (
                float(item[1]["sensitivity"] or 0.0),
                -float(item[1]["latency_median_seconds"] or float("inf")),
                -float(item[1]["false_alarms_per_hour"] or 0.0),
                item[0],
            ),
        )
        reason = "maximum episode sensitivity subject to validation FP/h <= 1.0"
    else:
        selected, metrics = min(
            grid,
            key=lambda item: (
                float(item[1]["false_alarms_per_hour"] or float("inf")),
                -float(item[1]["sensitivity"] or 0.0),
                item[0],
            ),
        )
        reason = "no threshold met validation FP/h target; minimum FP/h then maximum sensitivity"
    return selected, {
        "criterion": reason,
        "validation_fp_per_hour_target": MAX_VALIDATION_FALSE_ALARMS_PER_HOUR,
        "candidate_count": int(len(candidates)),
        "selected_metrics_without_ci": metrics,
    }


def run_validation() -> None:
    if VALIDATION_PATH.exists():
        raise SystemExit(f"{VALIDATION_PATH} already exists; refusing to silently overwrite the frozen model.")
    split = load_split()
    training = load_records(split, "train")
    validation = load_records(split, "validation")
    model = fit_model(training)
    threshold, selection = tune_threshold(validation, model)

    baseline_scores = scores_for_rule(validation)
    logistic_scores = scores_for_probability(validation, model, threshold)
    payload = {
        "stage": "validation",
        "source_commit": git_head(),
        "split_sha256": split_digest(),
        "protocol": {
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "target_sample_rate_hz": 250,
            "positive_window_overlap_seconds": MIN_POSITIVE_OVERLAP_SECONDS,
            "detection_merge_gap_seconds": 1.0,
            "feature_names": list(FEATURE_NAMES),
            "rule": RULE_CONFIG,
            "threshold_selection": selection,
            "holdout_policy": "No holdout records are loaded in the validation stage. Freeze this model and threshold before running --stage holdout.",
        },
        "frozen_threshold": threshold,
        "logistic_model": model,
        "validation": {
            "baseline_rule": {
                "overall": evaluate(baseline_scores),
                "by_dataset": evaluate_by_dataset(baseline_scores),
            },
            "logistic": {
                "overall": evaluate(logistic_scores),
                "by_dataset": evaluate_by_dataset(logistic_scores),
            },
        },
    }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "threshold": threshold,
        "baseline_validation": payload["validation"]["baseline_rule"]["overall"],
        "logistic_validation": payload["validation"]["logistic"]["overall"],
    }, indent=2))


def run_holdout() -> None:
    if HOLDOUT_PATH.exists():
        raise SystemExit(f"{HOLDOUT_PATH} already exists; refusing a second holdout evaluation.")
    if HOLDOUT_ATTEMPT_PATH.exists():
        raise SystemExit(f"{HOLDOUT_ATTEMPT_PATH} exists; holdout was already opened or attempted.")
    if not VALIDATION_PATH.exists():
        raise SystemExit("Run --stage validation and commit its frozen threshold before opening holdout.")
    frozen = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    if frozen.get("stage") != "validation" or frozen.get("frozen_threshold") is None:
        raise SystemExit("Validation model/threshold is missing or malformed.")
    split = load_split()
    if frozen.get("split_sha256") != split_digest():
        raise SystemExit("The frozen split changed after validation; refusing to open holdout.")

    # The sentinel is written before reading any holdout signal. If execution
    # fails after this point, it still cannot silently run a second time.
    HOLDOUT_ATTEMPT_PATH.write_text(
        json.dumps({"stage": "holdout_started", "commit": git_head(), "split_sha256": split_digest()}, indent=2) + "\n",
        encoding="utf-8",
    )

    records = load_records(split, "holdout")
    threshold = float(frozen["frozen_threshold"])
    model = frozen["logistic_model"]
    baseline_scores = scores_for_rule(records)
    logistic_scores = scores_for_probability(records, model, threshold)
    payload = {
        "stage": "holdout",
        "frozen_validation_commit": git_head(),
        "split_sha256": split_digest(),
        "threshold_from_validation": threshold,
        "baseline_rule": {
            "overall": evaluate(baseline_scores),
            "by_dataset": evaluate_by_dataset(baseline_scores),
        },
        "logistic": {
            "overall": evaluate(logistic_scores),
            "by_dataset": evaluate_by_dataset(logistic_scores),
        },
        "holdout_policy": "Single run; no threshold changes or model selection from these results.",
    }
    HOLDOUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "threshold": threshold,
        "baseline_holdout": payload["baseline_rule"]["overall"],
        "logistic_holdout": payload["logistic"]["overall"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "holdout"), required=True)
    args = parser.parse_args()
    if args.stage == "validation":
        run_validation()
    else:
        run_holdout()


if __name__ == "__main__":
    main()
