"""Create the record-level split before any detector results are computed."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detector.annotations import read_episodes


ROOT = Path("data/detector")
OUTPUT = Path("results/detector_split.json")
SEED = 42
RATIOS = {"train": 0.50, "validation": 0.25, "holdout": 0.25}


def stable_order(names: list[str], dataset: str, stratum: str) -> list[str]:
    salt = f"{SEED}:{dataset}:{stratum}".encode()
    return sorted(names, key=lambda name: hashlib.sha256(salt + name.encode()).hexdigest())


def partition(names: list[str]) -> dict[str, list[str]]:
    n = len(names)
    if not n:
        return {key: [] for key in RATIOS}
    n_train = max(1, int(round(n * RATIOS["train"])))
    n_validation = max(1, int(round(n * RATIOS["validation"]))) if n >= 3 else 0
    if n_train + n_validation >= n:
        n_validation = max(0, n - n_train - 1)
    n_holdout = n - n_train - n_validation
    if n_holdout == 0:
        n_holdout = 1
        if n_train > 1:
            n_train -= 1
        elif n_validation > 0:
            n_validation -= 1
    ordered = names
    a = n_train
    b = a + n_validation
    return {"train": ordered[:a], "validation": ordered[a:b], "holdout": ordered[b:]}


def main() -> None:
    split: dict[str, dict[str, list[str]]] = {}
    metadata: dict[str, dict[str, dict[str, int]]] = {}
    for dataset in ("vfdb", "cudb", "mitdb"):
        directory = ROOT / dataset
        records = sorted(path.stem for path in directory.glob("*.hea"))
        if not records:
            raise SystemExit(f"No downloaded WFDB headers in {directory}; run the dataset download first.")
        positives: list[str] = []
        negatives: list[str] = []
        per_record: dict[str, dict[str, int]] = {}
        for record in records:
            record_path = directory / record
            episodes = read_episodes(record_path, record)
            counts = Counter(episode.rhythm for episode in episodes)
            per_record[record] = {"VT": counts["VT"], "VF": counts["VF"]}
            (positives if episodes else negatives).append(record)

        positive_order = stable_order(positives, dataset, "positive")
        negative_order = stable_order(negatives, dataset, "negative")
        p_split = partition(positive_order)
        n_split = partition(negative_order)
        split[dataset] = {
            key: sorted(p_split[key] + n_split[key])
            for key in RATIOS
        }
        metadata[dataset] = per_record

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "record-level 50/25/25 split, stratified by dataset and presence of VT/VF rhythm annotations",
        "seed": SEED,
        "ratios": RATIOS,
        "holdout_policy": "Do not compute or inspect holdout predictions until model and threshold are frozen from training/validation.",
        "annotation_merge_gap_seconds": 1.0,
        "datasets": split,
        "annotation_counts_for_stratification_only": metadata,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    for dataset, groups in split.items():
        print(dataset, {name: len(records) for name, records in groups.items()})
        print("  held-out records:", ", ".join(groups["holdout"]))


if __name__ == "__main__":
    main()
