"""Episode-level scoring with record-cluster bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.detector.data import EventSeconds, SignalRecord
from src.detector.features import WINDOW_SECONDS, WindowFeatures


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 0
DETECTION_MERGE_GAP_SECONDS = 1.0


@dataclass(frozen=True)
class Detection:
    alarm_time: float
    support_start: float
    support_end: float


@dataclass(frozen=True)
class RecordScore:
    key: str
    dataset: str
    episode_count: int
    hits: int
    false_alarms: int
    non_event_hours: float
    latencies: tuple[float, ...]


def make_detections(starts: np.ndarray, positive: np.ndarray) -> list[Detection]:
    indices = np.flatnonzero(positive)
    if not len(indices):
        return []
    detections: list[Detection] = []
    gap = DETECTION_MERGE_GAP_SECONDS
    start = float(starts[indices[0]])
    alarm_time = start + WINDOW_SECONDS
    support_end = alarm_time
    for index in indices[1:]:
        window_start = float(starts[index])
        window_end = window_start + WINDOW_SECONDS
        if window_start <= support_end + gap:
            support_end = max(support_end, window_end)
        else:
            detections.append(Detection(alarm_time, start, support_end))
            start, alarm_time, support_end = window_start, window_end, window_end
    detections.append(Detection(alarm_time, start, support_end))
    return detections


def score_record(record: SignalRecord, windows: WindowFeatures, positive: np.ndarray) -> RecordScore:
    detections = make_detections(windows.starts_seconds, positive)
    used: set[int] = set()
    latencies: list[float] = []
    for detection in detections:
        eligible = [
            (index, event)
            for index, event in enumerate(record.events)
            if index not in used
            and detection.alarm_time >= event.start
            and detection.support_start <= event.end
            and detection.support_end >= event.start
        ]
        if eligible:
            index, event = min(eligible, key=lambda pair: (pair[1].start, pair[0]))
            used.add(index)
            latencies.append(max(0.0, detection.alarm_time - event.start))

    event_seconds = sum(max(0.0, event.end - event.start) for event in record.events)
    non_event_hours = max(0.0, record.duration - event_seconds) / 3600.0
    return RecordScore(
        key=record.key,
        dataset=record.dataset,
        episode_count=len(record.events),
        hits=len(used),
        false_alarms=len(detections) - len(used),
        non_event_hours=non_event_hours,
        latencies=tuple(latencies),
    )


def _aggregate(scores: list[RecordScore]) -> dict[str, float | int | None]:
    episodes = sum(score.episode_count for score in scores)
    hits = sum(score.hits for score in scores)
    false_alarms = sum(score.false_alarms for score in scores)
    non_event_hours = sum(score.non_event_hours for score in scores)
    latencies = [latency for score in scores for latency in score.latencies]
    return {
        "episodes_detected": hits,
        "episodes_total": episodes,
        "sensitivity": hits / episodes if episodes else None,
        "false_alarms": false_alarms,
        "non_event_hours": non_event_hours,
        "false_alarms_per_hour": false_alarms / non_event_hours if non_event_hours else None,
        "latency_median_seconds": float(np.median(latencies)) if latencies else None,
        "latency_p90_seconds": float(np.percentile(latencies, 90)) if latencies else None,
    }


def evaluate(scores: list[RecordScore], bootstrap: int = BOOTSTRAP_REPLICATES) -> dict[str, object]:
    point = _aggregate(scores)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {
        "sensitivity": [],
        "false_alarms_per_hour": [],
        "latency_median_seconds": [],
        "latency_p90_seconds": [],
    }
    if scores:
        for _ in range(bootstrap):
            indices = rng.randint(0, len(scores), len(scores))
            metric = _aggregate([scores[i] for i in indices])
            for key in samples:
                value = metric[key]
                if value is not None and np.isfinite(value):
                    samples[key].append(float(value))

    for key, values in samples.items():
        point[f"{key}_ci95"] = (
            [float(value) for value in np.percentile(values, [2.5, 97.5])]
            if values
            else None
        )
    point["bootstrap"] = {"unit": "record", "replicates": bootstrap, "seed": BOOTSTRAP_SEED}
    point["records"] = len(scores)
    return point


def evaluate_by_dataset(scores: Iterable[RecordScore]) -> dict[str, dict[str, object]]:
    scores = list(scores)
    return {
        dataset: evaluate([score for score in scores if score.dataset == dataset])
        for dataset in sorted({score.dataset for score in scores})
    }
