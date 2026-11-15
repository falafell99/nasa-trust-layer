"""Local WFDB loading and record-level target-event windows."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

from src.detector.annotations import Episode, read_episodes


DATA_ROOT = Path("data/detector")
TARGET_FS = 250


@dataclass(frozen=True)
class EventSeconds:
    rhythm: str
    start: float
    end: float


@dataclass(frozen=True)
class SignalRecord:
    dataset: str
    record: str
    fs: int
    signal: np.ndarray
    events: tuple[EventSeconds, ...]

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.record}"

    @property
    def duration(self) -> float:
        return len(self.signal) / self.fs


def load_record(dataset: str, record: str, root: Path = DATA_ROOT) -> SignalRecord:
    """Load one lead and anti-alias to TARGET_FS before feature extraction."""
    path = root / dataset / record
    header = wfdb.rdheader(str(path))
    raw = wfdb.rdrecord(str(path), physical=True)
    if not np.isclose(float(header.fs), float(raw.fs)):
        raise ValueError(f"WFDB header/data sampling-rate mismatch for {dataset}/{record}: {header.fs} vs {raw.fs}")
    source_fs = int(round(float(raw.fs)))
    signal = np.asarray(raw.p_signal[:, 0], dtype=np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    episodes: list[Episode] = read_episodes(path, record)
    if dataset == "mitdb":
        episodes = [episode for episode in episodes if episode.rhythm == "VT"]
    events = tuple(
        EventSeconds(
            rhythm=episode.rhythm,
            start=episode.start / source_fs,
            end=episode.end / source_fs,
        )
        for episode in episodes
    )
    # The detector target is dangerous rhythm vs other, so adjacent VT/VF
    # annotation changes are one episode when there is no intervening gap.
    merged_events = []
    for event in sorted(events, key=lambda item: (item.start, item.end)):
        if merged_events and event.start <= merged_events[-1].end:
            previous = merged_events[-1]
            merged_events[-1] = EventSeconds("VT/VF", previous.start, max(previous.end, event.end))
        else:
            merged_events.append(event)
    events = tuple(merged_events)

    if source_fs != TARGET_FS:
        common = gcd(source_fs, TARGET_FS)
        signal = resample_poly(signal, TARGET_FS // common, source_fs // common)

    if not len(signal) or not np.isfinite(signal).all():
        raise ValueError(f"Invalid ECG signal in {dataset}/{record}")
    if int(header.sig_len) <= 0:
        raise ValueError(f"Invalid WFDB header for {dataset}/{record}")
    return SignalRecord(dataset, record, TARGET_FS, signal, events)


def label_windows(
    starts: np.ndarray,
    window_seconds: float,
    events: tuple[EventSeconds, ...],
    min_positive_overlap_seconds: float = 0.5,
) -> np.ndarray:
    """Label windows by interval overlap; ignore ambiguous boundary windows."""
    labels = np.zeros(len(starts), dtype=np.int8)
    ends = starts + window_seconds
    overlap = np.zeros(len(starts), dtype=np.float32)
    for event in events:
        event_overlap = np.maximum(0.0, np.minimum(ends, event.end) - np.maximum(starts, event.start))
        overlap = np.maximum(overlap, event_overlap)
    labels[overlap >= min_positive_overlap_seconds] = 1
    labels[(overlap > 0) & (overlap < min_positive_overlap_seconds)] = -1
    return labels
