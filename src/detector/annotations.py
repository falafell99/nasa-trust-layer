"""Read VT/VF rhythm intervals from PhysioNet WFDB annotations.

Only rhythm-change annotations are used. Beat labels are intentionally ignored.
Intervals are half-open sample ranges: [start, end).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import wfdb


@dataclass(frozen=True)
class Episode:
    record: str
    rhythm: str
    start: int
    end: int

    @property
    def duration_samples(self) -> int:
        return self.end - self.start


def normalize_rhythm(aux_note: str) -> str | None:
    """Map PhysioNet rhythm notes to the two target classes."""
    note = aux_note.replace("\x00", "").strip().upper()
    if note.startswith("("):
        note = note[1:].strip()
    if note in {"VT"}:
        return "VT"
    if note in {"VF", "VFIB"}:
        return "VF"
    return None


def read_episodes(record_path: str | Path, record: str, merge_gap_seconds: float = 1.0) -> list[Episode]:
    """Extract target rhythm intervals, closing each at the next rhythm change.

    Repeated same-class annotations separated by at most one second are merged;
    this handles burst-like repeated VT/VF rhythm tags without counting every
    tag as an independent episode.
    """
    record_path = str(record_path)
    header = wfdb.rdheader(record_path)
    annotation = wfdb.rdann(record_path, "atr")
    changes: list[tuple[int, str | None]] = []
    for sample, symbol, aux_note in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        # WFDB rhythm changes use '+' with a parenthesized auxiliary note.
        if symbol != "+" or not aux_note.strip():
            continue
        note = aux_note.replace("\x00", "").strip()
        if not note.startswith("("):
            continue
        changes.append((int(sample), normalize_rhythm(note)))

    episodes: list[Episode] = []
    for index, (start, rhythm) in enumerate(changes):
        if rhythm is None:
            continue
        end = changes[index + 1][0] if index + 1 < len(changes) else int(header.sig_len)
        start = max(0, min(start, int(header.sig_len)))
        end = max(start, min(end, int(header.sig_len)))
        if end > start:
            episodes.append(Episode(record, rhythm, start, end))

    # Merge only adjacent annotations of the same rhythm; intervening non-target
    # rhythms already terminate an episode and are not bridged.
    max_gap = int(round(merge_gap_seconds * float(header.fs)))
    merged: list[Episode] = []
    for episode in episodes:
        if merged and merged[-1].rhythm == episode.rhythm and episode.start - merged[-1].end <= max_gap:
            previous = merged[-1]
            merged[-1] = Episode(record, previous.rhythm, previous.start, episode.end)
        else:
            merged.append(episode)
    return merged


def episode_dicts(episodes: list[Episode]) -> list[dict[str, int | str]]:
    return [asdict(episode) for episode in episodes]
