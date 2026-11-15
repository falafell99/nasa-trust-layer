"""Small CPU logistic model and fixed, record-balanced training procedure."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.detector.data import SignalRecord
from src.detector.features import FEATURE_NAMES, WindowFeatures


TRAIN_SEED = 42
MAX_WINDOWS_PER_CLASS_PER_RECORD = 1000
LOGISTIC_C = 0.1


def record_seed(key: str) -> int:
    digest = hashlib.sha256(f"{TRAIN_SEED}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def sample_record_windows(key: str, windows: WindowFeatures) -> tuple[np.ndarray, np.ndarray]:
    usable = windows.labels >= 0
    rng = np.random.RandomState(record_seed(key))
    chosen: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(usable & (windows.labels == label))
        if len(indices) > MAX_WINDOWS_PER_CLASS_PER_RECORD:
            indices = rng.choice(indices, MAX_WINDOWS_PER_CLASS_PER_RECORD, replace=False)
        if len(indices):
            chosen.append(windows.matrix[indices])
            labels.append(np.full(len(indices), label, dtype=np.int8))
    if not chosen:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int8)
    return np.concatenate(chosen), np.concatenate(labels)


def fit_model(training_records: list[tuple[SignalRecord, WindowFeatures]]) -> dict[str, Any]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for record, windows in training_records:
        x, y = sample_record_windows(record.key, windows)
        if len(y):
            features.append(x)
            labels.append(y)
    if not features:
        raise ValueError("No labelled training windows were found")
    x_train = np.concatenate(features)
    y_train = np.concatenate(labels)
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training split must contain both VT/VF and non-event windows")

    scaler = StandardScaler().fit(x_train)
    standardized = scaler.transform(x_train)
    classifier = LogisticRegression(
        C=LOGISTIC_C,
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        random_state=TRAIN_SEED,
    ).fit(standardized, y_train)
    return {
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "training_windows": int(len(y_train)),
        "training_positive_windows": int(y_train.sum()),
        "training_negative_windows": int((y_train == 0).sum()),
        "training_records": len(training_records),
        "seed": TRAIN_SEED,
        "max_windows_per_class_per_record": MAX_WINDOWS_PER_CLASS_PER_RECORD,
        "logistic_c": LOGISTIC_C,
        "class_weight": "balanced",
    }


def predict_probability(model: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    z = (matrix.astype(np.float64) - mean) / np.maximum(scale, 1e-12)
    logits = np.clip(z @ coefficients + float(model["intercept"]), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))
