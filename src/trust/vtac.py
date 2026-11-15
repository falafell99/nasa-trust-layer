"""Загрузка VTaC v1.1: каналы по именам (ECG = всё кроме PLETH/ABP/RESP), 250 Гц, обрезка до 300 с ДО обработки.
Тип тревоги в файлах VTaC не указан [не подтверждено, что все события VT] -> в признаках 'unknown'."""
import os
import numpy as np
import pandas as pd
import wfdb
from .cinc import FS, N_ALARM, pick_channels

ROOT = "data/vtac"


def events(root=ROOT):
    s = pd.read_csv(os.path.join(root, "benchmark_data_split.csv"))
    l = pd.read_csv(os.path.join(root, "event_labels.csv"))
    m = s.merge(l, on=["record", "event"])
    m["y"] = (m.decision.astype(str).str.lower() == "true").astype(int)  # метка только для оценки
    return m


def load(record, event, root=ROOT):
    r = wfdb.rdrecord(os.path.join(root, "annotated_waveforms", record, event), sampto=N_ALARM)
    assert r.fs == FS and r.p_signal.shape[0] == N_ALARM, (event, r.fs, r.p_signal.shape)
    ecg_n, ppg_n = pick_channels(r.sig_name)
    idx = {n: i for i, n in enumerate(r.sig_name)}
    return dict(ecg=[r.p_signal[:, idx[n]] for n in ecg_n], ecg_names=ecg_n,
                ppg=(r.p_signal[:, idx[ppg_n]] if ppg_n else None))
