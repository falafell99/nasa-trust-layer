"""Загрузка VTaC v1.1: каналы по именам (ECG = всё кроме PLETH/ABP/RESP), 250 Гц, обрезка до 300 с ДО обработки.
VTaC = только тревоги VT (по docs/11 лида) -> тип 'Ventricular_Tachycardia'."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from .cinc import FS, N_ALARM, pick_channels

ROOT = str(Path(__file__).resolve().parents[2] / "demo_data" / "vtac")


def events(root=ROOT):
    s = pd.read_csv(os.path.join(root, "benchmark_data_split.csv"))
    l = pd.read_csv(os.path.join(root, "event_labels.csv"))
    assert s.event.is_unique and l.event.is_unique, "duplicate events"
    m = s.merge(l, on=["record", "event"], validate="one_to_one")
    assert len(m) == len(s) == len(l), (len(m), len(s), len(l))
    dec = m.decision.astype(str).str.lower()
    assert dec.isin(["true", "false"]).all(), "unexpected decision values: %s" % sorted(set(dec) - {"true", "false"})  # не превращаем неожиданное в 0
    m["y"] = (dec == "true").astype(int)  # метка только для оценки
    return m


def load(record, event, root=ROOT):
    r = wfdb.rdrecord(os.path.join(root, "annotated_waveforms", record, event), sampto=N_ALARM)
    assert r.fs == FS and r.p_signal.shape[0] == N_ALARM, (event, r.fs, r.p_signal.shape)
    ecg_n, ppg_n = pick_channels(r.sig_name)
    idx = {n: i for i, n in enumerate(r.sig_name)}
    return dict(ecg=[r.p_signal[:, idx[n]] for n in ecg_n], ecg_names=ecg_n,
                ppg=(r.p_signal[:, idx[ppg_n]] if ppg_n else None))
