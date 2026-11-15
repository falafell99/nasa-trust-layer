"""Загрузка CinC 2015: выбор каналов по имени, обрезка до момента тревоги (300 с) ДО обработки."""
import os
import numpy as np
import wfdb

ROOT = "data/cinc2015/x/training"
FS = 250
N_ALARM = 300 * FS  # 75 000 отсчётов: данные после тревоги не читаем (AGENTS.md п.4)
NON_ECG = {"PLETH", "ABP", "RESP"}
ECG_PREF = ["II", "V", "MCL", "MCL1", "III", "I", "aVF", "aVR", "aVL"]


def read_alarms(root=ROOT):
    out = {}
    for line in open(os.path.join(root, "ALARMS")).read().splitlines():
        rec, typ, lab = [x.strip() for x in line.split(",")]
        out[rec] = (typ, int(lab))  # метка читается ТОЛЬКО для оценки, не для признаков
    return out


def channel_names(root, rec):
    """Только из .hea, сигнал не читаем."""
    ls = open(os.path.join(root, rec + ".hea")).read().splitlines()
    n = int(ls[0].split()[1])
    return [l.split()[-1] for l in ls[1:1 + n]]


def pick_channels(names):
    ecg = [n for n in names if n not in NON_ECG]
    ecg.sort(key=lambda n: ECG_PREF.index(n) if n in ECG_PREF else 99)
    return ecg, ("PLETH" if "PLETH" in names else None)


def load(rec, root=ROOT):
    """Возвращает dict(ecg=[arr,...], ecg_names, ppg|None) за первые 300 с."""
    r = wfdb.rdrecord(os.path.join(root, rec), sampto=N_ALARM)
    assert r.fs == FS and r.p_signal.shape[0] == N_ALARM
    ecg_n, ppg_n = pick_channels(r.sig_name)
    idx = {n: i for i, n in enumerate(r.sig_name)}
    return dict(
        ecg=[r.p_signal[:, idx[n]] for n in ecg_n], ecg_names=ecg_n,
        ppg=(r.p_signal[:, idx[ppg_n]] if ppg_n else None),
    )
