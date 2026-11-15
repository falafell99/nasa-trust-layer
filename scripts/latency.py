"""Задержка/память/размер для V5 (одна тревога = 300 с, 3-4 канала, 250 Гц). Один процесс, 1 поток. Данные: 100 событий VTaC validation (метки не используются)."""
import sys, os, json, time, resource, platform, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, ".")
os.environ["OMP_NUM_THREADS"] = "1"
import lightgbm as lgb
from src.trust import vtac, feat2

base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS: байты -> МБ
cfg = json.load(open("models/v5_config.json"))
b = lgb.Booster(model_file="models/v5_lgbm.txt", params={"num_threads": 1})
ev = vtac.events(); ev = ev[ev.split == "val"].head(100)
t_load, t_feat, t_pred, t_all = [], [], [], []
for r, e in zip(ev.record, ev.event):
    t0 = time.perf_counter(); d = vtac.load(r, e); t1 = time.perf_counter()
    f = feat2.extract(d, "Ventricular_Tachycardia"); t2 = time.perf_counter()
    x = pd.DataFrame([f])[cfg["features"]]; p = b.predict(x); t3 = time.perf_counter()
    t_load.append(t1 - t0); t_feat.append(t2 - t1); t_pred.append(t3 - t2); t_all.append(t3 - t1)
q = lambda a: dict(median_ms=round(1000 * float(np.median(a)), 1), p95_ms=round(1000 * float(np.percentile(a, 95)), 1))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
hw = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"], capture_output=True, text=True).stdout.split("\n")
out = dict(model="V5 LightGBM + feat2 features", n_alarms=len(ev), alarm_window_s=300, threads=1,
           latency_features_plus_predict=q(t_all), latency_features=q(t_feat), latency_predict_only=q(t_pred), latency_file_read_wfdb=q(t_load),
           peak_rss_mb=round(peak, 0), rss_before_pipeline_mb=round(base, 0), model_size_kb=round(os.path.getsize("models/v5_lgbm.txt") / 1024, 0),
           hardware=dict(cpu=hw[0], cores=hw[1], ram_gb=round(int(hw[2]) / 1e9, 0) if hw[2] else None, os=platform.platform(), python=platform.python_version()),
           note="Задержка = от готового массива 300 с до p_true; не включает чтение файла. Замер на ноутбуке разработки, не на бортовом железе. Пиковая память включает numpy/scipy/neurokit2/lightgbm.")
json.dump(out, open("results/latency.json", "w"), indent=1); print(json.dumps(out, indent=1))
