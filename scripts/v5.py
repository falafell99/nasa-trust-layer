"""V5 = LightGBM на feat2 (без CNN). Обучение: CinC train+val (600) + VTaC train (4060); порог T одним числом на VTaC val (495).
freeze: обучает, выбирает T, пишет models/v5_*.  test: ОДИН прогон на VTaC test (482) + база (keep-all, ECG-only V0, V3). Тип VTaC = VT (docs/11).
Правило T: наименьший порог, при котором на VTaC val доля настоящих в красной >= 0.85 (задано заранее)."""
import sys, json, subprocess
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, ".")
from scripts.cv_gbm import models
from src.trust.cinc import ECG_PREF  # noqa
META = ["split", "record", "group", "y", "alarm_type", "ds"]
PREC_TARGET = 0.85


def data():
    c = pd.read_csv("data/feat2_cinc_train-validation.csv").assign(ds="cinc")
    v = pd.read_csv("data/feat2_vtac_train-val.csv").assign(ds="vtac")
    tr = pd.concat([c, v[v.split == "train"]], ignore_index=True)
    va = v[v.split == "val"].reset_index(drop=True)
    feats = [x for x in tr.columns if x not in META and tr[x].notna().sum() > 0.02 * len(tr)]
    return tr, va, feats


def metrics(y, g, red, B=2000, seed=0):
    def m(r, yy):
        a = ((yy == 1) & r).sum(); n = r.sum()
        return np.array([a / n if n else np.nan, a / max(1, (yy == 1).sum()), n / len(yy)])
    pt = m(red, y)
    rng = np.random.RandomState(seed); ug, inv = np.unique(g, return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(ug))]
    bs = []
    for _ in range(B):
        s = np.concatenate([idx[i] for i in rng.randint(0, len(ug), len(ug))]); bs.append(m(red[s], y[s]))
    bs = np.array(bs); lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    a, n = int(((y == 1) & red).sum()), int(red.sum())
    return dict(true_red=a, false_red=n - a, red_n=n, n=len(y), true_total=int((y == 1).sum()),
                precision=pt[0], precision_ci=[lo[0], hi[0]], recall=pt[1], recall_ci=[lo[1], hi[1]], coverage=pt[2], coverage_ci=[lo[2], hi[2]])


def job(a):
    import src.trust.model as M
    from src.trust import vtac
    from src.trust.features import extract as ex3
    from src.trust.pipeline import verify
    rec, ev = a; d = vtac.load(rec, ev); f = ex3(d, "Ventricular_Tachycardia"); p3 = M.predict(M.load(), f)
    seg = {"ecg1": d["ecg"][0], "ecg2": d["ecg"][1] if len(d["ecg"]) > 1 else None, "ppg": None, "alarm_type": "Ventricular_Tachycardia", "fs": 250}
    return float(f["q"]), p3, int(verify(seg))


if __name__ == "__main__":
    tr, va, feats = data()
    if sys.argv[1] == "freeze":
        m = models()["lgbm"]().fit(tr[feats], tr.y)
        p = m.predict_proba(va[feats])[:, 1]
        cand = np.sort(np.unique(p))
        ok = [t for t in cand if (va.y[p >= t].mean() if (p >= t).sum() >= 20 else 0) >= PREC_TARGET]
        T = float(min(ok))
        m.booster_.save_model("models/v5_lgbm.txt")
        mv = metrics(va.y.values, va.group.values, p >= T)
        json.dump(dict(model="LightGBM on feat2 (62 features, VTaC type=VT)", features=feats, T=T, rule="lowest T with VTaC-val precision >= %.2f and red_n >= 20" % PREC_TARGET,
                       train="CinC train+val 600 + VTaC train 4060", val_report=mv), open("models/v5_config.json", "w"), indent=1)
        print("T=%.4f" % T, json.dumps(mv))
    else:  # test: один раз
        import src.trust.model as M
        from src.trust import vtac
        from src.trust.features import extract as ex3
        from src.trust.pipeline import verify
        from concurrent.futures import ProcessPoolExecutor
        cfg = json.load(open("models/v5_config.json"))
        te = pd.read_csv("data/feat2_vtac_test.csv"); y, g = te.y.values, te.group.values
        p = lgb.Booster(model_file="models/v5_lgbm.txt").predict(te[cfg["features"]])
        with ProcessPoolExecutor(16) as e:
            R = np.array(list(e.map(job, list(zip(te.group, te.record)), chunksize=8)))
        q3, p3, v0 = R[:, 0], R[:, 1], R[:, 2]
        meth = {"keep_all": np.ones(len(y), bool), "ECG_only_V0": v0 == 1, "V3_frozen_with_quality_gate": (q3 >= 0.6) & (p3 >= 0.5),
                "V3_no_quality_gate": p3 >= 0.5, "V5": p >= cfg["T"]}
        out = {"n": len(y), "patients": int(len(np.unique(g))), "true": int(y.sum()), "frozen": cfg["T"],
               "head": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(), "methods": {k: metrics(y, g, v) for k, v in meth.items()}}
        json.dump(out, open("results/v5_vtac_test.json", "w"), indent=1)
        for k, m in out["methods"].items():
            print("%-30s red %3d (true %3d / false %3d) | precision %.3f [%.3f,%.3f] | recall %.3f [%.3f,%.3f] | coverage %.3f [%.3f,%.3f]" % (
                k, m["red_n"], m["true_red"], m["false_red"], m["precision"], *m["precision_ci"], m["recall"], *m["recall_ci"], m["coverage"], *m["coverage_ci"]))
