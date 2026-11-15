"""ФИНАЛ: один прогон на VTaC test (482) и CinC hold-out (150, «second look», без подбора). Замороженное: models/v4_config.json, models/v3_logreg.json.
Требует заранее посчитанных data/feat2_vtac_test.csv и data/feat2_cinc_holdout.csv (make_feat2.py). Результат: results/v4_final.json"""
import sys, json, subprocess
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import lightgbm as lgb
sys.path.insert(0, ".")
from src.trust import model as M, vtac
from src.trust.cinc import load, read_alarms
from src.trust.features import extract as extract_v3
from scripts.cv_gbm import models, load_all

cfg = json.load(open("models/v4_config.json"))
feats, T, MODEL = cfg["features"], cfg["T"], cfg["model"]
AL = read_alarms()
V3P = M.load()


def v3_one(a):
    ds, key = a
    if ds == "cinc":
        d = load(key); t = AL[key][0]
    else:
        d = vtac.load(*key); t = "unknown"
    f = extract_v3(d, t)
    return M.decide_v3(f["q"], M.predict(V3P, f)) == "red", float(f["q"]), M.predict(V3P, f)


def boot(y, g, red, fn, B=2000, seed=0):
    rng = np.random.RandomState(seed); ug, inv = np.unique(g, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(ug))]
    vals = []
    for _ in range(B):
        s = np.concatenate([idx[i] for i in rng.randint(0, len(ug), len(ug))])
        v = fn(red[s], y[s]); vals.append(v)
    v = np.array(vals, float); v = v[~np.isnan(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [None, None]


def cell(y, g, red):
    a, b, c, d = int(((y == 1) & red).sum()), int(((y == 1) & ~red).sum()), int(((y == 0) & red).sum()), int(((y == 0) & ~red).sum())
    sr = lambda r, yy: ((yy == 1) & r).sum() / r.sum() if r.sum() else np.nan
    sy = lambda r, yy: ((yy == 1) & ~r).sum() / (~r).sum() if (~r).sum() else np.nan
    sz = lambda r, yy: r.mean()
    tr = lambda r, yy: ((yy == 1) & r).sum() / max(1, (yy == 1).sum())
    return dict(true_red=a, true_yellow=b, false_red=c, false_yellow=d, red_n=a + c, red_share_of_all=(a + c) / len(y),
                true_share_red=(a / (a + c) if a + c else None), true_share_red_ci=boot(y, g, red, sr),
                true_share_yellow=(b / (b + d) if b + d else None), true_share_yellow_ci=boot(y, g, red, sy),
                red_group_share_ci=boot(y, g, red, sz), true_alarms_in_red_frac=a / max(1, a + b), true_alarms_in_red_ci=boot(y, g, red, tr))


if __name__ == "__main__":
    out = {"frozen": {"v4": "models/v4_config.json", "v3": "1dad545", "T": T, "model": MODEL}, "head": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(), "sets": {}}
    tv, _ = load_all(True)
    booster = lgb.Booster(model_file="models/v4_lgbm.txt")  # замороженная модель
    cinc_tv = tv[tv.ds == "cinc"]
    rate = cinc_tv.groupby("alarm_type").y.mean()
    for name, path, ds in (("vtac_test", "data/feat2_vtac_test.csv", "vtac"), ("cinc_holdout_second_look", "data/feat2_cinc_holdout.csv", "cinc")):
        df = pd.read_csv(path)
        y, g = df.y.values, df.group.values
        p = booster.predict(df[feats])
        if cfg.get("blend_cnn"):
            p = 0.5 * p + 0.5 * np.load("results/pcnn_%s.npy" % name)  # считается отдельным процессом (scripts/cnn_final.py): lightgbm и torch в одном процессе падают (segfault)
        keys = [(ds, r) if ds == "cinc" else (ds, (gg, r)) for r, gg in zip(df.record, df.group)]
        with ProcessPoolExecutor(16) as ex:
            v3 = list(ex.map(v3_one, keys, chunksize=8))
        meth = {"keep_all": np.ones(len(y), bool), "V3_frozen": np.array([r[0] for r in v3]), "V4": p >= T}
        if ds == "cinc":
            meth["type_only"] = df.alarm_type.map(rate).values >= 0.5
        out["sets"][name] = {"n": len(y), "true": int(y.sum()), "false": int((1 - y).sum()), "groups": int(len(np.unique(g))),
                             "methods": {k: cell(y, g, v) for k, v in meth.items()}}
        pd.DataFrame(dict(record=df.record, y=y, p_v4=p, v3_red=meth["V3_frozen"])).to_csv("results/v4_predictions_%s.csv" % name, index=False)
        print("==", name, out["sets"][name]["n"], "true", out["sets"][name]["true"])
        for k, m in out["sets"][name]["methods"].items():
            f = lambda v: "n/a" if v is None else "%.3f" % v
            print("%-10s red: true %3d / false %3d | yellow: true %3d / false %3d | true share red %s [%s,%s] (n=%d) | yellow %s | true alarms in red %.2f [%.2f,%.2f]" % (
                k, m["true_red"], m["false_red"], m["true_yellow"], m["false_yellow"], f(m["true_share_red"]), f(m["true_share_red_ci"][0]), f(m["true_share_red_ci"][1]), m["red_n"], f(m["true_share_yellow"]),
                m["true_alarms_in_red_frac"], *m["true_alarms_in_red_ci"]))
    json.dump(out, open("results/v4_final.json", "w"), indent=1)
