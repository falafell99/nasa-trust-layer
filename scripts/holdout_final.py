"""ЕДИНСТВЕННЫЙ прогон hold-out (150 записей). Замороженные код/модели: коммит 1dad545. Ничего не подбирается.
Методы: keep-all, только тип тревоги (доля настоящих по train, порог 0.5), V0 ECG-only, V3 (red/yellow).
red = 'предсказано настоящее'. Результат: results/holdout_final.json"""
import json, sys, subprocess, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
from src.trust.cinc import load, read_alarms
from src.trust.features import extract
from src.trust.pipeline import verify, Q_GOOD
from src.trust import model as M

AL = read_alarms()
SP = json.load(open("data/split.json"))
P = M.load()


def one(rec):
    d = load(rec)
    typ = AL[rec][0]
    f = extract(d, typ)
    seg = {"ecg1": d["ecg"][0], "ecg2": d["ecg"][1] if len(d["ecg"]) > 1 else None, "ppg": None, "alarm_type": typ, "fs": 250}
    return dict(rec=rec, typ=typ, y=int(AL[rec][1]), q=float(f["q"]), p=M.predict(P, f), v0=int(verify(seg)))


def tbl(red, y):
    red, y = np.asarray(red, bool), np.asarray(y)
    a, b, c, d = int(((y == 1) & red).sum()), int(((y == 1) & ~red).sum()), int(((y == 0) & red).sum()), int(((y == 0) & ~red).sum())
    return a, b, c, d


def cinc(a, b, c, d):  # red=предсказано true: TP=a, FN=b, FP=c, TN=d
    return 100 * (a + d) / (a + d + c + 5 * b)


if __name__ == "__main__":
    ho = SP["holdout"]
    assert len(ho) == 150 and not set(ho) & set(SP["train"]) and not set(ho) & set(SP["validation"])
    with ProcessPoolExecutor() as ex:
        R = list(ex.map(one, ho, chunksize=8))
    y = np.array([r["y"] for r in R]); typ = [r["typ"] for r in R]
    rate = collections.defaultdict(lambda: [0, 0])
    for r in SP["train"]:
        t, l = AL[r]; rate[t][0] += int(l); rate[t][1] += 1
    methods = {
        "keep_all": np.ones(len(y), bool),
        "type_only": np.array([rate[t][0] / rate[t][1] >= 0.5 for t in typ]),
        "V0_ecg_only": np.array([r["v0"] == 1 for r in R]),
        "V3": np.array([M.decide_v3(r["q"], r["p"]) == "red" for r in R]),
    }
    rng = np.random.RandomState(0)
    B = 2000
    idx = [rng.randint(0, len(y), len(y)) for _ in range(B)]
    out = {"n": len(y), "true": int(y.sum()), "false": int((1 - y).sum()), "bootstrap": "by record, B=2000, seed 0, percentile 95%",
           "frozen_commit": "1dad545", "Q_GOOD": Q_GOOD, "T_RED": M.T_RED,
           "head": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(), "methods": {}}
    def ci(fn, red):
        v = []
        for i in idx:
            t = tbl(red[i], y[i]); v.append(fn(*t))
        v = np.array(v, float); v = v[~np.isnan(v)]
        return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [None, None]
    sh_r = lambda a, b, c, d: a / (a + c) if a + c else np.nan
    sh_y = lambda a, b, c, d: b / (b + d) if b + d else np.nan
    rg = lambda a, b, c, d: (a + c) / (a + b + c + d)
    for k, red in methods.items():
        a, b, c, d = tbl(red, y)
        out["methods"][k] = dict(true_red=a, true_yellow=b, false_red=c, false_yellow=d, red_n=a + c, yellow_n=b + d,
            true_share_red=(a / (a + c) if a + c else None), true_share_red_ci=ci(sh_r, red),
            true_share_yellow=(b / (b + d) if b + d else None), true_share_yellow_ci=ci(sh_y, red),
            red_group_share=(a + c) / len(y), red_group_share_ci=ci(rg, red),
            cinc_score_secondary=cinc(a, b, c, d), cinc_score_ci=ci(cinc, red),
            note="red = predicted true; for V3 yellow still keeps the alarm, so the official CinC score is only a secondary comparison")
    out["v3_extra"] = dict(q_below_good=int(sum(r["q"] < Q_GOOD for r in R)), p_ge_T_RED=int(sum(r["p"] >= M.T_RED for r in R)))
    out["per_record"] = [dict(rec=r["rec"], type=r["typ"], y=r["y"], q=round(r["q"], 4), p=round(r["p"], 4), v0=r["v0"]) for r in R]
    json.dump(out, open("results/holdout_final.json", "w"), indent=1)
    print("hold-out n=%d true=%d false=%d" % (out["n"], out["true"], out["false"]))
    for k, m in out["methods"].items():
        f = lambda v: "n/a" if v is None else "%.3f" % v
        print("%-12s red: true %2d / false %2d | yellow: true %2d / false %2d | true share red %s [%s,%s] (n=%d) | yellow %s [%s,%s] (n=%d) | red group %.3f | CinC %.1f [%.1f,%.1f]" % (
            k, m["true_red"], m["false_red"], m["true_yellow"], m["false_yellow"], f(m["true_share_red"]), f(m["true_share_red_ci"][0]), f(m["true_share_red_ci"][1]), m["red_n"],
            f(m["true_share_yellow"]), f(m["true_share_yellow_ci"][0]), f(m["true_share_yellow_ci"][1]), m["yellow_n"], m["red_group_share"], m["cinc_score_secondary"], *m["cinc_score_ci"]))
    print(out["v3_extra"])
