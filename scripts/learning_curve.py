"""Кривая обучения V5 для слайда. Обучение на 25/50/100% ГРУПП (запись CinC / пациент VTaC) пула CinC dev + VTaC train; оценка на VTaC validation;
красная группа одного размера (= размер при замороженном T у 100%-модели). Test не используется. 5 случайных подвыборок (seed 0-4) для 25% и 50%."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from scripts.v5 import data, metrics
from scripts.cv_gbm import models

tr, va, feats = data()
cfg = json.load(open("models/v5_config.json"))
y, g = va.y.values, va.group.values
groups = tr.group.unique()
rows = []
K = cfg["val_report"]["red_n"]  # размер красной группы замороженной модели V5 на val (85)
for frac in (0.25, 0.5, 1.0):
    for seed in ((0, 1, 2, 3, 4) if frac < 1 else (0,)):
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(groups, int(round(frac * len(groups))), replace=False)) if frac < 1 else set(groups)
        sub = tr[tr.group.isin(keep)]
        m = models()["lgbm"]().fit(sub[feats], sub.y)
        p = m.predict_proba(va[feats])[:, 1]
        red = np.zeros(len(y), bool); red[np.argsort(-p)[:K]] = True
        r = metrics(y, g, red, B=1000)
        rows.append(dict(frac=frac, seed=seed, n_train=len(sub), n_train_true=int(sub.y.sum()), **{k: r[k] for k in ("true_red", "red_n", "precision", "precision_ci", "recall")}))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
summ = [dict(frac=f, n_train_mean=float(d.n_train.mean()), red_n=K, precision_mean=float(d.precision.mean()), precision_min=float(d.precision.min()), precision_max=float(d.precision.max()),
             true_red_mean=float(d.true_red.mean())) for f, d in df.groupby("frac")]
json.dump(dict(validation="VTaC val 495 events (141 true), %d patients" % len(np.unique(g)), red_group_size=K, bootstrap="by patient, B=1000, 95%", draws=rows, summary=summ,
               note="25% и 50%: среднее по 5 случайным подвыборкам групп. Только для слайда; не оценка на тесте."), open("results/learning_curve.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
