"""5-fold CV маленькой CNN (те же фолды, что в cv_gbm.py). Слот: 45 минут CPU. Сохраняет data/oof_cnn.npy."""
import sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from src.trust.cnn import fit_predict
from src.trust.feat2 import TYPES
import torch
torch.set_num_threads(16)

t0 = time.time()
dfs = [pd.read_csv("data/feat2_cinc_train-validation.csv").assign(ds="cinc")]
X = [np.load("data/cnn_cinc_train-validation.npy")]
if len(sys.argv) > 1 and sys.argv[1] == "vtac":
    dfs.append(pd.read_csv("data/feat2_vtac_train-val.csv").assign(ds="vtac")); X.append(np.load("data/cnn_vtac_train-val.npy"))
df = pd.concat(dfs, ignore_index=True); X = np.concatenate(X).astype(np.float32)
E = df[["type_" + t for t in TYPES]].values.astype(np.float32)
y = df.y.values
p = np.zeros(len(df))
for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, df.group.values)):
    p[te] = fit_predict(X[tr], E[tr], y[tr], X[te], E[te])
    print("fold", k, "AUROC %.3f" % roc_auc_score(y[te], p[te]), "%.0fs" % (time.time() - t0), flush=True)
np.save("data/oof_cnn.npy", p)
for ds in df.ds.unique():
    m = (df.ds == ds).values; print(ds, "CNN OOF AUROC %.3f" % roc_auc_score(y[m], p[m]))
print("elapsed %.0fs" % (time.time() - t0))
