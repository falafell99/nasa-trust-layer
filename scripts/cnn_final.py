"""CNN финал (только torch). train: обучает на пуле CinC train+val + VTaC train+val -> models/v4_cnn.pt. predict <имя набора>: results/pcnn_<имя>.npy."""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from src.trust import cnn
TYPES = ["Asystole", "Bradycardia", "Tachycardia", "Ventricular_Tachycardia", "Ventricular_Flutter_Fib"]
torch.set_num_threads(16)
E = lambda df: df[["type_" + t for t in TYPES]].values.astype(np.float32)
if sys.argv[1] == "train":
    d = pd.concat([pd.read_csv("data/feat2_cinc_train-validation.csv"), pd.read_csv("data/feat2_vtac_train-val.csv")], ignore_index=True)
    X = np.concatenate([np.load("data/cnn_cinc_train-validation.npy"), np.load("data/cnn_vtac_train-val.npy")]).astype(np.float32)
    net = cnn.train(X, E(d), d.y.values)
    torch.save(net.state_dict(), "models/v4_cnn.pt"); print("saved", len(d))
else:
    name = sys.argv[2]
    path = {"vtac_test": "vtac_test", "cinc_holdout_second_look": "cinc_holdout"}[name]
    d = pd.read_csv("data/feat2_%s.csv" % path); X = np.load("data/cnn_%s.npy" % path).astype(np.float32)
    net = cnn.Net(5); net.load_state_dict(torch.load("models/v4_cnn.pt")); np.save("results/pcnn_%s.npy" % name, cnn.predict(net, X, E(d))); print("predicted", name, len(d))
