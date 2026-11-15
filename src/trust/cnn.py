"""Маленькая 1D-CNN (ЭКГ 2 отведения + PPG, последние 20 с, 125 Гц) + one-hot типа тревоги. CPU, ~25k параметров."""
import numpy as np
import torch, torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_extra=5, c=(16, 32, 48)):
        super().__init__()
        L = []
        cin = 3
        for co in c:
            L += [nn.Conv1d(cin, co, 7, stride=2, padding=3), nn.BatchNorm1d(co), nn.ReLU(), nn.Conv1d(co, co, 5, padding=2), nn.BatchNorm1d(co), nn.ReLU()]
            cin = co
        self.f = nn.Sequential(*L)
        self.h = nn.Sequential(nn.Linear(2 * cin + n_extra, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x, e):
        z = self.f(x)
        z = torch.cat([z.mean(2), z.amax(2), e], 1)
        return self.h(z).squeeze(1)


def train(Xtr, Etr, ytr, epochs=14, seed=0, bs=64, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(Etr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(Xtr) + bs - 1) // bs))
    Xtr, Etr, ytr = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Etr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = Xtr[b]
            sh = np.random.randint(-125, 126)                     # сдвиг до ±1 с
            x = torch.roll(x, sh, 2) * (0.8 + 0.4 * torch.rand(len(b), 1, 1))
            opt.zero_grad(); loss = lossf(net(x, Etr[b]), ytr[b]); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


def predict(net, Xte, Ete):
    net.eval()
    with torch.no_grad():
        Xte, Ete = torch.tensor(Xte, dtype=torch.float32), torch.tensor(Ete, dtype=torch.float32)
        return torch.sigmoid(torch.cat([net(Xte[i:i + 256], Ete[i:i + 256]) for i in range(0, len(Xte), 256)])).numpy()


def fit_predict(Xtr, Etr, ytr, Xte, Ete, **kw):
    return predict(train(Xtr, Etr, ytr, **kw), Xte, Ete)
