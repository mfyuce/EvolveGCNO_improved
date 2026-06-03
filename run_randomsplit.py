"""Competitors' OWN setup: random (node,timestep) 70/30 split (not vehicle-disjoint).

Classical MBD papers split tabular BSM rows randomly, so the same vehicle is in
train and test (different timesteps) — an EASY setup. This evaluates our models
the same way and reports BOTH weighted (their metric) and MACRO (+MCC), so we can
see whether we match the competitors' ~99% on equal footing, and what macro does.

Run:  python run_randomsplit.py <rf|gconvgru|static> [seed]
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef, roc_auc_score
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gconvgru"
SEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "random"   # random | temporal
EPOCHS, WINDOW, HIDDEN, LR = 40, 24, 32, 0.01


class GConvGRUNet(nn.Module):
    def __init__(self, in_f, h=HIDDEN, d=0.5):
        super().__init__(); self.rec = GConvGRU(in_f, h, 2)
        self.l1 = nn.Linear(h, h); self.cl = nn.Linear(h, 2); self.d = d
    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)
        return self.cl(F.dropout(F.relu(self.l1(H)), p=self.d, training=self.training)), H


class StaticGCN(nn.Module):
    def __init__(self, in_f, h=HIDDEN, d=0.5):
        super().__init__(); self.c1 = GCNConv(in_f, h); self.c2 = GCNConv(h, h)
        self.c3 = GCNConv(h, h); self.cl = nn.Linear(h, 2); self.d = d
    def forward(self, x, ei, ew):
        h = F.relu(self.c1(x, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.d, training=self.training)
        return self.cl(F.relu(self.c3(h, ei, ew)))


def focal(lg, t, a, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp) * a.to(lg.device)[t]).mean()


def report(name, P, L, w):
    roc = roc_auc_score(L, P)
    thr = max(((matthews_corrcoef(L, (P > t).astype(int)), t) for t in np.arange(0.05, 1.0, 0.025) if (P > t).any()), default=(0, .5))[1]
    pred = (P > thr).astype(int)
    w(f"\n[{name}] {SPLIT} split, thr={thr:.2f}")
    for avg in ("weighted", "macro", "micro"):
        p, r, f, _ = precision_recall_fscore_support(L, pred, average=avg, zero_division=0)
        w(f"  {avg:>9}: P={p*100:6.2f} R={r*100:6.2f} F1={f*100:6.2f}")
    w(f"  Accuracy={accuracy_score(L,pred)*100:.2f}  MCC={matthews_corrcoef(L,pred)*100:.2f}  ROC-AUC={roc*100:.2f}")


def main():
    out = open("randomsplit.log", "a")
    def w(m):
        print(m, flush=True); out.write(m + "\n"); out.flush()
    t0 = time.time()
    lb = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    ds = lb.get_dataset(lags=1); alls = list(ds)
    T, lags, N = lb._dataset["time_periods"], lb.lags, len(lb._dataset["node_labels"])
    aug_raw = np.load("data/features_augmented.npy")
    active = aug_raw[..., 0] != 0.0
    flat = aug_raw.reshape(-1, 10); augN = ((aug_raw - flat.mean(0)) / (flat.std(0) + 1e-8))
    Y = np.stack([lb.targets[i] for i in range(T - lags)])

    rng = np.random.default_rng(SEED)
    if SPLIT == "temporal":                                # first 70% timesteps -> train (paper protocol)
        n_train = int(0.7 * (T - lags))
        train_mask = np.zeros((T - lags, N), dtype=bool)
        train_mask[:n_train, :] = True
    else:                                                  # random per (node,timestep)
        train_mask = rng.random((T - lags, N)) < 0.7

    if MODEL == "rf":
        Xtr, ytr, Xte, yte = [], [], [], []
        for i in range(T - lags):
            a = active[i]; tr = a & train_mask[i]; te = a & (~train_mask[i])
            Xtr.append(augN[i][tr]); ytr.append(Y[i][tr]); Xte.append(augN[i][te]); yte.append(Y[i][te])
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr); Xte = np.concatenate(Xte); yte = np.concatenate(yte)
        w(f"RF random split seed={SEED}: train={len(ytr)}(mal {int(ytr.sum())}) test={len(yte)}(mal {int(yte.sum())})")
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=4, random_state=SEED)
        rf.fit(Xtr, ytr); proba = rf.predict_proba(Xte)[:, 1]
        report("RandomForest", proba, yte, w)
    else:
        in_f = lb.n_node_features
        X = [a.x for a in alls]; EI = [a.edge_index for a in alls]; EW = [a.edge_attr for a in alls]
        Yt = [torch.tensor(Y[i], dtype=torch.long) for i in range(T - lags)]
        trm = [torch.tensor(active[i] & train_mask[i]) for i in range(T - lags)]
        tem = [torch.tensor(active[i] & (~train_mask[i])) for i in range(T - lags)]
        is_rec = MODEL == "gconvgru"
        torch.manual_seed(SEED)
        model = GConvGRUNet(in_f) if is_rec else StaticGCN(in_f)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for _ in range(EPOCHS):
            H = None
            for st in range(0, T - lags, WINDOW):
                if is_rec and H is not None:
                    H = H.detach()
                opt.zero_grad(); wl = 0.0; c = 0
                for i in range(st, min(st + WINDOW, T - lags)):
                    if is_rec:
                        lg, H = model(X[i], EI[i], EW[i], H)
                    else:
                        lg = model(X[i], EI[i], EW[i])
                    m = trm[i]
                    if m.sum() == 0:
                        continue
                    wl = wl + focal(lg[m], Yt[i][m], base._snapshot_class_weights(Yt[i][m])); c += 1
                if c:
                    (wl / c).backward(); opt.step()
        model.eval(); P, L = [], []; H = None
        with torch.no_grad():
            for i in range(T - lags):
                if is_rec:
                    lg, H = model(X[i], EI[i], EW[i], H)
                else:
                    lg = model(X[i], EI[i], EW[i])
                m = tem[i]
                if m.sum() > 0:
                    P.append(torch.softmax(lg, 1)[m, 1].numpy()); L.append(Yt[i][m].numpy())
        report(MODEL, np.concatenate(P), np.concatenate(L), w)
    w(f"  ({(time.time()-t0)/60:.1f}min)")
    out.close()


if __name__ == "__main__":
    main()
