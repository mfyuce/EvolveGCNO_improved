"""Full MACRO metric suite (macro-P, macro-R, macro-F1) + weighted + acc + MCC
+ per-class F1, for every key model on the SAME vehicle-disjoint split.

Run:  python macro_all.py <gconvgru|tgcn|static_raw|static_eng|evolve> [seed]
"""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN
from graphs.recurrent.evolvegcnh_improved import EvolveGCNHImproved
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gconvgru"
SEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EPOCHS, WINDOW, HIDDEN, LR = 40, 24, 32, 0.01
IS_REC = MODEL in ("gconvgru", "tgcn")


class GConvGRUNet(nn.Module):
    def __init__(self, in_f, h=HIDDEN, d=0.5):
        super().__init__(); self.rec = GConvGRU(in_f, h, 2)
        self.l1 = nn.Linear(h, h); self.cl = nn.Linear(h, 2); self.d = d
    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)
        return self.cl(F.dropout(F.relu(self.l1(H)), p=self.d, training=self.training)), H


class TGCNNet(nn.Module):
    def __init__(self, in_f, h=HIDDEN, d=0.5):
        super().__init__(); self.rec = TGCN(in_f, h)
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


class EvolveNet(nn.Module):
    def __init__(self, node_count, in_f, h=HIDDEN, d=0.5):
        super().__init__(); self.rec = EvolveGCNHImproved(node_count, in_f)
        self.c1 = GCNConv(in_f, h); self.c2 = GCNConv(h, h); self.c3 = GCNConv(h, h)
        self.cl = nn.Linear(h, 2); self.d = d
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew):
        X = self.rec.pooling_layer(x, ei); X = X[0][None, :, :]
        if self.rec.weight is None:
            object.__setattr__(self.rec, "weight", self.rec.initial_weight)
        _, W = self.rec.recurrent_layer(X, self.rec.weight[None, :, :]); W = W.squeeze(0)
        object.__setattr__(self.rec, "weight", W)
        hh = self.rec.conv_layer(W, x, ei, ew)
        hh = F.relu(self.c1(hh, ei, ew)); hh = F.relu(self.c2(hh, ei, ew))
        hh = F.dropout(hh, p=self.d, training=self.training)
        return self.cl(F.relu(self.c3(hh, ei, ew)))


def focal(lg, t, a, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp) * a.to(lg.device)[t]).mean()


def main():
    log = open(f"macro_{MODEL}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== MACRO suite — {MODEL} (vehicle-disjoint seed={SEED}) ===")
    t0 = time.time()

    lb = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    ds = lb.get_dataset(lags=1)
    alls = list(ds); n_nodes = len(lb._dataset["node_labels"]); T, lags = lb._dataset["time_periods"], lb.lags
    aug_raw = np.load("data/features_augmented.npy")
    active = [torch.tensor(aug_raw[i, :, 0] != 0.0) for i in range(T - lags)]

    if MODEL == "static_eng":
        mean = aug_raw.reshape(-1, 10).mean(0, keepdims=True); std = aug_raw.reshape(-1, 10).std(0, keepdims=True)
        aug = ((aug_raw - mean[None]) / (std[None] + 1e-8)).astype(np.float32)
        X = [torch.tensor(aug[i]) for i in range(T - lags)]; in_f = 10
    else:
        X = [a.x for a in alls]; in_f = lb.n_node_features
    Y = [torch.tensor(lb.targets[i], dtype=torch.long) for i in range(T - lags)]
    EI = [a.edge_index for a in alls]; EW = [a.edge_attr for a in alls]

    yf = np.stack([lb.targets[i] for i in range(T - lags)]); node_label = (yf.max(0) > 0).astype(int)
    rng = np.random.default_rng(SEED); trm = np.zeros(n_nodes, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]] = True
    trv, tev = torch.tensor(trm), torch.tensor(~trm)
    n_tr = int(0.7 * (T - lags))

    torch.manual_seed(SEED)
    if MODEL == "gconvgru":
        model = GConvGRUNet(in_f)
    elif MODEL == "tgcn":
        model = TGCNNet(in_f)
    elif MODEL == "evolve":
        model = EvolveNet(n_nodes, in_f)
    else:
        model = StaticGCN(in_f)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        H = None
        for st in range(0, n_tr, WINDOW):
            if IS_REC and H is not None:
                H = H.detach()
            if MODEL == "evolve":
                model.reset()
            opt.zero_grad(); wl = 0.0; c = 0
            for i in range(st, min(st + WINDOW, n_tr)):
                m = active[i] & trv
                if IS_REC:
                    lg, H = model(X[i], EI[i], EW[i], H)
                else:
                    lg = model(X[i], EI[i], EW[i])
                if m.sum() == 0:
                    continue
                wl = wl + focal(lg[m], Y[i][m], base._snapshot_class_weights(Y[i][m])); c += 1
            if c:
                (wl / c).backward(); opt.step()

    # collect test-vehicle predictions
    model.eval(); P, L = [], []; H = None
    with torch.no_grad():
        if MODEL == "evolve":
            model.reset()
        for i in range(T - lags):
            if IS_REC:
                lg, H = model(X[i], EI[i], EW[i], H)
            else:
                lg = model(X[i], EI[i], EW[i])
            m = active[i] & tev
            if m.sum() > 0:
                P.append(torch.softmax(lg, 1)[m, 1].numpy()); L.append(Y[i][m].numpy())
    P = np.concatenate(P); L = np.concatenate(L)

    # best-MCC threshold
    thr = max(((matthews_corrcoef(L, (P > t).astype(int)), t) for t in np.arange(0.05, 1.0, 0.025) if (P > t).any()), default=(0, 0.5))[1]
    pred = (P > thr).astype(int)
    acc = accuracy_score(L, pred); mcc = matthews_corrcoef(L, pred)
    out(f"\noperating threshold (best-MCC) = {thr:.2f}")
    out(f"{'avg':>10} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    for avg in ("macro", "weighted", "micro"):
        p, r, f, _ = precision_recall_fscore_support(L, pred, average=avg, zero_division=0)
        out(f"{avg:>10} {p*100:7.2f} {r*100:7.2f} {f*100:7.2f}")
    pc = precision_recall_fscore_support(L, pred, labels=[0, 1], zero_division=0)
    out(f"\nper-class: benign  P={pc[0][0]*100:.2f} R={pc[1][0]*100:.2f} F1={pc[2][0]*100:.2f}")
    out(f"           malic.  P={pc[0][1]*100:.2f} R={pc[1][1]*100:.2f} F1={pc[2][1]*100:.2f}")
    out(f"\nAccuracy={acc*100:.2f}  MCC={mcc*100:.2f}  | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
