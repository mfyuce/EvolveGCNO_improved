"""Exp A — node-state recurrent GNN (GConvGRU/TGCN) on ENGINEERED-10 features, vehicle-disjoint.

Question: the engineered physics-consistency features lift a STATIC GCN from 0.56 -> 0.77 MCC.
The node-state recurrent GConvGRU already reaches 0.749 on RAW-5 (recurrence substitutes for the
hand features). Does feeding it the engineered features push it ABOVE static-eng (0.768), or does
it under-exploit them the way it did on CPM (+4 vs RF's +25)? Self-contained (no graphs.* import).

Usage: python expA_gconvgru_eng.py <model:gconvgru|tgcn> <featset:eng10|eng8nopos|raw5> <seed> [epochs] [window]
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (matthews_corrcoef, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)
torch.set_num_threads(1)
from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "gconvgru"
FEATSET = sys.argv[2] if len(sys.argv) > 2 else "eng10"     # eng10 | eng8nopos | raw5
SEED    = int(sys.argv[3]) if len(sys.argv) > 3 else 3
EPOCHS  = int(sys.argv[4]) if len(sys.argv) > 4 else 40
WINDOW  = int(sys.argv[5]) if len(sys.argv) > 5 else 24
HIDDEN, LR = 32, 0.01
TAG = f"expA_{MODEL}_{FEATSET}_s{SEED}"


class RecGNN(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, kind="gconvgru", dropout=0.5):
        super().__init__()
        self.kind = kind
        self.rec = TGCN(in_f, hidden) if kind == "tgcn" else GConvGRU(in_f, hidden, 2)
        self.lin1 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)
        h = F.relu(self.lin1(H))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


def focal(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return (((1 - logpt.exp()) ** gamma) * (-logpt) * alpha[target]).mean()


def cw(y):
    w = torch.ones(2)
    for c in (0, 1):
        n = int((y == c).sum())
        if n > 0:
            w[c] = len(y) / (2.0 * n)
    return w


def best_metrics(probs, labels):
    roc = roc_auc_score(labels, probs)
    pr = average_precision_score(labels, probs)
    best = (-2.0, None, None)
    for t in np.arange(0.05, 1.0, 0.025):
        pred = (probs > t).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(labels, pred)
        if mcc > best[0]:
            best = (mcc, t, dict(
                P=precision_score(labels, pred, pos_label=1, zero_division=0),
                R=recall_score(labels, pred, pos_label=1, zero_division=0),
                f1mal=f1_score(labels, pred, pos_label=1, zero_division=0),
                macro=f1_score(labels, pred, average="macro", zero_division=0)))
    return dict(roc=roc, pr=pr, mcc=best[0], thr=best[1], **(best[2] or {}))


def main():
    log = open(f"{TAG}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== {TAG} node-state recurrent (featset={FEATSET}, focal, stateful TBPTT W={WINDOW}) ep={EPOCHS} ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = loader.get_dataset(lags=1)
    T = loader._dataset["time_periods"]; lags = loader.lags
    nstep = T - lags

    aug_raw = np.load("data/features_augmented.npy")           # (T, N, 10)
    N = aug_raw.shape[1]
    active = aug_raw[..., 0] != 0.0                            # raw x != 0 => vehicle present
    mean = aug_raw.reshape(-1, aug_raw.shape[-1]).mean(0, keepdims=True)
    std = aug_raw.reshape(-1, aug_raw.shape[-1]).std(0, keepdims=True)
    aug = ((aug_raw - mean) / (std + 1e-8)).astype(np.float32)
    if FEATSET == "eng8nopos":
        aug = aug[..., [2, 3, 4, 5, 6, 7, 8, 9]]              # drop absolute x,y
    elif FEATSET == "raw5":
        aug = aug[..., [0, 1, 2, 3, 4]]                        # x,y,heading,speed,accel
    in_f = aug.shape[-1]

    y_full = np.zeros((T, N), dtype=np.int64)
    for i in range(nstep):
        y_full[i] = loader.targets[i]
    node_label = (y_full.max(0) > 0).astype(int)

    rng = np.random.default_rng(SEED); trm = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    train_vehicle = torch.tensor(trm); test_vehicle = torch.tensor(~trm)
    out(f"vehicles: train={trm.sum()}(mal={node_label[trm].sum()}) "
        f"test={(~trm).sum()}(mal={node_label[~trm].sum()}) in_f={in_f}")

    ei = [torch.tensor(e, dtype=torch.long) for e in loader._edges]
    ew = [torch.tensor(w, dtype=torch.float) for w in loader._edge_weights]
    X = [torch.tensor(aug[i]) for i in range(nstep)]
    Y = [torch.tensor(loader.targets[i], dtype=torch.long) for i in range(nstep)]
    act = [torch.tensor(active[i]) for i in range(nstep)]
    n_train = int(0.7 * nstep)

    torch.manual_seed(SEED)
    model = RecGNN(in_f, kind=MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    def evaluate():
        model.eval(); P, L = [], []; H = None
        with torch.no_grad():
            for i in range(nstep):
                logits, H = model(X[i], ei[i], ew[i], H)
                m = act[i] & test_vehicle
                if m.sum() > 0:
                    P.append(torch.softmax(logits, 1)[m, 1].numpy())
                    L.append(Y[i][m].numpy())
        model.train()
        return best_metrics(np.concatenate(P), np.concatenate(L))

    for epoch in range(1, EPOCHS + 1):
        H = None
        for start in range(0, n_train, WINDOW):
            if H is not None:
                H = H.detach()
            opt.zero_grad(); wl = 0.0; cnt = 0
            for i in range(start, min(start + WINDOW, n_train)):
                logits, H = model(X[i], ei[i], ew[i], H)
                m = act[i] & train_vehicle
                if m.sum() == 0:
                    continue
                wl = wl + focal(logits[m], Y[i][m], cw(Y[i][m])); cnt += 1
            if cnt:
                (wl / cnt).backward(); opt.step()
        if epoch % 10 == 0 or epoch == EPOCHS:
            r = evaluate()
            out(f"[ep {epoch:3d}] ROC={r['roc']:.4f} bestMCC={r['mcc']:.3f}@{r['thr']:.2f} "
                f"P={r['P']:.3f} R={r['R']:.3f} macroF1={r['macro']:.3f}")

    r = evaluate()
    out(f"\nFINAL[{TAG}] ROC-AUC={r['roc']:.4f} PR-AUC={r['pr']:.4f} MCC={r['mcc']:.3f}@thr{r['thr']:.2f} "
        f"P={r['P']:.3f} R={r['R']:.3f} F1mal={r['f1mal']:.3f} macroF1={r['macro']:.3f} | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
