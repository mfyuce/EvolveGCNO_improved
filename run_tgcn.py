"""Node-state recurrent GNN (TGCN / GConvGRU) — the architecturally-RIGHT
temporal model, given a fair chance. Unlike EvolveGCN-H (evolves a shared
weight matrix + TopK->5 bottleneck), these carry a PER-NODE hidden state h_i(t)
that evolves over time = exactly per-vehicle trajectory dynamics.

Key difference from the earlier (failed) EvolveGCN tests:
  - STATEFUL sequential processing: hidden state H carried across the whole
    in-order sequence (detached per window for truncated BPTT, NOT zero-reset),
    so long-memory temporal patterns can form.
  - Per-node recurrence (no pooling bottleneck), raw 5 features, focal loss.

Compare vs static GCN (raw5): temporal 0.878 / vehicle-disjoint ~0.63.
If TGCN/GConvGRU beats static, temporal finally earns its place.

Run:  python run_tgcn.py <tgcn|gconvgru> <temporal|disjoint> [epochs] [window] [seed]
"""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import TGCN, GConvGRU
import graphs.recurrent.graphs_base as base

MODEL  = sys.argv[1] if len(sys.argv) > 1 else "tgcn"
SPLIT  = sys.argv[2] if len(sys.argv) > 2 else "temporal"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
WINDOW = int(sys.argv[4]) if len(sys.argv) > 4 else 24
SEED   = int(sys.argv[5]) if len(sys.argv) > 5 else 3
HIDDEN, LR = 32, 0.01
TAG = f"{MODEL}_{SPLIT}_s{SEED}"


class RecGNN(nn.Module):
    def __init__(self, in_f=5, hidden=HIDDEN, kind="tgcn", dropout=0.5):
        super().__init__()
        self.kind = kind
        if kind == "tgcn":
            self.rec = TGCN(in_f, hidden)
        else:
            self.rec = GConvGRU(in_f, hidden, 2)
        self.lin1 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)          # (N, hidden) node-state recurrence
        h = F.relu(self.lin1(H))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


def focal(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return (((1 - logpt.exp()) ** gamma) * (-logpt) * alpha.to(logits.device)[target]).mean()


def best_metrics(probs, labels):
    roc = roc_auc_score(labels, probs); pr = average_precision_score(labels, probs)
    best = (-2, None, None)
    for t in np.arange(0.05, 1.0, 0.025):
        pred = (probs > t).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(labels, pred)
        if mcc > best[0]:
            best = (mcc, t, dict(
                P=precision_score(labels, pred, pos_label=1, zero_division=0),
                R=recall_score(labels, pred, pos_label=1, zero_division=0),
                macro=f1_score(labels, pred, average="macro", zero_division=0)))
    return dict(roc=roc, pr=pr, mcc=best[0], thr=best[1], **(best[2] or {}))


def main():
    log = open(f"{TAG}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== {TAG} node-state recurrent (raw5, focal, stateful TBPTT W={WINDOW}) epochs={EPOCHS} ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    ds = loader.get_dataset(lags=1)
    tr_ds, te_ds = temporal_signal_split(ds, train_ratio=0.7)
    alls = list(ds); n_nodes = len(loader._dataset["node_labels"])
    n_train = len(list(tr_ds))

    if SPLIT == "disjoint":
        T, lags = loader._dataset["time_periods"], loader.lags
        yf = np.zeros((T - lags, n_nodes), dtype=int)
        for i in range(T - lags):
            yf[i] = loader.targets[i]
        node_label = (yf.max(0) > 0).astype(int)
        rng = np.random.default_rng(SEED); trm = np.zeros(n_nodes, dtype=bool)
        for c in (0, 1):
            idx = np.where(node_label == c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]] = True
        train_vehicle = torch.tensor(trm); test_vehicle = torch.tensor(~trm)
        aug_raw = np.load("data/features_augmented.npy")
        active = [torch.tensor(aug_raw[i, :, 0] != 0.0) for i in range(len(alls))]
        out(f"disjoint: train_veh={trm.sum()}(mal={node_label[trm].sum()}) test_veh={(~trm).sum()}(mal={node_label[~trm].sum()})")

    torch.manual_seed(SEED)
    model = RecGNN(in_f=loader.n_node_features, kind=MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    def evaluate():
        model.eval(); P, L = [], []; H = None
        with torch.no_grad():
            for i, s in enumerate(alls):
                logits, H = model(s.x, s.edge_index, s.edge_attr, H)
                if SPLIT == "temporal":
                    if i >= n_train:               # test = last 30% timesteps
                        P.append(torch.softmax(logits, 1)[:, 1].numpy()); L.append(s.y.numpy().astype(int))
                else:
                    m = active[i] & test_vehicle
                    if m.sum() > 0:
                        P.append(torch.softmax(logits, 1)[m, 1].numpy()); L.append(s.y[m].numpy().astype(int))
        model.train()
        return best_metrics(np.concatenate(P), np.concatenate(L))

    train_idx = list(range(n_train))               # in-order first-70% snapshots
    for epoch in range(1, EPOCHS + 1):
        H = None
        for start in range(0, len(train_idx), WINDOW):
            if H is not None:
                H = H.detach()                     # truncated BPTT: carry VALUE, cut gradient
            opt.zero_grad(); wl = 0.0; cnt = 0
            for i in train_idx[start:start + WINDOW]:
                s = alls[i]
                logits, H = model(s.x, s.edge_index, s.edge_attr, H)
                if SPLIT == "disjoint":
                    m = active[i] & train_vehicle
                    if m.sum() == 0:
                        continue
                    wl = wl + focal(logits[m], s.y[m].long(), base._snapshot_class_weights(s.y[m])); cnt += 1
                else:
                    wl = wl + focal(logits, s.y.long(), base._snapshot_class_weights(s.y)); cnt += 1
            if cnt == 0:
                continue
            (wl / cnt).backward(); opt.step()
        if epoch % 10 == 0 or epoch == EPOCHS:
            m = evaluate()
            out(f"[ep {epoch:3d}] ROC={m['roc']:.4f} PR={m['pr']:.4f} bestMCC={m['mcc']:.3f}@{m['thr']:.2f} "
                f"P={m['P']:.3f} R={m['R']:.3f} macroF1={m['macro']:.3f}")

    m = evaluate()
    out(f"\nFINAL[{TAG}] ROC-AUC={m['roc']:.4f} PR-AUC={m['pr']:.4f} bestMCC={m['mcc']:.3f}@thr{m['thr']:.2f} "
        f"P={m['P']:.3f} R={m['R']:.3f} macroF1={m['macro']:.3f} | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
