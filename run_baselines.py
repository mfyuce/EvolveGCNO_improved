"""Honest baselines on the SAME data/split/features/head as desat EvolveGCN.

Answers the paper's central claim — does the graph / temporal structure help?

  mlp : per-node MLP on the 5 kinematic features, NO edges at all.
  gcn : static GCN (edge_index + edge_weight), NO temporal recurrence.

Compared against desat EvolveGCN (ROC-AUC 0.98, MCC 0.44) from run_experiments.
All share: 5 standardized features, full-batch class-weighted CE, Adam lr=0.01,
150 epochs, de-saturated ReLU head (hidden=16), identical eval (ROC/PR + best-MCC
threshold sweep).

Run:  python run_baselines.py <mlp|gcn> [epochs]
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
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "mlp"
EPOCHS  = int(sys.argv[2]) if len(sys.argv) > 2 else 150
LR = 0.01
HIDDEN = 16


class MLP(nn.Module):
    """Per-node MLP — ignores graph structure entirely."""
    def __init__(self, in_f=5, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.l1 = nn.Linear(in_f, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 2)
        self.dropout = dropout
    def forward(self, x, ei, ew):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.l3(h)


class StaticGCN(nn.Module):
    """Static GCN over the proximity graph — uses edges but no temporal state."""
    def __init__(self, in_f=5, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_f, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout
    def forward(self, x, ei, ew):
        h = F.relu(self.c1(x, ei, ew))
        h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h)


def evaluate(model, dataset):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for snap in dataset:
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            labels.append(snap.y.cpu().numpy().astype(int))
    model.train()
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    roc = roc_auc_score(labels, probs)
    pr  = average_precision_score(labels, probs)
    best = (-2, None, None)
    for thr in np.arange(0.05, 1.0, 0.025):
        pred = (probs > thr).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(labels, pred)
        if mcc > best[0]:
            best = (mcc, thr, dict(
                P=precision_score(labels, pred, pos_label=1, zero_division=0),
                R=recall_score(labels, pred, pos_label=1, zero_division=0),
                f1mal=f1_score(labels, pred, pos_label=1, zero_division=0),
                macro=f1_score(labels, pred, average="macro", zero_division=0)))
    return dict(roc=roc, pr=pr, mcc=best[0], thr=best[1], **(best[2] or {}))


def main():
    log = open(f"baseline_{VARIANT}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    out(f"=== BASELINE '{VARIANT}' epochs={EPOCHS} hidden={HIDDEN} ===")
    t0 = time.time()
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    dataset = loader.get_dataset(lags=1)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)

    torch.manual_seed(0)
    model = MLP() if VARIANT == "mlp" else StaticGCN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        cost = 0.0; n = 0
        for snap in train_dataset:
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            cw = base._snapshot_class_weights(snap.y)
            cost = cost + F.cross_entropy(logits, snap.y.long(), weight=cw)
            n += 1
        (cost / n).backward()
        opt.step()
        if epoch % 30 == 0 or epoch == EPOCHS:
            m = evaluate(model, test_dataset)
            out(f"[ep {epoch:3d}] CE={float(cost)/n:.4f} | ROC={m['roc']:.4f} "
                f"PR={m['pr']:.4f} | bestMCC={m['mcc']:.3f}@{m['thr']:.2f} "
                f"P={m['P']:.3f} R={m['R']:.3f} F1mal={m['f1mal']:.3f} macroF1={m['macro']:.3f}")

    m = evaluate(model, test_dataset)
    out(f"\nFINAL[{VARIANT}] ROC-AUC={m['roc']:.4f} PR-AUC={m['pr']:.4f} "
        f"bestMCC={m['mcc']:.3f}@thr{m['thr']:.2f} "
        f"P={m['P']:.3f} R={m['R']:.3f} F1mal={m['f1mal']:.3f} macroF1={m['macro']:.3f} "
        f"| {(time.time()-t0)/60:.1f}min")
    torch.save(model, f"model/baseline_{VARIANT}.pt")
    log.close()


if __name__ == "__main__":
    main()
