"""Honest-model improvement experiments (ablation of two fixable issues).

The ceiling model has ROC-AUC=0.78 (real signal) but MCC=0.19 because its
outputs saturate (triple-tanh + 2-dim bottleneck => bimodal probabilities,
threshold tuning can't help).  Two candidate levers, ablated independently:

  desaturate : ReLU + wider channels + NO final tanh + no 2-dim bottleneck
               -> calibrated logits so the 0.78 separability reaches the
                  decision boundary.
  temporal   : persist the GRU hidden weight W across snapshots and let
               initial_weight receive gradient (the vendored cell freezes it
               via `self.weight = initial_weight.data`, so EvolveGCN-H never
               actually evolves).

Variants (select via argv[1]): base | desat | temporal | both
Reports ROC-AUC / PR-AUC (threshold-free) + best-MCC over a threshold sweep.

Run:  python run_experiments.py <variant> [epochs]
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
from graphs.recurrent.evolvegcnh_improved import EvolveGCNHImproved
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "both"
EPOCHS  = int(sys.argv[2]) if len(sys.argv) > 2 else 150
LR = 0.01
DESAT    = VARIANT in ("desat", "both")
TEMPORAL = VARIANT in ("temporal", "both")


class ExpGCN(nn.Module):
    def __init__(self, node_count, node_features=5, hidden=16,
                 desaturate=True, temporal_fix=True, dropout=0.5):
        super().__init__()
        self.desaturate = desaturate
        self.temporal_fix = temporal_fix
        self.dropout = dropout
        self.recurrent = EvolveGCNHImproved(node_count, node_features)
        if desaturate:
            self.conv1 = GCNConv(node_features, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.conv3 = GCNConv(hidden, hidden)
            self.classifier = nn.Linear(hidden, 2)
        else:
            # paper-faithful narrow tanh stack
            self.conv1 = GCNConv(node_features, 4)
            self.conv2 = GCNConv(4, 4)
            self.conv3 = GCNConv(4, 2)
            self.classifier = nn.Linear(2, 2)

    def reset_hidden_state(self):
        object.__setattr__(self.recurrent, "weight", None)

    def _recurrent(self, x, ei, ew):
        if not self.temporal_fix:
            return self.recurrent(x, ei, ew)  # original (dead) behaviour
        # temporal-fixed: use initial_weight (grad!) on first call, persist W after
        X_tilde = self.recurrent.pooling_layer(x, ei)
        X_tilde = X_tilde[0][None, :, :]
        if self.recurrent.weight is None:
            object.__setattr__(self.recurrent, "weight", self.recurrent.initial_weight)
        W_in = self.recurrent.weight[None, :, :]
        _, W_out = self.recurrent.recurrent_layer(X_tilde, W_in)
        W_new = W_out.squeeze(dim=0)
        object.__setattr__(self.recurrent, "weight", W_new.detach())  # evolve value, no BPTT
        return self.recurrent.conv_layer(W_new, x, ei, ew)

    def forward(self, x, ei, ew):
        h = self._recurrent(x, ei, ew)
        if self.desaturate:
            h = F.relu(self.conv1(h, ei, ew))
            h = F.relu(self.conv2(h, ei, ew))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(self.conv3(h, ei, ew))
            return self.classifier(h)
        else:
            h = self.conv1(h, ei, ew).tanh()
            h = self.conv2(h, ei, ew).tanh()
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv3(h, ei, ew).tanh()
            return self.classifier(h)


def evaluate(model, dataset):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        if hasattr(model, "reset_hidden_state"):
            model.reset_hidden_state()
        for snap in dataset:
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            labels.append(snap.y.cpu().numpy().astype(int))
    model.train()
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    roc = roc_auc_score(labels, probs)
    pr  = average_precision_score(labels, probs)
    best_mcc, best_thr, best_stats = -2, None, None
    for thr in np.arange(0.05, 1.0, 0.025):
        pred = (probs > thr).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(labels, pred)
        if mcc > best_mcc:
            best_mcc = mcc; best_thr = thr
            best_stats = dict(
                P=precision_score(labels, pred, pos_label=1, zero_division=0),
                R=recall_score(labels, pred, pos_label=1, zero_division=0),
                f1mal=f1_score(labels, pred, pos_label=1, zero_division=0),
                macro=f1_score(labels, pred, average="macro", zero_division=0),
            )
    return dict(roc=roc, pr=pr, mcc=best_mcc, thr=best_thr, **(best_stats or {}))


def main():
    log = open(f"exp_{VARIANT}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    out(f"=== EXPERIMENT '{VARIANT}' desat={DESAT} temporal={TEMPORAL} "
        f"epochs={EPOCHS} ===")
    t0 = time.time()
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    dataset = loader.get_dataset(lags=1)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
    n_nodes = len(loader._dataset["node_labels"])

    torch.manual_seed(0)
    model = ExpGCN(n_nodes, node_features=loader.n_node_features,
                   desaturate=DESAT, temporal_fix=TEMPORAL)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_grad_capable = n_params
    out(f"params={n_params}")

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        if hasattr(model, "reset_hidden_state"):
            model.reset_hidden_state()
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
    torch.save(model, f"model/exp_{VARIANT}.pt")
    log.close()


if __name__ == "__main__":
    main()
