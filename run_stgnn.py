"""Spatio-temporal GNN training with in-order windowed truncated BPTT on raw kinematic features."""

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

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "evolve"
EPOCHS  = int(sys.argv[2]) if len(sys.argv) > 2 else 60
WINDOW  = int(sys.argv[3]) if len(sys.argv) > 3 else 16
HIDDEN  = int(sys.argv[4]) if len(sys.argv) > 4 else 32
LR = 0.01
TAG = f"stgnn_{VARIANT}"


class STGNN(nn.Module):
    """EvolveGCN-H recurrence (proper BPTT) OR static, + de-saturated head."""
    def __init__(self, node_count, in_f=5, hidden=HIDDEN, recurrent=True, dropout=0.5):
        super().__init__()
        self.recurrent_on = recurrent
        self.dropout = dropout
        if recurrent:
            self.recurrent = EvolveGCNHImproved(node_count, in_f)
        self.c1 = GCNConv(in_f, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)

    def reset_hidden_state(self):
        if self.recurrent_on:
            object.__setattr__(self.recurrent, "weight", None)

    def _evolve(self, x, ei, ew):
        # local re-implementation that persists the GRU-updated W WITH gradient
        # (vendored cell freezes it). reset_hidden_state() controls window start.
        X_tilde = self.recurrent.pooling_layer(x, ei)
        X_tilde = X_tilde[0][None, :, :]
        if self.recurrent.weight is None:
            object.__setattr__(self.recurrent, "weight", self.recurrent.initial_weight)
        W_in = self.recurrent.weight[None, :, :]
        _, W_out = self.recurrent.recurrent_layer(X_tilde, W_in)
        W_new = W_out.squeeze(dim=0)
        object.__setattr__(self.recurrent, "weight", W_new)   # keep attached -> BPTT across window
        return self.recurrent.conv_layer(W_new, x, ei, ew)

    def forward(self, x, ei, ew):
        h = self._evolve(x, ei, ew) if self.recurrent_on else x
        h = F.relu(self.c1(h, ei, ew))
        h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h)


def focal(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = ((1 - logpt.exp()) ** gamma) * (-logpt)
    return (loss * alpha.to(logits.device)[target]).mean()


def evaluate(model, snaps):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        model.reset_hidden_state()                  # warm from learned seed, evolve in order
        for s in snaps:
            lg = model(s.x, s.edge_index, s.edge_attr)
            probs.append(torch.softmax(lg, 1)[:, 1].numpy())
            labels.append(s.y.numpy().astype(int))
    model.train()
    probs = np.concatenate(probs); labels = np.concatenate(labels)
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
                f1mal=f1_score(labels, pred, pos_label=1, zero_division=0),
                macro=f1_score(labels, pred, average="macro", zero_division=0)))
    return dict(roc=roc, pr=pr, mcc=best[0], thr=best[1], **(best[2] or {}))


def main():
    log = open(f"{TAG}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== ST-GNN {TAG} (RAW 5 feat, in-order BPTT W={WINDOW}) epochs={EPOCHS} hidden={HIDDEN} ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    dataset = loader.get_dataset(lags=1)          # loader.features = standardized RAW 5 feats
    train_ds, test_ds = temporal_signal_split(dataset, train_ratio=0.7)
    train_snaps = list(train_ds); test_snaps = list(test_ds)   # KEEP ORDER (no shuffle)
    n_nodes = len(loader._dataset["node_labels"])
    out(f"n_nodes={n_nodes} in_features={loader.n_node_features} "
        f"train_snaps={len(train_snaps)} test_snaps={len(test_snaps)}")

    torch.manual_seed(0)
    model = STGNN(n_nodes, in_f=loader.n_node_features, recurrent=(VARIANT == "evolve"))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(1, EPOCHS + 1):
        tot, nb = 0.0, 0
        for start in range(0, len(train_snaps), WINDOW):
            window = train_snaps[start:start + WINDOW]
            model.reset_hidden_state()              # truncated BPTT: independent windows
            opt.zero_grad()
            wloss = 0.0
            for s in window:
                lg = model(s.x, s.edge_index, s.edge_attr)
                wloss = wloss + focal(lg, s.y.long(), base._snapshot_class_weights(s.y))
            (wloss / len(window)).backward()
            opt.step()
            tot += float(wloss) / len(window); nb += 1
        if epoch % 10 == 0 or epoch == EPOCHS:
            m = evaluate(model, test_snaps)
            out(f"[ep {epoch:3d}] loss={tot/nb:.4f} | ROC={m['roc']:.4f} PR={m['pr']:.4f} "
                f"| bestMCC={m['mcc']:.3f}@{m['thr']:.2f} P={m['P']:.3f} R={m['R']:.3f} "
                f"F1mal={m['f1mal']:.3f} macroF1={m['macro']:.3f}")

    m = evaluate(model, test_snaps)
    out(f"\nFINAL[{TAG}] ROC-AUC={m['roc']:.4f} PR-AUC={m['pr']:.4f} bestMCC={m['mcc']:.3f}@thr{m['thr']:.2f} "
        f"P={m['P']:.3f} R={m['R']:.3f} F1mal={m['f1mal']:.3f} macroF1={m['macro']:.3f} | {(time.time()-t0)/60:.1f}min")
    torch.save(model, f"model/{TAG}.pt")
    log.close()


if __name__ == "__main__":
    main()
