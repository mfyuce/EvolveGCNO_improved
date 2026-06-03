"""Train the detector with augmented consistency features using per-snapshot focal-loss training."""

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
from torch_geometric_temporal.signal import temporal_signal_split, DynamicGraphTemporalSignal
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

MODEL  = sys.argv[1] if len(sys.argv) > 1 else "gcn"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
HIDDEN = int(sys.argv[3]) if len(sys.argv) > 3 else 32
LR = 0.01
TAG = f"aug_{MODEL}"


class MLP(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.l1 = nn.Linear(in_f, hidden); self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 2); self.dropout = dropout
    def forward(self, x, ei, ew):
        h = F.relu(self.l1(x)); h = F.relu(self.l2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.l3(h)


class StaticGCN(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_f, hidden); self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout
    def forward(self, x, ei, ew):
        h = F.relu(self.c1(x, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h)


def focal_loss(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    loss = ((1 - pt) ** gamma) * (-logpt)
    if alpha is not None:
        loss = loss * alpha.to(logits.device)[target]
    return loss.mean()


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
    roc = roc_auc_score(labels, probs); pr = average_precision_score(labels, probs)
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
    log = open(f"{TAG}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    out(f"=== AUGMENTED {TAG} epochs={EPOCHS} hidden={HIDDEN} (persnapshot+focal) ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = loader.get_dataset(lags=1)   # populates _edges, _edge_weights, targets
    lags = loader.lags
    T = loader._dataset["time_periods"]

    aug = np.load("data/features_augmented.npy")          # (T, N, 10)
    mean = aug.reshape(-1, aug.shape[-1]).mean(0, keepdims=True)
    std  = aug.reshape(-1, aug.shape[-1]).std(0, keepdims=True)
    aug_std = (aug - mean[None]) / (std[None] + 1e-8)
    in_f = aug.shape[-1]
    feats = [aug_std[i].astype(np.float32) for i in range(T - lags)]

    signal = DynamicGraphTemporalSignal(
        loader._edges, loader._edge_weights, feats, loader.targets
    )
    train_dataset, test_dataset = temporal_signal_split(signal, train_ratio=0.7)
    train_snaps = list(train_dataset)
    out(f"in_features={in_f}  train_snaps={len(train_snaps)}")

    torch.manual_seed(0)
    model = MLP(in_f) if MODEL == "mlp" else StaticGCN(in_f)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(0)

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_snaps))
        tot = 0.0
        for idx in order:
            snap = train_snaps[idx]
            opt.zero_grad()
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            cw = base._snapshot_class_weights(snap.y)
            loss = focal_loss(logits, snap.y.long(), cw, gamma=2.0)
            loss.backward(); opt.step()
            tot += float(loss)
        if epoch % 10 == 0 or epoch == EPOCHS:
            m = evaluate(model, test_dataset)
            out(f"[ep {epoch:3d}] loss={tot/len(train_snaps):.4f} | ROC={m['roc']:.4f} "
                f"PR={m['pr']:.4f} | bestMCC={m['mcc']:.3f}@{m['thr']:.2f} "
                f"P={m['P']:.3f} R={m['R']:.3f} F1mal={m['f1mal']:.3f} macroF1={m['macro']:.3f}")

    m = evaluate(model, test_dataset)
    out(f"\nFINAL[{TAG}] ROC-AUC={m['roc']:.4f} PR-AUC={m['pr']:.4f} "
        f"bestMCC={m['mcc']:.3f}@thr{m['thr']:.2f} P={m['P']:.3f} R={m['R']:.3f} "
        f"F1mal={m['f1mal']:.3f} macroF1={m['macro']:.3f} | {(time.time()-t0)/60:.1f}min")
    torch.save(model, f"model/{TAG}.pt")
    log.close()


if __name__ == "__main__":
    main()
