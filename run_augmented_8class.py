"""8-class with the WINNING recipe: augmented features + focal + per-snapshot.

Does the multiclass task (paper's real 0..7 label space) also recover, the way
binary did (MCC 0.18 -> 0.96)?  Earlier 8-class (raw 5 feat, CE-ish) was
macro-F1 0.21.

Run:  python run_augmented_8class.py [epochs] [hidden]
"""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split, DynamicGraphTemporalSignal
from torch_geometric.nn import GCNConv

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
LR = 0.01
N_CLASSES = 8


class StaticGCN(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, n_classes=N_CLASSES, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_f, hidden); self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden); self.classifier = nn.Linear(hidden, n_classes)
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


def class_weights(y, n_classes=N_CLASSES):
    y = y.cpu().numpy().astype(int)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y); w = torch.ones(n_classes)
    for c, cnt in zip(classes, counts):
        if 0 <= int(c) < n_classes:
            w[int(c)] = n / (len(classes) * cnt)
    return w


def evaluate(model, dataset):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for snap in dataset:
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(snap.y.cpu().numpy().astype(int))
    model.train()
    yp = np.concatenate(preds); yt = np.concatenate(labels)
    pc = f1_score(yt, yp, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    return dict(acc=accuracy_score(yt, yp),
                macro=f1_score(yt, yp, average="macro", zero_division=0),
                weighted=f1_score(yt, yp, average="weighted", zero_division=0),
                mcc=matthews_corrcoef(yt, yp), per_class=pc)


def main():
    log = open("aug_8class.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== AUG 8-CLASS (focal+persnapshot) epochs={EPOCHS} hidden={HIDDEN} ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False,
                                    features_as_self_edge=True, binary=False)
    _ = loader.get_dataset(lags=1)
    T = loader._dataset["time_periods"]; lags = loader.lags

    aug = np.load("data/features_augmented.npy")
    mean = aug.reshape(-1, aug.shape[-1]).mean(0, keepdims=True)
    std  = aug.reshape(-1, aug.shape[-1]).std(0, keepdims=True)
    aug_std = ((aug - mean[None]) / (std[None] + 1e-8)).astype(np.float32)
    in_f = aug.shape[-1]
    feats = [aug_std[i] for i in range(T - lags)]

    signal = DynamicGraphTemporalSignal(loader._edges, loader._edge_weights, feats, loader.targets)
    train_dataset, test_dataset = temporal_signal_split(signal, train_ratio=0.7)
    train_snaps = list(train_dataset)

    torch.manual_seed(0)
    model = StaticGCN(in_f)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(0)

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_snaps))
        tot = 0.0
        for idx in order:
            snap = train_snaps[idx]
            opt.zero_grad()
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            loss = focal_loss(logits, snap.y.long(), class_weights(snap.y), gamma=2.0)
            loss.backward(); opt.step()
            tot += float(loss)
        if epoch % 10 == 0 or epoch == EPOCHS:
            m = evaluate(model, test_dataset)
            out(f"[ep {epoch:3d}] loss={tot/len(train_snaps):.4f} | acc={m['acc']:.4f} "
                f"macroF1={m['macro']:.4f} weightedF1={m['weighted']:.4f} MCC={m['mcc']:.4f}")

    m = evaluate(model, test_dataset)
    out(f"\nFINAL[aug8class] acc={m['acc']:.4f} macroF1={m['macro']:.4f} "
        f"weightedF1={m['weighted']:.4f} MCC={m['mcc']:.4f} | {(time.time()-t0)/60:.1f}min")
    out("per-class F1 (0..7): " + " ".join(f"{f:.3f}" for f in m["per_class"]))
    torch.save(model, "model/aug_8class.pt")
    log.close()


if __name__ == "__main__":
    main()
