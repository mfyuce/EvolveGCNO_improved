"""8-class attacker-type classification run (labels 0..7)."""

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
from torch_geometric_temporal.signal import temporal_signal_split
from graphs.recurrent.evolvegcnh_improved import EvolveGCNHImproved
from torch_geometric.nn import GCNConv

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
LR = 0.01
HIDDEN = 32
N_CLASSES = 8


def class_weights(y, n_classes=N_CLASSES):
    y = y.cpu().numpy().astype(int)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    w = torch.ones(n_classes, dtype=torch.float32)
    for c, cnt in zip(classes, counts):
        if 0 <= int(c) < n_classes:
            w[int(c)] = n / (len(classes) * cnt)
    return w


class Desat8(nn.Module):
    def __init__(self, node_count, in_f=5, hidden=HIDDEN, n_classes=N_CLASSES, dropout=0.5):
        super().__init__()
        self.recurrent = EvolveGCNHImproved(node_count, in_f)
        self.c1 = GCNConv(in_f, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, n_classes)
        self.dropout = dropout
    def forward(self, x, ei, ew):
        h = self.recurrent(x, ei, ew)           # original (non-evolving) core
        h = F.relu(self.c1(h, ei, ew))
        h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h)


def evaluate(model, dataset):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for snap in dataset:
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(snap.y.cpu().numpy().astype(int))
    model.train()
    y_pred = np.concatenate(preds); y_true = np.concatenate(labels)
    per_class = f1_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                         average=None, zero_division=0)
    return dict(
        acc=accuracy_score(y_true, y_pred),
        macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
        per_class=per_class,
    )


def main():
    log = open("run_8class.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    out(f"=== 8-CLASS honest run (desat arch) epochs={EPOCHS} hidden={HIDDEN} ===")
    t0 = time.time()
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False,
                                    features_as_self_edge=True, binary=False)
    dataset = loader.get_dataset(lags=1)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
    n_nodes = len(loader._dataset["node_labels"])

    # class balance on test
    ty = np.concatenate([s.y.cpu().numpy().astype(int) for s in test_dataset])
    cls, cnt = np.unique(ty, return_counts=True)
    out("test class balance: " + ", ".join(f"{c}:{n}({n/len(ty)*100:.1f}%)" for c, n in zip(cls, cnt)))

    torch.manual_seed(0)
    model = Desat8(n_nodes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        cost = 0.0; n = 0
        for snap in train_dataset:
            logits = model(snap.x, snap.edge_index, snap.edge_attr)
            cw = class_weights(snap.y)
            cost = cost + F.cross_entropy(logits, snap.y.long(), weight=cw)
            n += 1
        (cost / n).backward()
        opt.step()
        if epoch % 30 == 0 or epoch == EPOCHS:
            m = evaluate(model, test_dataset)
            out(f"[ep {epoch:3d}] CE={float(cost)/n:.4f} | acc={m['acc']:.4f} "
                f"macroF1={m['macro']:.4f} weightedF1={m['weighted']:.4f} MCC={m['mcc']:.4f}")

    m = evaluate(model, test_dataset)
    out(f"\nFINAL[8class] acc={m['acc']:.4f} macroF1={m['macro']:.4f} "
        f"weightedF1={m['weighted']:.4f} MCC={m['mcc']:.4f} | {(time.time()-t0)/60:.1f}min")
    out("per-class F1 (0..7): " + " ".join(f"{f:.3f}" for f in m["per_class"]))
    torch.save(model, "model/exp_8class.pt")
    log.close()


if __name__ == "__main__":
    main()
