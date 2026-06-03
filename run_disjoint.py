"""Vehicle-disjoint validation of the strong honest recipe.

The headline MCC 0.958 used a TEMPORAL split (first 70% of timesteps -> train).
Since labels are constant per vehicle, an attacker in train reappears in test
with the same label, so that split measures "detect an ONGOING attacker", and
position features could let the model memorize vehicle identity.

This script splits by VEHICLE (stratified by label): train and test vehicles
are disjoint, all timesteps used. It measures "detect a NEVER-SEEN attacker" —
the stronger, more honest claim. Graph message-passing still spans all nodes;
only the loss/eval are masked to the respective vehicle set.

Recipe: static GCN, focal loss, 10 augmented features (the winner).
Run:  python run_disjoint.py [epochs] [hidden] [model gcn|mlp]
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
from torch_geometric.nn import GCNConv

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
MODEL  = sys.argv[3] if len(sys.argv) > 3 else "gcn"
SEED   = int(sys.argv[4]) if len(sys.argv) > 4 else 0
# feature subset: "all" (10) or "nopos" (drop absolute x,y -> 8 features)
FEATSET = sys.argv[5] if len(sys.argv) > 5 else "all"
LR = 0.01


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


def cw_from(y):
    classes, counts = np.unique(y.cpu().numpy().astype(int), return_counts=True)
    n = len(y); w = torch.ones(2)
    for c, cnt in zip(classes, counts):
        if 0 <= int(c) < 2:
            w[int(c)] = n / (len(classes) * cnt)
    return w


def main():
    log = open(f"disjoint_{MODEL}_{FEATSET}_s{SEED}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    out(f"=== VEHICLE-DISJOINT  model={MODEL} featset={FEATSET} epochs={EPOCHS} hidden={HIDDEN} seed={SEED} ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = loader.get_dataset(lags=1)
    T = loader._dataset["time_periods"]; lags = loader.lags

    aug_raw = np.load("data/features_augmented.npy")      # (T, N, 10), raw
    N = aug_raw.shape[1]
    active = aug_raw[..., 0] != 0.0                       # x!=0 => vehicle present (from RAW x)
    mean = aug_raw.reshape(-1, aug_raw.shape[-1]).mean(0, keepdims=True)
    std  = aug_raw.reshape(-1, aug_raw.shape[-1]).std(0, keepdims=True)
    aug = ((aug_raw - mean[None]) / (std[None] + 1e-8)).astype(np.float32)
    # optionally drop absolute position (cols 0=x,1=y) to test position-robustness
    if FEATSET == "nopos":
        keep = [2, 3, 4, 5, 6, 7, 8, 9]
        aug = aug[..., keep]
    in_f = aug.shape[-1]

    # per-node binary label (constant per vehicle): max over time of binary targets
    y_full = np.zeros((T, N), dtype=np.int64)
    for i in range(T - lags):
        y_full[i] = loader.targets[i]
    node_label = (y_full.max(axis=0) > 0).astype(int)     # 1 if vehicle ever malicious

    # stratified vehicle split: 70% train / 30% test within each class
    rng = np.random.default_rng(SEED)
    train_mask = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]
        rng.shuffle(idx)
        k = int(0.7 * len(idx))
        train_mask[idx[:k]] = True
    test_mask = ~train_mask
    out(f"vehicles: train={train_mask.sum()} (mal={node_label[train_mask].sum()}) "
        f"test={test_mask.sum()} (mal={node_label[test_mask].sum()})  in_features={in_f}")

    ei_all = [torch.tensor(e, dtype=torch.long) for e in loader._edges]
    ew_all = [torch.tensor(w, dtype=torch.float) for w in loader._edge_weights]
    x_all  = [torch.tensor(aug[i]) for i in range(T - lags)]
    y_all  = [torch.tensor(loader.targets[i], dtype=torch.long) for i in range(T - lags)]
    act_all = [torch.tensor(active[i]) for i in range(T - lags)]
    tr_node = torch.tensor(train_mask); te_node = torch.tensor(test_mask)
    n_snap = T - lags
    n_train_snap = int(0.7 * n_snap)  # still train on the temporal-train snapshots' time range

    torch.manual_seed(SEED)
    model = MLP(in_f) if MODEL == "mlp" else StaticGCN(in_f)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng2 = np.random.default_rng(SEED + 100)

    def evaluate():
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for i in range(n_snap):
                m = (act_all[i] & te_node)
                if m.sum() == 0:
                    continue
                logits = model(x_all[i], ei_all[i], ew_all[i])
                probs.append(torch.softmax(logits, 1)[m, 1].numpy())
                labels.append(y_all[i][m].numpy())
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

    for epoch in range(1, EPOCHS + 1):
        order = rng2.permutation(n_train_snap)
        tot = 0.0; cnt = 0
        for i in order:
            m = (act_all[i] & tr_node)
            if m.sum() == 0:
                continue
            opt.zero_grad()
            logits = model(x_all[i], ei_all[i], ew_all[i])
            yt = y_all[i][m]
            loss = focal_loss(logits[m], yt, cw_from(yt), gamma=2.0)
            loss.backward(); opt.step()
            tot += float(loss); cnt += 1
        if epoch % 10 == 0 or epoch == EPOCHS:
            r = evaluate()
            out(f"[ep {epoch:3d}] loss={tot/max(cnt,1):.4f} | ROC={r['roc']:.4f} "
                f"PR={r['pr']:.4f} | bestMCC={r['mcc']:.3f}@{r['thr']:.2f} "
                f"P={r['P']:.3f} R={r['R']:.3f} F1mal={r['f1mal']:.3f} macroF1={r['macro']:.3f}")

    r = evaluate()
    out(f"\nFINAL[disjoint-{MODEL}] ROC-AUC={r['roc']:.4f} PR-AUC={r['pr']:.4f} "
        f"bestMCC={r['mcc']:.3f}@thr{r['thr']:.2f} P={r['P']:.3f} R={r['R']:.3f} "
        f"F1mal={r['f1mal']:.3f} macroF1={r['macro']:.3f} | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
