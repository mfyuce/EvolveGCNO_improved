"""Permutation feature importance on a vehicle-disjoint static-GCN (full 10 feat).

Trains the winning recipe vehicle-disjoint (unseen attackers), then for each
feature shuffles that column among active TEST nodes (per snapshot) and measures
the drop in ROC-AUC / MCC. Big drop => the model relies on that feature to
GENERALIZE. Directly answers "which features matter" and "do the engineered
consistency features carry the generalization".

Run:  python run_feature_ablation.py [seed]
"""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric.nn import GCNConv

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EPOCHS, HIDDEN, LR = 40, 32, 0.01
COLS = ["x", "y", "heading", "speed", "accel",
        "pos_jump", "dspeed", "accel_resid", "abs_head_change", "motion_head_resid"]


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


def focal(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = ((1 - logpt.exp()) ** gamma) * (-logpt)
    return (loss * alpha.to(logits.device)[target]).mean()


def cw_from(y):
    cl, cnt = np.unique(y.cpu().numpy().astype(int), return_counts=True)
    w = torch.ones(2)
    for c, n in zip(cl, cnt):
        if 0 <= int(c) < 2:
            w[int(c)] = len(y) / (len(cl) * n)
    return w


def main():
    log = open("feature_ablation.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== PERMUTATION FEATURE IMPORTANCE (vehicle-disjoint, seed={SEED}) ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = loader.get_dataset(lags=1)
    T, lags = loader._dataset["time_periods"], loader.lags
    aug_raw = np.load("data/features_augmented.npy")
    N = aug_raw.shape[1]
    active = aug_raw[..., 0] != 0.0
    mean = aug_raw.reshape(-1, 10).mean(0, keepdims=True)
    std = aug_raw.reshape(-1, 10).std(0, keepdims=True)
    aug = ((aug_raw - mean[None]) / (std[None] + 1e-8)).astype(np.float32)

    y_full = np.zeros((T, N), dtype=np.int64)
    for i in range(T - lags):
        y_full[i] = loader.targets[i]
    node_label = (y_full.max(0) > 0).astype(int)

    rng = np.random.default_rng(SEED)
    train_mask = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx)
        train_mask[idx[:int(0.7 * len(idx))]] = True
    te_node = torch.tensor(~train_mask); tr_node = torch.tensor(train_mask)

    ei = [torch.tensor(e, dtype=torch.long) for e in loader._edges]
    ew = [torch.tensor(w, dtype=torch.float) for w in loader._edge_weights]
    X  = [torch.tensor(aug[i]) for i in range(T - lags)]
    Y  = [torch.tensor(loader.targets[i], dtype=torch.long) for i in range(T - lags)]
    AC = [torch.tensor(active[i]) for i in range(T - lags)]
    n_snap = T - lags; n_tr = int(0.7 * n_snap)

    torch.manual_seed(SEED)
    model = StaticGCN(10); opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng2 = np.random.default_rng(SEED + 100)
    for epoch in range(1, EPOCHS + 1):
        for i in rng2.permutation(n_tr):
            m = AC[i] & tr_node
            if m.sum() == 0:
                continue
            opt.zero_grad()
            lg = model(X[i], ei[i], ew[i])
            yt = Y[i][m]
            focal(lg[m], yt, cw_from(yt)).backward(); opt.step()

    def eval_perm(perm_k=None, prng=None):
        model.eval(); probs, labels = [], []
        with torch.no_grad():
            for i in range(n_snap):
                m = AC[i] & te_node
                if m.sum() == 0:
                    continue
                xi = X[i]
                if perm_k is not None:
                    xi = xi.clone()
                    act_idx = torch.where(m)[0]
                    perm = act_idx[torch.tensor(prng.permutation(len(act_idx)))]
                    xi[act_idx, perm_k] = X[i][perm, perm_k]
                lg = model(xi, ei[i], ew[i])
                probs.append(torch.softmax(lg, 1)[m, 1].numpy())
                labels.append(Y[i][m].numpy())
        p = np.concatenate(probs); l = np.concatenate(labels)
        roc = roc_auc_score(l, p)
        best = max((matthews_corrcoef(l, (p > t).astype(int))
                    for t in np.arange(0.05, 1.0, 0.025) if (p > t).any()), default=0)
        return roc, best

    base_roc, base_mcc = eval_perm()
    out(f"baseline (no perm): ROC-AUC={base_roc:.4f}  best-MCC={base_mcc:.4f}\n")
    out(f"{'feature':>18} {'ROC drop':>10} {'MCC drop':>10}")
    results = []
    for k, c in enumerate(COLS):
        prng = np.random.default_rng(1000 + k)
        roc, mcc = eval_perm(perm_k=k, prng=prng)
        results.append((c, base_roc - roc, base_mcc - mcc))
    for c, dr, dm in sorted(results, key=lambda r: -r[2]):
        eng = "  (engineered)" if c in COLS[5:] else ""
        out(f"{c:>18} {dr:10.4f} {dm:10.4f}{eng}")
    out(f"\n(larger drop = more important)  | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
