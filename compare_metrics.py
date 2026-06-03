"""Full metric comparison: published (leaky) paper numbers vs current honest model.

Computes the COMPLETE metric suite (precision/recall/F1 under micro, macro AND
weighted averaging, plus accuracy and MCC) for the honest static-GCN+focal+
augmented-features model, on BOTH the paper's temporal split and the rigorous
vehicle-disjoint split, at the argmax (0.5) operating point (comparable to the
paper) and at the best-MCC threshold.

Paper numbers are quoted from the published tables (main.tex):
  Table 9 (main result, weighted-style, leaky): Acc 99.92 P 99.93 R 99.92 F1 99.88 MCC 97.35
  Table 8 (initial, macro):                     Acc 95.89 P 98.75 R 98.73 F1 98.10

Run:  python compare_metrics.py
"""

import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, matthews_corrcoef,
)
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split, DynamicGraphTemporalSignal
from torch_geometric.nn import GCNConv

EPOCHS, HIDDEN, LR, SEED = 40, 32, 0.01, 3


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


def cw(y):
    cl, c = np.unique(y.cpu().numpy().astype(int), return_counts=True)
    w = torch.ones(2)
    for k, n in zip(cl, c):
        if 0 <= int(k) < 2:
            w[int(k)] = len(y) / (len(cl) * n)
    return w


def full_metrics(y_true, y_pred):
    out = {"acc": accuracy_score(y_true, y_pred), "mcc": matthews_corrcoef(y_true, y_pred)}
    for avg in ("micro", "macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[avg] = (p, r, f)
    return out


def report(tag, probs, labels, out):
    out(f"\n### {tag}")
    out(f"{'operating pt':>14} {'avg':>9} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Acc':>7} {'MCC':>7}")
    # argmax (0.5) — comparable to the paper
    for thr_name, thr in [("argmax(0.5)", 0.5), None_best(probs, labels)]:
        pred = (probs > thr).astype(int)
        m = full_metrics(labels, pred)
        for avg in ("micro", "macro", "weighted"):
            p, r, f = m[avg]
            head = f"{thr_name:>14}" if avg == "micro" else " " * 14
            out(f"{head} {avg:>9} {p*100:7.2f} {r*100:7.2f} {f*100:7.2f} "
                f"{m['acc']*100:7.2f} {m['mcc']*100:7.2f}")


def None_best(probs, labels):
    best = (-2, 0.5)
    for t in np.arange(0.05, 1.0, 0.025):
        if (probs > t).any():
            mc = matthews_corrcoef(labels, (probs > t).astype(int))
            if mc > best[0]:
                best = (mc, t)
    return (f"bestMCC({best[1]:.2f})", best[1])


def build_binary():
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = loader.get_dataset(lags=1)
    T, lags = loader._dataset["time_periods"], loader.lags
    aug = np.load("data/features_augmented.npy")
    mean = aug.reshape(-1, 10).mean(0, keepdims=True); std = aug.reshape(-1, 10).std(0, keepdims=True)
    aug = ((aug - mean[None]) / (std[None] + 1e-8)).astype(np.float32)
    return loader, aug, T, lags


def train_eval_temporal(out):
    loader, aug, T, lags = build_binary()
    feats = [aug[i] for i in range(T - lags)]
    sig = DynamicGraphTemporalSignal(loader._edges, loader._edge_weights, feats, loader.targets)
    tr, te = temporal_signal_split(sig, train_ratio=0.7)
    tr = list(tr)
    torch.manual_seed(SEED); model = StaticGCN(10); opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        for i in rng.permutation(len(tr)):
            s = tr[i]; opt.zero_grad()
            lg = model(s.x, s.edge_index, s.edge_attr)
            focal(lg, s.y.long(), cw(s.y)).backward(); opt.step()
    model.eval(); P, L = [], []
    with torch.no_grad():
        for s in te:
            lg = model(s.x, s.edge_index, s.edge_attr)
            P.append(torch.softmax(lg, 1)[:, 1].numpy()); L.append(s.y.numpy().astype(int))
    return np.concatenate(P), np.concatenate(L)


def train_eval_disjoint(out, featset="all"):
    loader, aug, T, lags = build_binary()
    N = aug.shape[1]
    aug_raw = np.load("data/features_augmented.npy")
    active = aug_raw[..., 0] != 0.0
    if featset == "nopos":
        aug = aug[..., [2, 3, 4, 5, 6, 7, 8, 9]]
    in_f = aug.shape[-1]
    y_full = np.zeros((T, N), dtype=np.int64)
    for i in range(T - lags):
        y_full[i] = loader.targets[i]
    node_label = (y_full.max(0) > 0).astype(int)
    rng = np.random.default_rng(SEED); train_mask = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx); train_mask[idx[:int(0.7*len(idx))]] = True
    trn, ten = torch.tensor(train_mask), torch.tensor(~train_mask)
    ei = [torch.tensor(e, dtype=torch.long) for e in loader._edges]
    ew = [torch.tensor(w, dtype=torch.float) for w in loader._edge_weights]
    X = [torch.tensor(aug[i]) for i in range(T - lags)]
    Y = [torch.tensor(loader.targets[i], dtype=torch.long) for i in range(T - lags)]
    AC = [torch.tensor(active[i]) for i in range(T - lags)]
    n_snap = T - lags; n_tr = int(0.7 * n_snap)
    torch.manual_seed(SEED); model = StaticGCN(in_f); opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng2 = np.random.default_rng(SEED + 100)
    for _ in range(EPOCHS):
        for i in rng2.permutation(n_tr):
            m = AC[i] & trn
            if m.sum() == 0:
                continue
            opt.zero_grad(); lg = model(X[i], ei[i], ew[i]); yt = Y[i][m]
            focal(lg[m], yt, cw(yt)).backward(); opt.step()
    model.eval(); P, L = [], []
    with torch.no_grad():
        for i in range(n_snap):
            m = AC[i] & ten
            if m.sum() == 0:
                continue
            lg = model(X[i], ei[i], ew[i])
            P.append(torch.softmax(lg, 1)[m, 1].numpy()); L.append(Y[i][m].numpy())
    return np.concatenate(P), np.concatenate(L)


def main():
    log = open("compare_metrics.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    t0 = time.time()
    out("=" * 74)
    out("PUBLISHED PAPER (leaky pipeline; quoted from main.tex)")
    out("  Table 9 main result (weighted-style): Acc 99.92  P 99.93  R 99.92  F1 99.88  MCC 97.35")
    out("  Table 8 initial (macro):              Acc 95.89  P 98.75  R 98.73  F1 98.10")
    out("=" * 74)
    out("CURRENT HONEST MODEL (static GCN + focal + 10 augmented features)")

    pT, lT = train_eval_temporal(out)
    report("HONEST — temporal split (paper's protocol, de-leaked)", pT, lT, out)

    pD, lD = train_eval_disjoint(out, "all")
    report("HONEST — vehicle-disjoint (unseen attackers, with position)", pD, lD, out)

    pN, lN = train_eval_disjoint(out, "nopos")
    report("HONEST — vehicle-disjoint, position-independent (transferable)", pN, lN, out)

    out(f"\n(micro P=R=F1=accuracy for single-label; time {(time.time()-t0)/60:.1f}min)")
    log.close()


if __name__ == "__main__":
    main()
