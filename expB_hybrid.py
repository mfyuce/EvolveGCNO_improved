"""Exp B — GNN-embedding -> gradient-boosting hybrid on BuST (vehicle-disjoint), ported from
EvolveGCNOi_CPM/hybrid_cpm.py. Train GConvGRU (node-state) on train vehicles/train-time, extract
its per-node temporal hidden state H (32-d) + its own malicious prob, then train HGB / RF on
[H, gnn_prob, engineered-features] with the vehicle-disjoint split.

Tests whether the GNN's temporal context COMPLEMENTS the explicit physics features (it did on CPM
with noisy feats: 65.1 > RF 61.4, but only TIED RF once position/noise were pruned: 66.9 vs 67.0).
Honest: GNN trained only on train vehicles; test embeddings are forward passes (no label use).

Usage: python expB_hybrid.py <featset:eng10|eng8nopos> <seed> [epochs] [window]
Prints: RESULT hybrid seed=.. gnn_mcc=.. hgb_mcc=.. hgb_auc=.. hgb_malf1=.. rf_mcc=.. rf_auc=.. rf_malf1=..
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_fscore_support
torch.set_num_threads(4)
from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU

FEATSET = sys.argv[1] if len(sys.argv) > 1 else "eng10"
SEED    = int(sys.argv[2]) if len(sys.argv) > 2 else 1
EPOCHS  = int(sys.argv[3]) if len(sys.argv) > 3 else 40
WINDOW  = int(sys.argv[4]) if len(sys.argv) > 4 else 24
HIDDEN, LR = 32, 0.01


class RecGNN(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.rec = GConvGRU(in_f, hidden, 2)
        self.lin1 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)
        h = F.relu(self.lin1(H))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


def focal(lg, t, a, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp) * a[t]).mean()


def cw(y):
    w = torch.ones(2)
    for c in (0, 1):
        n = int((y == c).sum())
        if n > 0:
            w[c] = len(y) / (2.0 * n)
    return w


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n > 0:
            w[y == c] = len(y) / (2.0 * n)
    return w


def bmcc(y, p):
    bt, bm = 0.5, -1.0
    for thr in np.arange(0.05, 1.0, 0.025):
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            continue
        m = matthews_corrcoef(y, pred)
        if m > bm:
            bm, bt = m, thr
    return bm, bt


def metrics(y, p):
    mcc, thr = bmcc(y, p)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float('nan')
    pr = precision_recall_fscore_support(y, (p >= thr).astype(int), average=None,
                                         labels=[0, 1], zero_division=0)
    return mcc * 100, auc, pr[2][1] * 100


def main():
    t0 = time.time()
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = loader.get_dataset(lags=1)
    T = loader._dataset["time_periods"]; lags = loader.lags
    nstep = T - lags

    aug_raw = np.load("data/features_augmented.npy")
    N = aug_raw.shape[1]
    active = aug_raw[..., 0] != 0.0
    mean = aug_raw.reshape(-1, aug_raw.shape[-1]).mean(0, keepdims=True)
    std = aug_raw.reshape(-1, aug_raw.shape[-1]).std(0, keepdims=True)
    aug = ((aug_raw - mean) / (std + 1e-8)).astype(np.float32)
    if FEATSET == "eng8nopos":
        aug = aug[..., [2, 3, 4, 5, 6, 7, 8, 9]]
    in_f = aug.shape[-1]

    y_full = np.zeros((T, N), dtype=np.int64)
    for i in range(nstep):
        y_full[i] = loader.targets[i]
    node_label = (y_full.max(0) > 0).astype(int)

    rng = np.random.default_rng(SEED); trm = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    trv = torch.tensor(trm)

    ei = [torch.tensor(e, dtype=torch.long) for e in loader._edges]
    ew = [torch.tensor(w, dtype=torch.float) for w in loader._edge_weights]
    X = [torch.tensor(aug[i]) for i in range(nstep)]
    Y = [torch.tensor(loader.targets[i], dtype=torch.long) for i in range(nstep)]
    act = [torch.tensor(active[i]) for i in range(nstep)]
    n_train = int(0.7 * nstep)

    # fixed-order active node-step feature matrix / labels / vehicle ids
    Xc, yb, vid = [], [], []
    for t in range(nstep):
        m = active[t].astype(bool)
        Xc.append(aug[t][m]); yb.append((y_full[t][m] != 0).astype(int)); vid.append(np.where(m)[0])
    Xc = np.concatenate(Xc); yb = np.concatenate(yb); vid = np.concatenate(vid)

    torch.manual_seed(SEED)
    model = RecGNN(in_f)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train(); H = None
        for st in range(0, n_train, WINDOW):
            if H is not None:
                H = H.detach()
            opt.zero_grad(); wl = 0.0; cnt = 0
            for i in range(st, min(st + WINDOW, n_train)):
                lg, H = model(X[i], ei[i], ew[i], H)
                m = act[i] & trv
                if m.sum() == 0:
                    continue
                wl = wl + focal(lg[m], Y[i][m], cw(Y[i][m])); cnt += 1
            if cnt:
                (wl / cnt).backward(); opt.step()

    # extract temporal hidden state H + GNN prob for ALL active node-steps (stateful, no label use)
    model.eval(); embs, probs, H = [], [], None
    with torch.no_grad():
        for t in range(nstep):
            Hn = model.rec(X[t], ei[t], ew[t], H)
            z = F.relu(model.lin1(Hn)); lg = model.classifier(z)
            H = Hn
            m = active[t].astype(bool)
            embs.append(Hn.numpy()[m]); probs.append(torch.softmax(lg, 1)[:, 1].numpy()[m])
    EMB = np.concatenate(embs); PB = np.concatenate(probs)
    Xhy = np.concatenate([EMB, PB[:, None], Xc], axis=1)

    tr = trm[vid]; te = ~tr
    gnn_mcc, _ = bmcc(yb[te], PB[te])
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=63, l2_regularization=1.0,
                                         random_state=SEED)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=SEED)
    out = {}
    for name, mk in (('hgb', hgb), ('rf', rf)):
        mk.fit(Xhy[tr], yb[tr], sample_weight=bal_sw(yb[tr]))
        out[name] = metrics(yb[te], mk.predict_proba(Xhy[te])[:, 1])
    print(f"RESULT hybrid featset={FEATSET} seed={SEED} gnn_mcc={gnn_mcc*100:.2f} "
          f"hgb_mcc={out['hgb'][0]:.2f} hgb_auc={out['hgb'][1]:.4f} hgb_malf1={out['hgb'][2]:.2f} "
          f"rf_mcc={out['rf'][0]:.2f} rf_auc={out['rf'][1]:.4f} rf_malf1={out['rf'][2]:.2f} "
          f"feat={Xhy.shape[1]} ({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
