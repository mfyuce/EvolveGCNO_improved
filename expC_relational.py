"""Exp C — does RELATIONAL (neighbor-consistency) information let a GNN earn its place over RF?

The engineered features are all PER-NODE (a vehicle vs its own past). A GNN can additionally model
how a vehicle's report disagrees with its NEIGHBOURS' reports (relational implausibility) — something
a per-node RF cannot do structurally. This script isolates that question with 4 modes:

  rf_eng   RandomForest on engineered-10           (baseline; ~RF 74.7)
  rf_rel   RandomForest on engineered-10 + 5 relational features   (can RF use precomputed relational scalars?)
  gnn_rel  GConvGRU on engineered-10 + 5 relational features       (relational info as node input)
  gnn_edge GConvGRU temporal state -> GATv2 with edge_attr = pairwise neighbour-disagreement
           (LEARNED multi-hop relational aggregation — the only thing RF cannot replicate)

Verdict logic: the GNN "earns its place" iff gnn_edge > rf_rel (learned relational beats precomputed
scalars). If rf_rel already captures the relational signal, the GNN adds nothing.

Relational node feats (from RAW kinematics over the 5-NN graph, self excluded): in_degree,
speed-dev-from-neighbour-mean, accel-dev, heading-dev (circular), position-dev-from-neighbour-centroid.
Edge_attr (per directed edge i->j): |dspeed|, circular |dheading|, |daccel|, reported distance.

Usage: python expC_relational.py <mode:rf_eng|rf_rel|gnn_rel|gnn_edge> <seed> [epochs] [window]
Prints: RESULT expC mode=.. seed=.. mcc=.. auc=.. malf1=.. macrof1=.. (+ gnn_mcc for gnn modes)
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, f1_score,
                             precision_recall_fscore_support)
torch.set_num_threads(2)
from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.nn import GATv2Conv, GCNConv

MODE   = sys.argv[1] if len(sys.argv) > 1 else "gnn_edge"
SEED   = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
WINDOW = int(sys.argv[4]) if len(sys.argv) > 4 else 24
HIDDEN, LR = 32, 0.01
# static_eng/static_rel = recurrence-OFF controls (3-layer static GCN, no temporal state)
need_rel = MODE in ("rf_rel", "gnn_rel", "gnn_edge", "static_rel")
need_edge = MODE == "gnn_edge"


def ang_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def compute_relational(aug_raw, edges, ews, nstep, N):
    """Return rel node feats (nstep,N,5) raw, and edge_attr list [(E_t,4)] raw."""
    rel = np.zeros((nstep, N, 5), dtype=np.float32)
    edge_attrs = []
    for t in range(nstep):
        ei_t = edges[t]
        src, dst = ei_t[0].astype(int), ei_t[1].astype(int)
        xt = aug_raw[t]
        px, py, head, spd, acc = xt[:, 0], xt[:, 1], xt[:, 2], xt[:, 3], xt[:, 4]
        nm = src != dst
        s, d = src[nm], dst[nm]
        deg = np.bincount(d, minlength=N).astype(np.float32)
        degc = np.maximum(deg, 1.0)
        spd_nb = np.bincount(d, weights=spd[s], minlength=N) / degc
        acc_nb = np.bincount(d, weights=acc[s], minlength=N) / degc
        sin_nb = np.bincount(d, weights=np.sin(np.radians(head[s])), minlength=N) / degc
        cos_nb = np.bincount(d, weights=np.cos(np.radians(head[s])), minlength=N) / degc
        head_nb = np.degrees(np.arctan2(sin_nb, cos_nb)) % 360.0
        cx = np.bincount(d, weights=px[s], minlength=N) / degc
        cy = np.bincount(d, weights=py[s], minlength=N) / degc
        spd_dev = spd - spd_nb
        acc_dev = acc - acc_nb
        head_dev = ang_diff_deg(head, head_nb)
        pos_dev = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        iso = deg == 0
        for arr in (spd_dev, acc_dev, head_dev, pos_dev):
            arr[iso] = 0.0
        rel[t] = np.stack([deg, spd_dev, acc_dev, head_dev, pos_dev], axis=1)
        if need_edge:
            ea = np.stack([
                np.abs(spd[src] - spd[dst]),
                ang_diff_deg(head[src], head[dst]),
                np.abs(acc[src] - acc[dst]),
                np.asarray(ews[t], dtype=np.float32),
            ], axis=1).astype(np.float32)
            edge_attrs.append(ea)
    return rel, edge_attrs


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


def best_eval(y, p):
    bt, bm = 0.5, -1.0
    for thr in np.arange(0.05, 1.0, 0.025):
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            continue
        m = matthews_corrcoef(y, pred)
        if m > bm:
            bm, bt = m, thr
    pred = (p >= bt).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float('nan')
    pr = precision_recall_fscore_support(y, pred, average=None, labels=[0, 1], zero_division=0)
    macro = f1_score(y, pred, average="macro", zero_division=0)
    return bm * 100, auc, pr[2][1] * 100, macro * 100


class RecGNN(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.rec = GConvGRU(in_f, hidden, 2)
        self.lin1 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, ea=None, H=None):
        H = self.rec(x, ei, ew, H)
        h = F.relu(self.lin1(H))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


class RelEdgeGNN(nn.Module):
    """Node-state recurrence (per-vehicle memory) THEN learned relational attention over
    neighbour-disagreement edges (GATv2 with edge features)."""
    def __init__(self, in_f, edge_dim, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.rec = GConvGRU(in_f, hidden, 2)
        self.gat = GATv2Conv(hidden, hidden, heads=2, concat=False,
                             edge_dim=edge_dim, add_self_loops=False)
        self.lin1 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, ea=None, H=None):
        H = self.rec(x, ei, ew, H)
        g = F.elu(self.gat(H, ei, edge_attr=ea))
        h = F.relu(self.lin1(g))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


class StaticGCN(nn.Module):
    """Recurrence-OFF control: 3-layer static GCN (same arch as committed static-eng), no temporal
    hidden state. Isolates whether the GNN's win over RF is the temporal recurrence or just the GCN."""
    def __init__(self, in_f, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_f, hidden); self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, ea=None, H=None):
        h = F.relu(self.c1(x, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h), None


def main():
    t0 = time.time()
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = loader.get_dataset(lags=1)
    T = loader._dataset["time_periods"]; lags = loader.lags
    nstep = T - lags

    aug_raw = np.load("data/features_augmented.npy")          # (T, N, 10)
    N = aug_raw.shape[1]
    active = aug_raw[..., 0] != 0.0

    def zscore(a):
        m = a.reshape(-1, a.shape[-1]).mean(0, keepdims=True)
        s = a.reshape(-1, a.shape[-1]).std(0, keepdims=True)
        return ((a - m) / (s + 1e-8)).astype(np.float32)

    eng = zscore(aug_raw)                                     # (T,N,10) standardized
    edges = loader._edges; ews = loader._edge_weights

    if need_rel:
        rel_raw, edge_attrs = compute_relational(aug_raw, edges, ews, nstep, N)
        rel = zscore(rel_raw)                                 # (nstep,N,5)
        feats = np.concatenate([eng[:nstep], rel], axis=2)    # (nstep,N,15)
    else:
        feats = eng[:nstep]
    in_f = feats.shape[-1]

    if need_edge:
        allea = np.concatenate(edge_attrs, axis=0)
        em = allea.mean(0, keepdims=True); es = allea.std(0, keepdims=True)
        edge_attrs = [torch.tensor((ea - em) / (es + 1e-8), dtype=torch.float) for ea in edge_attrs]

    y_full = np.zeros((T, N), dtype=np.int64)
    for i in range(nstep):
        y_full[i] = loader.targets[i]
    node_label = (y_full.max(0) > 0).astype(int)

    rng = np.random.default_rng(SEED); trm = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    n_train = int(0.7 * nstep)

    # ---------- RF modes ----------
    if MODE in ("rf_eng", "rf_rel"):
        Xc, yb, vid = [], [], []
        for t in range(nstep):
            m = active[t].astype(bool)
            Xc.append(feats[t][m]); yb.append((y_full[t][m] != 0).astype(int)); vid.append(np.where(m)[0])
        Xc = np.concatenate(Xc); yb = np.concatenate(yb); vid = np.concatenate(vid)
        tr = trm[vid]; te = ~tr
        rf = RandomForestClassifier(n_estimators=300, max_features=0.5, n_jobs=4, random_state=SEED)
        rf.fit(Xc[tr], yb[tr], sample_weight=bal_sw(yb[tr]))
        mcc, auc, malf1, macrof1 = best_eval(yb[te], rf.predict_proba(Xc[te])[:, 1])
        print(f"RESULT expC mode={MODE} seed={SEED} mcc={mcc:.2f} auc={auc:.4f} "
              f"malf1={malf1:.2f} macrof1={macrof1:.2f} in_f={in_f} ({(time.time()-t0)/60:.1f}min)", flush=True)
        return

    # ---------- GNN modes ----------
    ei = [torch.tensor(e, dtype=torch.long) for e in edges]
    ew = [torch.tensor(w, dtype=torch.float) for w in ews]
    X = [torch.tensor(feats[i]) for i in range(nstep)]
    Y = [torch.tensor(loader.targets[i], dtype=torch.long) for i in range(nstep)]
    act = [torch.tensor(active[i]) for i in range(nstep)]
    trv = torch.tensor(trm); tev = torch.tensor(~trm)

    torch.manual_seed(SEED)
    if MODE == "gnn_edge":
        model = RelEdgeGNN(in_f, edge_dim=edge_attrs[0].shape[1])
    elif MODE in ("static_eng", "static_rel"):
        model = StaticGCN(in_f)
    else:
        model = RecGNN(in_f)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def ea_of(i):
        return edge_attrs[i] if need_edge else None

    def evaluate():
        model.eval(); P, L = [], []; H = None
        with torch.no_grad():
            for i in range(nstep):
                logits, H = model(X[i], ei[i], ew[i], ea_of(i), H)
                m = act[i] & tev
                if m.sum() > 0:
                    P.append(torch.softmax(logits, 1)[m, 1].numpy()); L.append(Y[i][m].numpy())
        model.train()
        return best_eval(np.concatenate(L), np.concatenate(P))

    for ep in range(1, EPOCHS + 1):
        H = None
        for st in range(0, n_train, WINDOW):
            if H is not None:
                H = H.detach()
            opt.zero_grad(); wl = 0.0; cnt = 0
            for i in range(st, min(st + WINDOW, n_train)):
                logits, H = model(X[i], ei[i], ew[i], ea_of(i), H)
                m = act[i] & trv
                if m.sum() == 0:
                    continue
                wl = wl + focal(logits[m], Y[i][m], cw(Y[i][m])); cnt += 1
            if cnt:
                (wl / cnt).backward(); opt.step()

    mcc, auc, malf1, macrof1 = evaluate()
    print(f"RESULT expC mode={MODE} seed={SEED} mcc={mcc:.2f} auc={auc:.4f} "
          f"malf1={malf1:.2f} macrof1={macrof1:.2f} in_f={in_f} ({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
