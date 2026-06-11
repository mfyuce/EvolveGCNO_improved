"""EvolveGCN rescue attempts on the vehicle-disjoint split.

Untried levers from the audit:
  feat=eng   -> 10 engineered consistency features (like static-eng 0.768)
  mode=carry -> no per-window W reset; W carried (detached) across windows = full-epoch horizon
  mode=alr   -> carry + separate 5x LR param-group for the weight-GRU (anti-collapse)
  mode=std   -> original truncated-BPTT training (reference)

Tracks ||dW||/||W|| during training to verify whether evolution stays alive.
Usage: python run_evolve_rescue.py <evolveo|evolveh> <raw|eng> <std|carry|alr> <seed>
"""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import EvolveGCNO, EvolveGCNH
from torch_geometric.nn import GCNConv

def _snapshot_class_weights(y_tensor: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights (inlined from graphs_base to avoid hiddenlayer dep)."""
    y_np = y_tensor.cpu().numpy().astype(int)
    classes, counts = np.unique(y_np, return_counts=True)
    n = len(y_np)
    w = torch.ones(2, dtype=torch.float32)
    for c, cnt in zip(classes, counts):
        if 0 <= int(c) < 2:
            w[int(c)] = n / (max(len(classes), 1) * cnt)
    return w

MODEL = sys.argv[1] if len(sys.argv) > 1 else "evolveo"
FEAT  = sys.argv[2] if len(sys.argv) > 2 else "eng"
MODE  = sys.argv[3] if len(sys.argv) > 3 else "std"
SEED  = int(sys.argv[4]) if len(sys.argv) > 4 else 3
EPOCHS, WINDOW, HIDDEN, LR = 40, 16, 32, 0.01
TAG = f"rescue_{MODEL}_{FEAT}_{MODE}_s{SEED}"


class Net(nn.Module):
    def __init__(self, node_count, in_f, h=HIDDEN, kind="evolveo", d=0.5):
        super().__init__()
        self.kind = kind
        self.rec = EvolveGCNO(in_f) if kind == "evolveo" else EvolveGCNH(node_count, in_f)
        self.c1 = GCNConv(in_f, h); self.c2 = GCNConv(h, h); self.c3 = GCNConv(h, h)
        self.cl = nn.Linear(h, 2); self.d = d

    def reset(self):
        try:
            self.rec.weight = None
        except Exception:
            object.__setattr__(self.rec, "weight", None)

    def detach_state(self):
        w = getattr(self.rec, "weight", None)
        if w is not None:
            try:
                self.rec.weight = w.detach()
            except Exception:
                object.__setattr__(self.rec, "weight", w.detach())

    def forward(self, x, ei, ew):
        h = self.rec(x, ei, ew)
        h = F.relu(self.c1(h, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.d, training=self.training)
        return self.cl(F.relu(self.c3(h, ei, ew)))


def focal(lg, t, a, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp) * a.to(lg.device)[t]).mean()


def main():
    log = open(f"{TAG}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== {TAG} (focal, vehicle-disjoint) ===")
    t0 = time.time()

    lb = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    ds = lb.get_dataset(lags=1); alls = list(ds)
    n_nodes = len(lb._dataset["node_labels"]); T, lags = lb._dataset["time_periods"], lb.lags
    aug_raw = np.load("data/features_augmented.npy")
    active = [torch.tensor(aug_raw[i, :, 0] != 0.0) for i in range(T - lags)]
    if FEAT == "eng":
        mean = aug_raw.reshape(-1, 10).mean(0, keepdims=True); std = aug_raw.reshape(-1, 10).std(0, keepdims=True)
        aug = ((aug_raw - mean[None]) / (std[None] + 1e-8)).astype(np.float32)
        X = [torch.tensor(aug[i]) for i in range(T - lags)]; in_f = 10
    else:
        X = [a.x for a in alls]; in_f = lb.n_node_features
    EI = [a.edge_index for a in alls]; EW = [a.edge_attr for a in alls]
    Y = [torch.tensor(lb.targets[i], dtype=torch.long) for i in range(T - lags)]

    yf = np.stack([lb.targets[i] for i in range(T - lags)]); node_label = (yf.max(0) > 0).astype(int)
    rng = np.random.default_rng(SEED); trm = np.zeros(n_nodes, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]] = True
    trv, tev = torch.tensor(trm), torch.tensor(~trm)
    n_tr = int(0.7 * (T - lags))
    out(f"test_veh={(~trm).sum()}(mal={node_label[~trm].sum()}) in_f={in_f} mode={MODE}")

    torch.manual_seed(SEED)
    model = Net(n_nodes, in_f=in_f, kind=MODEL)
    if MODE == "alr":
        rec_ids = {id(p) for p in model.rec.parameters()}
        rest = [p for p in model.parameters() if id(p) not in rec_ids]
        opt = torch.optim.Adam([
            {"params": list(model.rec.parameters()), "lr": 5 * LR},
            {"params": rest, "lr": LR},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    def wnorm():
        w = getattr(model.rec, "weight", None)
        return None if w is None else w.detach().reshape(-1).clone()

    def evaluate():
        model.eval(); P, L = [], []
        with torch.no_grad():
            model.reset()
            for i in range(T - lags):
                lg = model(X[i], EI[i], EW[i])
                m = active[i] & tev
                if m.sum() > 0:
                    P.append(torch.softmax(lg, 1)[m, 1].numpy()); L.append(Y[i][m].numpy())
        model.train()
        P = np.concatenate(P); L = np.concatenate(L)
        roc = roc_auc_score(L, P)
        best = max(((matthews_corrcoef(L, (P > t).astype(int)),
                     f1_score(L, (P > t).astype(int), average="macro", zero_division=0), t)
                    for t in np.arange(0.05, 1.0, 0.025) if (P > t).any()), default=(0, 0, 0.5))
        return roc, best[0], best[1], best[2]

    for epoch in range(1, EPOCHS + 1):
        jumps = []
        if MODE != "std":
            model.reset()                     # one reset per EPOCH (full-sequence horizon)
        for st in range(0, n_tr, WINDOW):
            if MODE == "std":
                model.reset()                 # original truncated BPTT
            else:
                model.detach_state()          # carry W across windows, truncate gradients
            opt.zero_grad(); wl = 0.0; c = 0
            prev = wnorm()
            for i in range(st, min(st + WINDOW, n_tr)):
                lg = model(X[i], EI[i], EW[i]); m = active[i] & trv
                cur = wnorm()
                if prev is not None and cur is not None and prev.shape == cur.shape:
                    jumps.append(((cur - prev).norm() / (prev.norm() + 1e-12)).item())
                prev = cur
                if m.sum() == 0:
                    continue
                wl = wl + focal(lg[m], Y[i][m], _snapshot_class_weights(Y[i][m])); c += 1
            if c:
                (wl / c).backward(); opt.step()
        if epoch % 10 == 0 or epoch in (1, EPOCHS):
            roc, mcc, mac, thr = evaluate()
            jm = float(np.mean(jumps)) if jumps else float("nan")
            out(f"[ep {epoch:3d}] ROC={roc:.4f} bestMCC={mcc:.3f}@{thr:.2f} macroF1={mac:.3f} dW/W={jm:.4f}")

    roc, mcc, mac, thr = evaluate()
    out(f"\nFINAL[{TAG}] ROC-AUC={roc:.4f} bestMCC={mcc:.3f}@thr{thr:.2f} macroF1={mac:.3f} | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
