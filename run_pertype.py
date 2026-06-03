"""Where does the temporal advantage come from? Per-attack-type recall.

Trains GConvGRU (node-state temporal) and static GCN on the SAME vehicle-disjoint
split (raw5, focal), then breaks down detection recall by the TRUE attack type
(0..7) on unseen test vehicles. Hypothesis: the temporal model's edge concentrates
on temporally-defined attacks (1=Const Random Position / frozen, 4=Const Random
Speed) that need history to spot, vs per-frame types.

Run:  python run_pertype.py [seed]
"""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EPOCHS, WINDOW, HIDDEN, LR = 40, 24, 32, 0.01
ATTACK_NAMES = {0: "Benign", 1: "ConstRandPos", 2: "PosOffset+", 3: "PosOffset-",
                4: "ConstRandSpd", 5: "SpdOffset+", 6: "SpdOffset-", 7: "RevHeading"}


class GConvGRUNet(nn.Module):
    def __init__(self, in_f=5, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.rec = GConvGRU(in_f, hidden, 2)
        self.lin1 = nn.Linear(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout
    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)
        h = F.relu(self.lin1(H)); h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


class StaticGCN(nn.Module):
    def __init__(self, in_f=5, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_f, hidden); self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout
    def forward(self, x, ei, ew):
        h = F.relu(self.c1(x, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training); h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h)


def focal(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return (((1 - logpt.exp()) ** gamma) * (-logpt) * alpha.to(logits.device)[target]).mean()


def main():
    log = open("pertype.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== PER-ATTACK-TYPE recall: GConvGRU vs static (disjoint seed={SEED}) ===")
    t0 = time.time()

    lb = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True, binary=True)
    ds = lb.get_dataset(lags=1)
    l8 = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True, binary=False)
    _ = l8.get_dataset(lags=1)
    alls = list(ds); n_nodes = len(lb._dataset["node_labels"]); T, lags = lb._dataset["time_periods"], lb.lags

    yb = np.stack([lb.targets[i] for i in range(T - lags)])     # (T', N) binary
    y8 = np.stack([l8.targets[i] for i in range(T - lags)])     # (T', N) 0..7
    node_label = (yb.max(0) > 0).astype(int)
    rng = np.random.default_rng(SEED); trm = np.zeros(n_nodes, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]] = True
    train_vehicle = torch.tensor(trm); test_vehicle = torch.tensor(~trm)
    aug_raw = np.load("data/features_augmented.npy")
    active = [torch.tensor(aug_raw[i, :, 0] != 0.0) for i in range(T - lags)]
    n_train = int(0.7 * (T - lags))
    out(f"test_veh={(~trm).sum()} (mal={node_label[~trm].sum()})")

    def train_static():
        torch.manual_seed(SEED); m = StaticGCN(); opt = torch.optim.Adam(m.parameters(), lr=LR)
        for _ in range(EPOCHS):
            for st in range(0, n_train, WINDOW):
                opt.zero_grad(); wl = 0.0; c = 0
                for i in range(st, min(st + WINDOW, n_train)):
                    s = alls[i]; msk = active[i] & train_vehicle
                    if msk.sum() == 0:
                        continue
                    lg = m(s.x, s.edge_index, s.edge_attr)
                    wl = wl + focal(lg[msk], s.y[msk].long(), base._snapshot_class_weights(s.y[msk])); c += 1
                if c:
                    (wl / c).backward(); opt.step()
        return m

    def train_gconv():
        torch.manual_seed(SEED); m = GConvGRUNet(); opt = torch.optim.Adam(m.parameters(), lr=LR)
        for _ in range(EPOCHS):
            H = None
            for st in range(0, n_train, WINDOW):
                if H is not None:
                    H = H.detach()
                opt.zero_grad(); wl = 0.0; c = 0
                for i in range(st, min(st + WINDOW, n_train)):
                    s = alls[i]; msk = active[i] & train_vehicle
                    lg, H = m(s.x, s.edge_index, s.edge_attr, H)
                    if msk.sum() == 0:
                        continue
                    wl = wl + focal(lg[msk], s.y[msk].long(), base._snapshot_class_weights(s.y[msk])); c += 1
                if c:
                    (wl / c).backward(); opt.step()
        return m

    def collect(m, is_rec):
        m.eval(); probs, binl, typl = [], [], []; H = None
        with torch.no_grad():
            for i, s in enumerate(alls):
                if is_rec:
                    lg, H = m(s.x, s.edge_index, s.edge_attr, H)
                else:
                    lg = m(s.x, s.edge_index, s.edge_attr)
                msk = active[i] & test_vehicle
                if msk.sum() == 0:
                    continue
                probs.append(torch.softmax(lg, 1)[msk, 1].numpy())
                binl.append(yb[i][msk.numpy()]); typl.append(y8[i][msk.numpy()])
        return np.concatenate(probs), np.concatenate(binl), np.concatenate(typl)

    out("training static..."); ms = train_static()
    out("training gconvgru..."); mg = train_gconv()
    ps, bs, ts = collect(ms, False)
    pg, bg, tg = collect(mg, True)

    def best_thr(p, b):
        return max(((matthews_corrcoef(b, (p > t).astype(int)), t)
                    for t in np.arange(0.05, 1.0, 0.025) if (p > t).any()), default=(0, 0.5))

    mcc_s, th_s = best_thr(ps, bs); mcc_g, th_g = best_thr(pg, bg)
    out(f"\nstatic   bestMCC={mcc_s:.3f}@{th_s:.2f}   GConvGRU bestMCC={mcc_g:.3f}@{th_g:.2f}")
    out(f"\n{'attack type':>16} {'support':>8} {'static R':>9} {'GConvGRU R':>11}  {'Δ':>6}")
    for t in range(1, 8):
        sup = int((ts == t).sum())
        if sup == 0:
            out(f"{ATTACK_NAMES[t]:>16} {0:>8}  (absent in test)"); continue
        rs = float(((ps > th_s)[ts == t]).mean())
        rg = float(((pg > th_g)[tg == t]).mean())
        out(f"{ATTACK_NAMES[t]:>16} {sup:>8} {rs:9.3f} {rg:11.3f}  {rg-rs:+6.3f}")
    # benign false-positive rate
    fps = float(((ps > th_s)[ts == 0]).mean()); fpg = float(((pg > th_g)[tg == 0]).mean())
    out(f"{'Benign FP rate':>16} {int((ts==0).sum()):>8} {fps:9.3f} {fpg:11.3f}  {fpg-fps:+6.3f}")
    out(f"\n| {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
