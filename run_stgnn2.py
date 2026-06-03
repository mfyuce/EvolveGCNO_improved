"""ST-GNN repair (residual skip) + vehicle-disjoint validation.

modes:
  static     : x -> head (no core)
  evolve     : x -> EvolveGCN-H core (BPTT) -> head
  evolve_res : x -> x + core(x) (RESIDUAL skip around core) -> head
               => bypasses the core's over-smoothing; if temporal is useless the
                  model can drive core(x)->0 and recover static performance.

splits:
  temporal : first 70% timesteps -> train (paper protocol)
  disjoint : stratified vehicle split (unseen attackers); loss/eval masked to
             train/test vehicles, trained on first-70%-snapshots in order.

All: raw 5 features, focal, in-order windowed truncated BPTT, de-saturated head.

Run:  python run_stgnn2.py <static|evolve|evolve_res> <temporal|disjoint> [epochs] [window] [seed]
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
from torch_geometric_temporal.signal import temporal_signal_split
from graphs.recurrent.evolvegcnh_improved import EvolveGCNHImproved
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

MODE   = sys.argv[1] if len(sys.argv) > 1 else "evolve_res"
SPLIT  = sys.argv[2] if len(sys.argv) > 2 else "temporal"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
WINDOW = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SEED   = int(sys.argv[5]) if len(sys.argv) > 5 else 3
HIDDEN, LR = 32, 0.01
TAG = f"stgnn2_{MODE}_{SPLIT}_s{SEED}"


class STGNN(nn.Module):
    def __init__(self, node_count, in_f=5, hidden=HIDDEN, mode="evolve_res", dropout=0.5):
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        if mode in ("evolve", "evolve_res"):
            self.recurrent = EvolveGCNHImproved(node_count, in_f)
        self.c1 = GCNConv(in_f, hidden); self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden); self.classifier = nn.Linear(hidden, 2)

    def reset_hidden_state(self):
        if self.mode in ("evolve", "evolve_res"):
            object.__setattr__(self.recurrent, "weight", None)

    def _core(self, x, ei, ew):
        X_tilde = self.recurrent.pooling_layer(x, ei)
        X_tilde = X_tilde[0][None, :, :]
        if self.recurrent.weight is None:
            object.__setattr__(self.recurrent, "weight", self.recurrent.initial_weight)
        W_in = self.recurrent.weight[None, :, :]
        _, W_out = self.recurrent.recurrent_layer(X_tilde, W_in)
        W_new = W_out.squeeze(dim=0)
        object.__setattr__(self.recurrent, "weight", W_new)
        return self.recurrent.conv_layer(W_new, x, ei, ew)

    def forward(self, x, ei, ew):
        if self.mode == "static":
            h = x
        elif self.mode == "evolve":
            h = self._core(x, ei, ew)
        else:  # evolve_res: residual skip preserves raw signal
            h = x + self._core(x, ei, ew)
        h = F.relu(self.c1(h, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h)


def focal(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return (((1 - logpt.exp()) ** gamma) * (-logpt) * alpha.to(logits.device)[target]).mean()


def best_metrics(probs, labels):
    roc = roc_auc_score(labels, probs); pr = average_precision_score(labels, probs)
    best = (-2, None, None)
    for t in np.arange(0.05, 1.0, 0.025):
        pred = (probs > t).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(labels, pred)
        if mcc > best[0]:
            best = (mcc, t, dict(
                P=precision_score(labels, pred, pos_label=1, zero_division=0),
                R=recall_score(labels, pred, pos_label=1, zero_division=0),
                macro=f1_score(labels, pred, average="macro", zero_division=0)))
    return dict(roc=roc, pr=pr, mcc=best[0], thr=best[1], **(best[2] or {}))


def main():
    log = open(f"{TAG}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== {TAG} (raw5, focal, in-order BPTT W={WINDOW}) epochs={EPOCHS} ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    ds = loader.get_dataset(lags=1)
    tr_ds, te_ds = temporal_signal_split(ds, train_ratio=0.7)
    train_all = list(ds) if SPLIT == "disjoint" else None
    tr, te = list(tr_ds), list(te_ds)
    n_nodes = len(loader._dataset["node_labels"])

    # vehicle masks for disjoint
    if SPLIT == "disjoint":
        T = loader._dataset["time_periods"]; lags = loader.lags
        yf = np.zeros((T - lags, n_nodes), dtype=int)
        for i in range(T - lags):
            yf[i] = loader.targets[i]
        node_label = (yf.max(0) > 0).astype(int)
        rng = np.random.default_rng(SEED); trm = np.zeros(n_nodes, dtype=bool)
        for c in (0, 1):
            idx = np.where(node_label == c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]] = True
        train_vehicle = torch.tensor(trm); test_vehicle = torch.tensor(~trm)
        # presence per node per snapshot from raw x (col0) != 0
        aug_raw = np.load("data/features_augmented.npy")
        active = [torch.tensor(aug_raw[i, :, 0] != 0.0) for i in range(T - lags)]
        train_snaps = tr            # first-70%-snapshots, in order, masked to train vehicles
        out(f"disjoint: train_veh={trm.sum()} (mal={node_label[trm].sum()}) "
            f"test_veh={(~trm).sum()} (mal={node_label[~trm].sum()})")
    else:
        train_snaps = tr

    torch.manual_seed(SEED)
    model = STGNN(n_nodes, in_f=loader.n_node_features, mode=MODE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    def evaluate():
        model.eval(); P, L = [], []
        with torch.no_grad():
            model.reset_hidden_state()
            for i, s in enumerate(te if SPLIT == "temporal" else range(len(active))):
                if SPLIT == "temporal":
                    lg = model(s.x, s.edge_index, s.edge_attr)
                    P.append(torch.softmax(lg, 1)[:, 1].numpy()); L.append(s.y.numpy().astype(int))
                else:
                    snap = train_all[i]; m = active[i] & test_vehicle
                    lg = model(snap.x, snap.edge_index, snap.edge_attr)
                    if m.sum() > 0:
                        P.append(torch.softmax(lg, 1)[m, 1].numpy()); L.append(snap.y[m].numpy().astype(int))
        model.train()
        return best_metrics(np.concatenate(P), np.concatenate(L))

    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(train_snaps), WINDOW):
            win_idx = range(start, min(start + WINDOW, len(train_snaps)))
            model.reset_hidden_state(); opt.zero_grad()
            wl = 0.0; cnt = 0
            for i in win_idx:
                s = train_snaps[i]
                lg = model(s.x, s.edge_index, s.edge_attr)
                if SPLIT == "disjoint":
                    m = active[i] & train_vehicle
                    if m.sum() == 0:
                        continue
                    wl = wl + focal(lg[m], s.y[m].long(), base._snapshot_class_weights(s.y[m])); cnt += 1
                else:
                    wl = wl + focal(lg, s.y.long(), base._snapshot_class_weights(s.y)); cnt += 1
            if cnt == 0:
                continue
            (wl / cnt).backward(); opt.step()
        if epoch % 15 == 0 or epoch == EPOCHS:
            m = evaluate()
            out(f"[ep {epoch:3d}] ROC={m['roc']:.4f} PR={m['pr']:.4f} bestMCC={m['mcc']:.3f}@{m['thr']:.2f} "
                f"P={m['P']:.3f} R={m['R']:.3f} macroF1={m['macro']:.3f}")

    m = evaluate()
    out(f"\nFINAL[{TAG}] ROC-AUC={m['roc']:.4f} PR-AUC={m['pr']:.4f} bestMCC={m['mcc']:.3f}@thr{m['thr']:.2f} "
        f"P={m['P']:.3f} R={m['R']:.3f} macroF1={m['macro']:.3f} | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
