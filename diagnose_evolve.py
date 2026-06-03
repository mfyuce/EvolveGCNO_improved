"""Diagnostic comparing recurrent-core variants (no core / fixed weight / GRU-updated weight) of the EvolveGCN-H cell."""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, average_precision_score
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from graphs.recurrent.evolvegcnh_improved import EvolveGCNHImproved
from torch_geometric.nn import GCNConv
import graphs.recurrent.graphs_base as base

MODE   = sys.argv[1] if len(sys.argv) > 1 else "core_gru"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
WINDOW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
HIDDEN, LR = 32, 0.01
TAG = f"diag_{MODE}"


class DiagGCN(nn.Module):
    def __init__(self, node_count, in_f=5, hidden=HIDDEN, mode="core_gru", dropout=0.5):
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        if mode in ("core_fixed", "core_gru"):
            self.recurrent = EvolveGCNHImproved(node_count, in_f)
        self.c1 = GCNConv(in_f, hidden); self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self._last_W = None        # for jumpiness logging (core_gru)

    def reset_hidden_state(self):
        if self.mode == "core_gru":
            object.__setattr__(self.recurrent, "weight", None)

    def _core(self, x, ei, ew):
        if self.mode == "core_fixed":
            # constant pre-conv transform: W = learned-but-fixed initial_weight
            W = self.recurrent.initial_weight
            return self.recurrent.conv_layer(W, x, ei, ew)
        # core_gru: evolve W via GRU over TopK-pooled nodes (proper BPTT across window)
        X_tilde = self.recurrent.pooling_layer(x, ei)
        X_tilde = X_tilde[0][None, :, :]
        if self.recurrent.weight is None:
            object.__setattr__(self.recurrent, "weight", self.recurrent.initial_weight)
        W_in = self.recurrent.weight[None, :, :]
        _, W_out = self.recurrent.recurrent_layer(X_tilde, W_in)
        W_new = W_out.squeeze(dim=0)
        object.__setattr__(self.recurrent, "weight", W_new)
        self._last_W = W_new.detach()
        return self.recurrent.conv_layer(W_new, x, ei, ew)

    def forward(self, x, ei, ew):
        h = x if self.mode == "static" else self._core(x, ei, ew)
        h = F.relu(self.c1(h, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h)


def focal(logits, target, alpha, gamma=2.0):
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return (((1 - logpt.exp()) ** gamma) * (-logpt) * alpha.to(logits.device)[target]).mean()


def evaluate(model, snaps, measure_W=False):
    model.eval()
    probs, labels, wjumps = [], [], []
    prevW = None
    with torch.no_grad():
        model.reset_hidden_state()
        for s in snaps:
            lg = model(s.x, s.edge_index, s.edge_attr)
            probs.append(torch.softmax(lg, 1)[:, 1].numpy()); labels.append(s.y.numpy().astype(int))
            if measure_W and model._last_W is not None:
                W = model._last_W
                if prevW is not None:
                    wjumps.append((torch.norm(W - prevW) / (torch.norm(W) + 1e-9)).item())
                prevW = W
    model.train()
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    roc = roc_auc_score(labels, probs); pr = average_precision_score(labels, probs)
    best = max(((matthews_corrcoef(labels, (probs > t).astype(int)),
                 f1_score(labels, (probs > t).astype(int), average="macro", zero_division=0))
                for t in np.arange(0.05, 1.0, 0.025) if (probs > t).any()), default=(0, 0))
    wj = float(np.mean(wjumps)) if wjumps else float("nan")
    return roc, pr, best[0], best[1], wj


def main():
    log = open(f"{TAG}.log", "w")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    out(f"=== DIAGNOSE {TAG} (raw5, focal, in-order W={WINDOW}) epochs={EPOCHS} ===")
    t0 = time.time()
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    ds = loader.get_dataset(lags=1)
    tr, te = temporal_signal_split(ds, train_ratio=0.7)
    tr, te = list(tr), list(te)
    n_nodes = len(loader._dataset["node_labels"])

    torch.manual_seed(0)
    model = DiagGCN(n_nodes, in_f=loader.n_node_features, mode=MODE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(tr), WINDOW):
            win = tr[start:start + WINDOW]
            model.reset_hidden_state(); opt.zero_grad()
            wl = 0.0
            for s in win:
                lg = model(s.x, s.edge_index, s.edge_attr)
                wl = wl + focal(lg, s.y.long(), base._snapshot_class_weights(s.y))
            (wl / len(win)).backward(); opt.step()
        if epoch % 20 == 0 or epoch == EPOCHS:
            roc, pr, mcc, mac, wj = evaluate(model, te, measure_W=(MODE == "core_gru"))
            wjs = f" Wjump={wj:.3f}" if MODE == "core_gru" else ""
            out(f"[ep {epoch:3d}] ROC={roc:.4f} PR={pr:.4f} bestMCC={mcc:.3f} macroF1={mac:.3f}{wjs}")

    roc, pr, mcc, mac, wj = evaluate(model, te, measure_W=(MODE == "core_gru"))
    wjs = f"  mean_W_jumpiness={wj:.3f} (rel.Frob ‖ΔW‖/‖W‖ per step)" if MODE == "core_gru" else ""
    out(f"\nFINAL[{TAG}] ROC-AUC={roc:.4f} PR-AUC={pr:.4f} bestMCC={mcc:.3f} macroF1={mac:.3f}{wjs} | {(time.time()-t0)/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
