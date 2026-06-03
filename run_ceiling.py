"""No-attack ceiling run for the HONEST pipeline (post-leakage-fix).

Trains the improved EvolveGCN-H detector on the kinematic-feature signal
(X = standardized [x, y, heading, speed, accel], edge_weight = 1/(1+dist),
binary node labels) and reports macro / weighted / micro F1 plus a per-class
breakdown so the malicious-class F1 is visible (the metric that the old
weighted-F1 ~0.99 was hiding).

Faithful to the reference training style: one full-batch gradient step per
epoch (mean CE accumulated over all train snapshots), but with periodic
evaluation so we can see whether the honest task is learnable and where it
plateaus.

Run:
    CUDA_VISIBLE_DEVICES=-1 python run_ceiling.py [num_epochs] [eval_every]
"""

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from graphs.recurrent.graphs_evolvegcn_h_improved import RecurrentGCN1
import graphs.recurrent.graphs_base as base

NUM_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
EVAL_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 25
LR = 0.01

LOG_PATH = "ceiling_run.log"
_logf = open(LOG_PATH, "w")


def log(msg):
    print(msg, flush=True)
    _logf.write(msg + "\n")
    _logf.flush()


def evaluate(model, dataset):
    """Return dict of metrics over the whole dataset (pooled over snapshots)."""
    model.eval()
    all_y, all_pred = [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for snap in dataset:
            y_hat, _ = model(snap.x, snap.edge_index, snap.edge_attr)
            y = snap.y.long()
            cw = base._snapshot_class_weights(snap.y)
            total_loss += float(F.cross_entropy(y_hat, y, weight=cw)) * len(y)
            total += len(y)
            all_pred.append(y_hat.argmax(dim=1).cpu().numpy())
            all_y.append(y.cpu().numpy())
    model.train()
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    out = {"loss": total_loss / max(total, 1)}
    for avg in ("micro", "macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        out[avg] = {"p": p, "r": r, "f": f}
    out["acc"] = accuracy_score(y_true, y_pred)
    out["mcc"] = matthews_corrcoef(y_true, y_pred)
    # Per-class (0=benign, 1=malicious)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    out["per_class"] = {
        "benign":    {"p": p[0], "r": r[0], "f": f[0], "support": int(s[0])},
        "malicious": {"p": p[1], "r": r[1], "f": f[1], "support": int(s[1])},
    }
    out["pred_pos_rate"] = float((y_pred == 1).mean())
    out["true_pos_rate"] = float((y_true == 1).mean())
    return out


def main():
    log(f"=== HONEST CEILING RUN (no attack) | epochs={NUM_EPOCHS} eval_every={EVAL_EVERY} lr={LR} ===")
    t0 = time.time()

    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    dataset = loader.get_dataset(lags=1)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
    n_nodes = len(loader._dataset["node_labels"])
    log(f"n_node_features={loader.n_node_features}  n_nodes={n_nodes}  "
        f"train_snaps={train_dataset.snapshot_count}  test_snaps={test_dataset.snapshot_count}")

    # Global label balance on test
    test_y = np.concatenate([s.y.cpu().numpy() for s in test_dataset])
    log(f"test label balance: benign={int((test_y==0).sum())} "
        f"malicious={int((test_y==1).sum())} "
        f"(malicious rate={float((test_y==1).mean()):.4f})")

    model = RecurrentGCN1(node_features=loader.n_node_features, node_count=n_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        cost = 0.0
        n_snaps = 0
        for snap in train_dataset:
            y_hat, _ = model(snap.x, snap.edge_index, snap.edge_attr)
            cw = base._snapshot_class_weights(snap.y)
            cost = cost + F.cross_entropy(y_hat, snap.y.long(), weight=cw)
            n_snaps += 1
        cost = cost / max(n_snaps, 1)
        cost.backward()
        optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            m = evaluate(model, test_dataset)
            pc = m["per_class"]
            log(
                f"[epoch {epoch:3d}] train_CE={float(cost):.4f} | "
                f"test: acc={m['acc']:.4f} mcc={m['mcc']:.4f} "
                f"F1(macro={m['macro']['f']:.4f} weighted={m['weighted']['f']:.4f} "
                f"micro={m['micro']['f']:.4f}) | "
                f"mal[P={pc['malicious']['p']:.3f} R={pc['malicious']['r']:.3f} "
                f"F={pc['malicious']['f']:.3f}] | "
                f"pred_pos={m['pred_pos_rate']:.3f}"
            )

    # Final detailed report
    m = evaluate(model, test_dataset)
    log("\n=== FINAL (honest, no leakage) ===")
    log(f"  accuracy        : {m['acc']:.4f}")
    log(f"  MCC             : {m['mcc']:.4f}")
    log(f"  F1 macro        : {m['macro']['f']:.4f}   <-- honest headline metric")
    log(f"  F1 weighted     : {m['weighted']['f']:.4f}   (old leaky headline ~0.99)")
    log(f"  F1 micro        : {m['micro']['f']:.4f}")
    for cls in ("benign", "malicious"):
        c = m["per_class"][cls]
        log(f"  {cls:9s}: P={c['p']:.4f} R={c['r']:.4f} F1={c['f']:.4f} (support={c['support']})")
    log(f"\n  total time: {(time.time()-t0)/60:.1f} min")

    torch.save(model, "model/ceiling_honest_model.pt")
    log("  saved: model/ceiling_honest_model.pt")
    _logf.close()


if __name__ == "__main__":
    main()
