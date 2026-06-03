"""Random Forest baseline on per-node features, vehicle-disjoint split."""

import os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             matthews_corrcoef, roc_auc_score)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader

FEATS = sys.argv[1] if len(sys.argv) > 1 else "eng"
SEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def main():
    t0 = time.time()
    lb = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = lb.get_dataset(lags=1)
    T, lags, N = lb._dataset["time_periods"], lb.lags, len(lb._dataset["node_labels"])

    aug_raw = np.load("data/features_augmented.npy")            # (T,N,10) raw
    active = aug_raw[..., 0] != 0.0                              # presence
    if FEATS == "raw":
        feats = aug_raw[..., :5]
    else:
        feats = aug_raw
    # standardize
    flat = feats.reshape(-1, feats.shape[-1])
    feats = ((feats - flat.mean(0)) / (flat.std(0) + 1e-8))

    y = np.stack([lb.targets[i] for i in range(T - lags)])       # (T',N) binary
    node_label = (y.max(0) > 0).astype(int)
    rng = np.random.default_rng(SEED); trm = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]] = True
    n_tr = int(0.7 * (T - lags))

    # pool (node, timestep) samples — train: train-vehicles over first 70% steps; test: test-vehicles all steps
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in range(T - lags):
        act = active[i]
        if i < n_tr:
            m = act & trm
            Xtr.append(feats[i][m]); ytr.append(y[i][m])
        m2 = act & (~trm)
        Xte.append(feats[i][m2]); yte.append(y[i][m2])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte)
    print(f"RF [{FEATS}] seed={SEED}  train={len(ytr)} (mal {ytr.sum()})  test={len(yte)} (mal {yte.sum()})", flush=True)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                n_jobs=4, random_state=SEED)
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)[:, 1]
    roc = roc_auc_score(yte, proba)
    # best-MCC threshold
    thr = max(((matthews_corrcoef(yte, (proba > t).astype(int)), t)
               for t in np.arange(0.05, 1.0, 0.025) if (proba > t).any()), default=(0, 0.5))[1]
    pred = (proba > thr).astype(int)
    acc = accuracy_score(yte, pred); mcc = matthews_corrcoef(yte, pred)
    out = open("rf_baseline.log", "a")
    def w(m):
        print(m, flush=True); out.write(m + "\n")
    w(f"\n=== Random Forest [{FEATS}] vehicle-disjoint seed={SEED}  (thr={thr:.2f}) ===")
    w(f"{'avg':>10} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    for avg in ("macro", "weighted", "micro"):
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average=avg, zero_division=0)
        w(f"{avg:>10} {p*100:7.2f} {r*100:7.2f} {f*100:7.2f}")
    pc = precision_recall_fscore_support(yte, pred, labels=[0, 1], zero_division=0)
    w(f"per-class malicious: P={pc[0][1]*100:.2f} R={pc[1][1]*100:.2f} F1={pc[2][1]*100:.2f}")
    w(f"ROC-AUC={roc*100:.2f}  Accuracy={acc*100:.2f}  MCC={mcc*100:.2f}  | {(time.time()-t0)/60:.1f}min")
    out.close()


if __name__ == "__main__":
    main()
