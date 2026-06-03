"""Decision-threshold sweep and ROC/PR separability diagnostics."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import graphs.recurrent.graphs_evolvegcn_h_improved  # noqa: needed for torch.load to resolve class


def main():
    loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    dataset = loader.get_dataset(lags=1)
    _, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)

    model = torch.load("model/ceiling_honest_model.pt", weights_only=False)
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for snap in test_dataset:
            y_hat, _ = model(snap.x, snap.edge_index, snap.edge_attr)
            p_mal = torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy()
            probs.append(p_mal)
            labels.append(snap.y.cpu().numpy().astype(int))
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    base_rate = labels.mean()
    print(f"test nodes={len(labels)}  malicious base rate={base_rate:.4f}")
    print(f"prob[mal] range: [{probs.min():.4f}, {probs.max():.4f}]  "
          f"mean(benign)={probs[labels==0].mean():.4f}  mean(mal)={probs[labels==1].mean():.4f}")

    # Threshold-INDEPENDENT separability
    roc = roc_auc_score(labels, probs)
    pr  = average_precision_score(labels, probs)
    print(f"\n=== Separability (threshold-independent) ===")
    print(f"  ROC-AUC : {roc:.4f}   (0.5=random, 1.0=perfect)")
    print(f"  PR-AUC  : {pr:.4f}   (baseline=base rate={base_rate:.4f}; "
          f"lift x{pr/base_rate:.1f})")

    print(f"\n=== Threshold sweep ===")
    print(f"  {'thr':>5} {'pred_pos':>8} {'P(mal)':>7} {'R(mal)':>7} "
          f"{'F1(mal)':>8} {'macroF1':>8} {'MCC':>7}")
    best = {"mcc": -2, "thr": None}
    best_f1 = {"f": -1, "thr": None}
    for thr in np.arange(0.05, 1.00, 0.05):
        pred = (probs > thr).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(labels, pred)
        p   = precision_score(labels, pred, pos_label=1, zero_division=0)
        r   = recall_score(labels, pred, pos_label=1, zero_division=0)
        f1m = f1_score(labels, pred, pos_label=1, zero_division=0)
        macro = f1_score(labels, pred, average="macro", zero_division=0)
        flag = ""
        if mcc > best["mcc"]:
            best = {"mcc": mcc, "thr": thr}; flag += " *MCC"
        if f1m > best_f1["f"]:
            best_f1 = {"f": f1m, "thr": thr}; flag += " *F1mal"
        print(f"  {thr:5.2f} {pred.mean():8.4f} {p:7.3f} {r:7.3f} "
              f"{f1m:8.3f} {macro:8.3f} {mcc:7.3f}{flag}")

    print(f"\nBest MCC = {best['mcc']:.3f} at threshold {best['thr']:.2f}")
    print(f"Best malicious-F1 = {best_f1['f']:.3f} at threshold {best_f1['thr']:.2f}")
    print("\nInterpretation:")
    if roc < 0.65:
        print("  -> ROC-AUC low: features barely separate classes. Thresholding")
        print("     won't help much; need better features/model/temporal signal.")
    elif best["mcc"] > 0.18 + 0.10:
        print("  -> Thresholding RECOVERS significant MCC: the 0.19 was largely a")
        print("     decision-cutoff artifact from heavy class weighting.")
    else:
        print("  -> Moderate ROC-AUC but thresholding gives limited MCC gain:")
        print("     signal exists but is weak; temporal fix / features are the lever.")


if __name__ == "__main__":
    main()
