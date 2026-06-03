"""Generate honest figures for the paper revisions (PDF, vector).
A: protocol drives the numbers (3 protocols x RF/GConvGRU)
B: vehicle-disjoint model comparison (MCC, mean+/-std)
C: EvolveGCN recurrence self-neutralizes (W-jumpiness decay)
D: ROC curves on vehicle-disjoint (from roc_*.npz, if present)
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spilframe" if False else "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})

# ---------- A: protocol drives the numbers ----------
prot = ["Random\n(competitor)", "Temporal\n(original)", "Vehicle-disjoint\n(ours)"]
rf   = [94.28, 97.33, 74.7]
gg   = [94.98, 75.42, 74.9]
x = np.arange(len(prot)); w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 3.8))
b1 = ax.bar(x - w/2, rf, w, label="Random Forest", color="#b0b0b0", edgecolor="k")
b2 = ax.bar(x + w/2, gg, w, label="GConvGRU (ST-GNN)", color="#2c7fb8", edgecolor="k")
ax.axhline(97.35, ls="--", c="crimson", lw=1)
ax.text(2.4, 97.6, "published 97.35", color="crimson", fontsize=8, ha="right")
ax.set_ylabel("MCC (%)"); ax.set_xticks(x); ax.set_xticklabels(prot)
ax.set_ylim(0, 105); ax.legend(loc="lower left", fontsize=9)
ax.set_title("The evaluation protocol, not the model, drives the score")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}",
            ha="center", fontsize=8)
fig.tight_layout(); fig.savefig("fig_protocols.pdf"); plt.close(fig)
print("fig_protocols.pdf")

# ---------- B: vehicle-disjoint model comparison ----------
models = ["Static GCN\n(+feats)", "GConvGRU", "RF", "TGCN", "Static GCN\n(raw)",
          "EvolveGCN-H\n(improved)", "EvolveGCN-H\n(std)", "EvolveGCN-O\n(std)"]
mcc  = [76.8, 74.9, 74.7, 71.3, 55.8, 44.0, 39.8, 35.9]
err  = [2.8, 5.9, 5.8, 8.4, 10.7, 0, 12.6, 5.1]
istemp = [0, 1, 0, 1, 0, 1, 1, 1]
colors = ["#2c7fb8" if t else "#b0b0b0" for t in istemp]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
xb = np.arange(len(models))
ax.bar(xb, mcc, yerr=err, capsize=3, color=colors, edgecolor="k")
ax.set_ylabel("MCC (%)  —  vehicle-disjoint"); ax.set_xticks(xb)
ax.set_xticklabels(models, fontsize=8)
ax.set_title("Generalization to unseen attackers (4 seeds)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2c7fb8", label="temporal (ST-GNN)"),
                   Patch(color="#b0b0b0", label="non-temporal")], fontsize=9)
fig.tight_layout(); fig.savefig("fig_models.pdf"); plt.close(fig)
print("fig_models.pdf")

# ---------- C: EvolveGCN recurrence self-neutralizes ----------
ep = [20, 40, 60]; wj = [0.231, 0.023, 0.002]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(ep, wj, "o-", color="#d95f0e", lw=2, ms=8)
ax.set_yscale("log"); ax.set_xlabel("training epoch")
ax.set_ylabel(r"GRU weight change  $\|\Delta W\|/\|W\|$")
ax.set_title("EvolveGCN recurrence self-neutralizes\n(drives toward a constant weight)")
for e, v in zip(ep, wj):
    ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(6, 6), fontsize=9)
fig.tight_layout(); fig.savefig("fig_wjumpiness.pdf"); plt.close(fig)
print("fig_wjumpiness.pdf")

# ---------- D: ROC curves (if data present) ----------
from sklearn.metrics import roc_curve, roc_auc_score
roc_models = [("gconvgru", "GConvGRU", "#2c7fb8", "-"),
              ("rf", "Random Forest", "#000000", "--"),
              ("static", "Static GCN", "#7fbf7b", "-."),
              ("evolveo", "EvolveGCN-O", "#d95f0e", ":")]
have = [m for m in roc_models if os.path.exists(f"roc_{m[0]}.npz")]
if have:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for key, name, col, ls in have:
        d = np.load(f"roc_{key}.npz"); fpr, tpr, _ = roc_curve(d["label"], d["prob"])
        auc = roc_auc_score(d["label"], d["prob"])
        ax.plot(fpr, tpr, col, ls=ls, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — vehicle-disjoint (unseen attackers)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig("fig_roc.pdf"); plt.close(fig)
    print(f"fig_roc.pdf  ({len(have)} models)")
else:
    print("fig_roc.pdf SKIPPED (no roc_*.npz yet)")
