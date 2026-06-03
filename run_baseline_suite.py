"""Literature-style baseline suite — the classical algorithms MBD/V2X papers
tabulate, all on OUR honest vehicle-disjoint split (engineered features),
reported with the same honest metrics (macro-F1, MCC, accuracy, ROC-AUC).

Run:  python run_baseline_suite.py [seed]
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             matthews_corrcoef, roc_auc_score)
from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def load_disjoint(seed):
    lb = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)
    _ = lb.get_dataset(lags=1)
    T, lags, N = lb._dataset["time_periods"], lb.lags, len(lb._dataset["node_labels"])
    aug = np.load("data/features_augmented.npy")
    active = aug[..., 0] != 0.0
    flat = aug.reshape(-1, 10)
    aug = ((aug - flat.mean(0)) / (flat.std(0) + 1e-8))
    y = np.stack([lb.targets[i] for i in range(T - lags)])
    node_label = (y.max(0) > 0).astype(int)
    rng = np.random.default_rng(seed); trm = np.zeros(N, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]] = True
    n_tr = int(0.7 * (T - lags))
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in range(T - lags):
        a = active[i]
        if i < n_tr:
            m = a & trm; Xtr.append(aug[i][m]); ytr.append(y[i][m])
        m2 = a & (~trm); Xte.append(aug[i][m2]); yte.append(y[i][m2])
    return (np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xte), np.concatenate(yte))


def score_model(name, clf, Xtr, ytr, Xte, yte, subsample=None):
    t0 = time.time()
    if subsample and len(ytr) > subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ytr), subsample, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    clf.fit(Xtr, ytr)
    if hasattr(clf, "predict_proba"):
        s = clf.predict_proba(Xte)[:, 1]
    else:
        s = clf.decision_function(Xte)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    roc = roc_auc_score(yte, s)
    thr = max(((matthews_corrcoef(yte, (s > t).astype(int)), t)
               for t in np.arange(0.05, 1.0, 0.025) if (s > t).any()),
              default=(0, 0.5))[1]
    pred = (s > thr).astype(int)
    mac = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    acc = accuracy_score(yte, pred); mcc = matthews_corrcoef(yte, pred)
    return dict(name=name, macroP=mac[0]*100, macroR=mac[1]*100, macroF1=mac[2]*100,
                acc=acc*100, mcc=mcc*100, roc=roc*100, sec=time.time()-t0)


def main():
    out = open("baseline_suite.log", "w")
    def w(m):
        print(m, flush=True); out.write(m + "\n"); out.flush()
    w(f"=== Literature baseline suite — vehicle-disjoint seed={SEED}, engineered feats ===")
    Xtr, ytr, Xte, yte = load_disjoint(SEED)
    w(f"train={len(ytr)} (mal {int(ytr.sum())})  test={len(yte)} (mal {int(yte.sum())})\n")

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=300, class_weight="balanced"), None),
        ("NaiveBayes",         GaussianNB(), None),
        ("KNN(k=15)",          KNeighborsClassifier(n_neighbors=15), 30000),
        ("DecisionTree",       DecisionTreeClassifier(class_weight="balanced", max_depth=12), None),
        ("RandomForest",       RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=4), None),
        ("ExtraTrees",         ExtraTreesClassifier(n_estimators=200, class_weight="balanced", n_jobs=4), None),
        ("AdaBoost",           AdaBoostClassifier(n_estimators=200), None),
        ("HistGradBoost",      HistGradientBoostingClassifier(max_iter=300), None),
        ("LinearSVM(SGD)",     SGDClassifier(loss="hinge", class_weight="balanced", max_iter=50), None),
        ("MLP",                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=120), 80000),
    ]
    rows = []
    for name, clf, sub in models:
        try:
            r = score_model(name, clf, Xtr, ytr, Xte, yte, subsample=sub)
            rows.append(r)
            w(f"{name:>20} | macroF1={r['macroF1']:6.2f}  MCC={r['mcc']:6.2f}  "
              f"Acc={r['acc']:6.2f}  ROC={r['roc']:6.2f}  ({r['sec']:.1f}s)")
        except Exception as e:
            w(f"{name:>20} | FAILED: {e}")

    rows.sort(key=lambda r: -r["mcc"])
    w("\n--- ranked by MCC ---")
    w(f"{'model':>20} {'macroF1':>8} {'MCC':>7} {'Acc':>7} {'ROC':>7}")
    for r in rows:
        w(f"{r['name']:>20} {r['macroF1']:8.2f} {r['mcc']:7.2f} {r['acc']:7.2f} {r['roc']:7.2f}")
    w("\nFor reference (our GNNs, same split): GConvGRU MCC~84.8/ROC~99.4 (s3), "
      "static-GCN-eng MCC~82.8, TGCN MCC~82.8; EvolveGCN-H MCC~44.")
    out.close()


if __name__ == "__main__":
    main()
