"""Paired GNN-vs-RF analysis over all seeds (same seed => same vehicle split => paired).
Reads a text file of FINAL[...]/RESULT lines (seeds 0-9). Prints 10-seed mean±SEM per config and
paired comparisons (mean diff, bootstrap 95% CI, sign count, Wilcoxon signed-rank p)."""
import re, sys
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

text = open(sys.argv[1]).read()
data = {}  # config -> {seed: mcc}
for m in re.finditer(r'FINAL\[expA_gconvgru_(\w+)_s(\d+)\][^\n]*?MCC=([\d.]+)', text):
    data.setdefault(f'gnn_{m.group(1)}', {})[int(m.group(2))] = float(m.group(3)) * 100
for m in re.finditer(r'RESULT expC mode=(\w+) seed=(\d+) mcc=([\d.]+)', text):
    data.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
for m in re.finditer(r'RESULT hybrid featset=\w+ seed=(\d+)[^\n]*?rf_mcc=([\d.]+)', text):
    data.setdefault('hybrid_rf', {})[int(m.group(1))] = float(m.group(2))
for m in re.finditer(r'RESULT hybrid featset=\w+ seed=(\d+)[^\n]*?hgb_mcc=([\d.]+)', text):
    data.setdefault('hybrid_hgb', {})[int(m.group(1))] = float(m.group(2))


def arr(cfg):
    d = data.get(cfg, {}); seeds = sorted(d)
    return seeds, np.array([d[s] for s in seeds], float)


print("=== per-config MCC (mean±std, SEM), all seeds ===")
for cfg in ['gnn_eng10', 'gnn_rel', 'gnn_edge', 'hybrid_rf', 'hybrid_hgb',
            'static_rel', 'static_eng', 'rf_rel', 'rf_eng', 'gnn_eng8nopos']:
    seeds, v = arr(cfg)
    if len(v) == 0:
        continue
    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
    print(f"{cfg:15s} {v.mean():5.1f} ± {v.std():4.1f}   SEM {sem:4.2f}   n={len(v)}   seeds={seeds}")


def paired(a, b):
    sa, va = arr(a); sb, vb = arr(b)
    da = dict(zip(sa, va)); db = dict(zip(sb, vb))
    common = sorted(set(sa) & set(sb))
    diff = np.array([da[s] - db[s] for s in common])
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(diff).pvalue if (wilcoxon and len(diff) > 0 and np.any(diff != 0)) else float('nan')
    print(f"  {a:10s} - {b:8s}: Δ={diff.mean():+5.1f}  95%CI[{lo:+5.1f},{hi:+5.1f}]  "
          f"{(diff > 0).sum()}/{len(diff)} seeds>0  Wilcoxon p={p:.3f}")


print("\n=== paired: GNN vs RF (same seed = same split) ===")
paired('gnn_eng10', 'rf_eng')
paired('gnn_rel', 'rf_rel')
paired('gnn_edge', 'rf_rel')
paired('gnn_rel', 'rf_eng')
paired('rf_rel', 'rf_eng')

print("\n=== paired: RECURRENCE isolation (GConvGRU vs static-GCN, same feats) ===")
paired('gnn_rel', 'static_rel')
paired('gnn_eng10', 'static_eng')
print("\n=== paired: is static-GCN just RF? (architecture-without-recurrence vs RF) ===")
paired('static_rel', 'rf_rel')
paired('static_eng', 'rf_eng')
