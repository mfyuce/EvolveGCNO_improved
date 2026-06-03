"""Build augmented node features that target misbehavior SEMANTICS.

The raw 5 features [x, y, heading, speed, accel] are instantaneous absolute
values. Most attacks are CONSISTENCY violations across time:
  - Constant Random Position : claimed speed > 0 but position frozen
                               (||Δpos|| ~ 0 while speed high)
  - Position offset          : implausible position jumps
  - Speed offset             : claimed speed vs implied (from Δpos)
  - Reversed heading         : claimed heading vs motion direction (~180 off)
  - Accel inconsistency      : claimed accel vs Δspeed/Δt

We compute per-vehicle temporal deltas (dt=1, labels constant per vehicle) and
emit a (T, N, K) array aligned to the SAME node ordering / timestep layout as
the JSON, so it can drop into the existing edge/label structure.

Output: data/features_augmented.npy  (T, N, 10)  +  data/features_augmented_cols.txt
"""

import numpy as np
import pandas as pd

CSV = "data/myoutput.csv"
df = pd.read_csv(CSV)

unique_ids = np.unique(df["id"]).tolist()
node_to_index = {v: i for i, v in enumerate(unique_ids)}
N = len(unique_ids)
T = int(df["timestep"].max()) + 1  # 1000

COLS = ["x", "y", "heading", "speed", "accel",
        "pos_jump", "dspeed", "accel_resid", "abs_head_change", "motion_head_resid"]
K = len(COLS)

feats = np.zeros((T, N, K), dtype=np.float32)


def ang_diff_deg(a, b):
    """Smallest absolute circular difference in degrees, range [0,180]."""
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


for vid, g in df.groupby("id"):
    g = g.sort_values("timestep")
    idx = node_to_index[vid]
    ts   = g["timestep"].to_numpy().astype(int)
    x    = g["x"].to_numpy(); y = g["y"].to_numpy()
    head = g["heading"].to_numpy()
    spd  = g["speed"].to_numpy()
    acc  = np.nan_to_num(g["acceleration"].to_numpy(), nan=0.0)

    dx = np.diff(x, prepend=x[0]); dy = np.diff(y, prepend=y[0])
    pos_jump = np.sqrt(dx * dx + dy * dy)                      # degrees; ~0 = frozen
    dspeed = np.diff(spd, prepend=spd[0])                      # implied accel * dt
    accel_resid = acc - dspeed                                 # claimed vs implied (dt=1)
    abs_head_change = np.empty_like(head)
    abs_head_change[0] = 0.0
    abs_head_change[1:] = ang_diff_deg(head[1:], head[:-1])
    motion_head = np.degrees(np.arctan2(dy, dx)) % 360.0       # direction of travel
    motion_head_resid = ang_diff_deg(head, motion_head)
    # when frozen (no motion) the motion direction is undefined -> 0 residual
    motion_head_resid[pos_jump < 1e-9] = 0.0

    block = np.stack([x, y, head, spd, acc,
                      pos_jump, dspeed, accel_resid,
                      abs_head_change, motion_head_resid], axis=1).astype(np.float32)
    feats[ts, idx, :] = block

np.save("data/features_augmented.npy", feats)
with open("data/features_augmented_cols.txt", "w") as f:
    f.write("\n".join(COLS) + "\n")

print(f"saved data/features_augmented.npy  shape={feats.shape}")
# quick sanity: for malicious vs benign, mean of the engineered cols
y = np.zeros((T, N), dtype=int)
for vid, g in df.groupby("id"):
    idx = node_to_index[vid]
    g = g.sort_values("timestep")
    y[g["timestep"].to_numpy().astype(int), idx] = g["label"].to_numpy().astype(int)
active = feats.any(axis=2)              # nodes present at (t)
mal = (y == 1) & active                  # use binary for the quick check
ben = (y == 0) & active
print(f"{'col':>18} {'benign_mean':>12} {'malic_mean':>12}")
for k, c in enumerate(COLS):
    print(f"{c:>18} {feats[...,k][ben].mean():12.4f} {feats[...,k][mal].mean():12.4f}")
