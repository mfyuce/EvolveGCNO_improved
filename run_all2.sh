#!/usr/bin/env bash
# Seeds 5-9 for the 6 core configs (confirmation run, paired with seeds 0-4).
cd ~/bust
PY=~/miniconda3/envs/bust/bin/python
mkdir -p logs
JOBS=$(mktemp)
for s in 5 6 7 8 9; do
  echo "$PY expA_gconvgru_eng.py gconvgru eng10 $s 40 24 > logs/expA_eng10_s$s.log 2>&1" >> "$JOBS"
  echo "$PY expB_hybrid.py        eng10         $s 40 24 > logs/expB_eng10_s$s.log 2>&1" >> "$JOBS"
  for m in rf_eng rf_rel gnn_rel gnn_edge; do
    echo "$PY expC_relational.py $m $s 40 24 > logs/expC_${m}_s$s.log 2>&1" >> "$JOBS"
  done
done
N=$(wc -l < "$JOBS")
echo "[run_all2] $N jobs, P=10, start $(date)" | tee logs/run_all2.status
cat "$JOBS" | xargs -P 10 -I CMD bash -c CMD
echo "[run_all2] DONE $(date)" | tee -a logs/run_all2.status
echo "[run_all2] collected"
