#!/usr/bin/env bash
# Run the full BuST GNN experiment matrix on .100 (CPU, parallelism-capped, memory-light).
cd ~/bust
PY=~/miniconda3/envs/bust/bin/python
mkdir -p logs
JOBS=$(mktemp)
for s in 0 1 2 3 4; do
  echo "$PY expA_gconvgru_eng.py gconvgru eng10     $s 40 24 > logs/expA_eng10_s$s.log 2>&1" >> "$JOBS"
  echo "$PY expA_gconvgru_eng.py gconvgru eng8nopos $s 40 24 > logs/expA_eng8_s$s.log  2>&1" >> "$JOBS"
  echo "$PY expB_hybrid.py        eng10             $s 40 24 > logs/expB_eng10_s$s.log 2>&1" >> "$JOBS"
  for m in rf_eng rf_rel gnn_rel gnn_edge; do
    echo "$PY expC_relational.py $m $s 40 24 > logs/expC_${m}_s$s.log 2>&1" >> "$JOBS"
  done
done
for s in 0 3; do
  echo "$PY expA_gconvgru_eng.py gconvgru raw5 $s 40 24 > logs/expA_raw5_s$s.log 2>&1" >> "$JOBS"
done
N=$(wc -l < "$JOBS")
echo "[run_all] $N jobs, P=10, start $(date)" | tee logs/run_all.status
cat "$JOBS" | xargs -P 10 -I CMD bash -c CMD
echo "[run_all] DONE $(date)" | tee -a logs/run_all.status
{ echo "=== RESULTS $(date) ==="; grep -hE "RESULT expC|RESULT hybrid|FINAL\[" logs/*.log | sort; } > ~/bust/RESULTS.txt
echo "[run_all] collected -> ~/bust/RESULTS.txt"
