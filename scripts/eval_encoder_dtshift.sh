#!/usr/bin/env bash
# Encoder stress test: evaluate grid_v5 checkpoints on dt-shifted regular
# schedules (train dt=1.0 -> eval dt in {0.25, 2.0}) where identical
# positions correspond to different elapsed times, making timestamps
# information-theoretically relevant. Resolved eval generator (s_eval=8).
#   bash encoder_dtshift.sh <cuda:N> <cells...>
set -u
R=$HOME/repos/continuous-time-causal-pfn
PY=$HOME/miniconda3/envs/dotime-fullscale/bin/python
OUT=$R/results/encoder_dtshift
mkdir -p "$OUT"
cd "$R" || exit 1; export PYTHONPATH=$R
DEV="$1"; shift
for cell in "$@"; do
  case "$cell" in *_OU) mech=linear;; *) mech=neural;; esac
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    ckpt="$R/results/grid_v5/$cell/seed_$seed/continuous_do_over_time_pfn_best.pt"
    for dt in 0.25 2.0; do
      out="$OUT/${cell}_seed${seed}_dt${dt}.json"
      [ -f "$out" ] && continue
      $PY scripts/ct_synth_eval.py --checkpoint "$ckpt" \
          --mechanism-kind "$mech" --schedule regular --dt "$dt" \
          --substeps 8 --intervention-value-scale 1.0 --vectorize \
          --n-eval-batches 50 --batch-size 24 --device "$DEV" \
          --save-json "$out" >>"$OUT/log_${cell}.log" 2>&1 \
        && echo "[$DEV] $cell s$seed dt$dt" >>"$OUT/progress.log"
    done
  done
done
echo "[$DEV] dtshift worker done $(date -Is)" >>"$OUT/progress.log"
