#!/usr/bin/env bash
# Phase-1 reproducible ablation: 5 seeds x 4 headline cells.
# Cooperative: each worker claims (cell,seed) jobs atomically via mkdir, so
# workers on any free GPU cooperate -- add more anytime, no fixed sharding.
#   bash grid_v5.sh <cuda:N>
set -u
R=$HOME/repos/continuous-time-causal-pfn
PY=$HOME/miniconda3/envs/dotime-fullscale/bin/python
CFG=$R/configs/continuous_ablation_grid.yaml
CKPT_ROOT=$R/results/grid_v5
EVAL_ROOT=$R/results/grid_v5_eval
CLAIM=$R/results/grid_v5_claims
mkdir -p "$CKPT_ROOT" "$EVAL_ROOT" "$CLAIM"
cd "$R" || exit 1; export PYTHONPATH=$R
DEV="$1"
CELLS=("time_naive_OU linear 1" "time_fine_OU linear 8"
       "time_naive_neural neural 1" "time_fine_neural neural 8")
for spec in "${CELLS[@]}"; do
  set -- $spec; cell=$1; mech=$2; sub=$3
  for seed in 0 1 2 3 4; do
    # atomic claim; skip if another worker took it
    mkdir "$CLAIM/${cell}_s${seed}" 2>/dev/null || continue
    sdir="$CKPT_ROOT/$cell/seed_$seed"
    ckpt="$sdir/continuous_do_over_time_pfn_best.pt"
    if [ ! -f "$ckpt" ]; then
      $PY scripts/ct_train.py --config "$CFG" \
          --mechanism-kind "$mech" --num-substeps "$sub" --vectorize \
          --seed "$seed" --total-steps 10000 --save-dir "$sdir" --device "$DEV" \
          >>"$EVAL_ROOT/train_${cell}_s${seed}.log" 2>&1 \
        && echo "[$DEV] trained $cell s$seed $(date +%H:%M)" >>"$EVAL_ROOT/progress.log" \
        || { echo "[$DEV] TRAIN FAIL $cell s$seed" >>"$EVAL_ROOT/progress.log"; rmdir "$CLAIM/${cell}_s${seed}"; continue; }
    fi
    for ecfg in "regular 1 reg_s1" "mixed 1 mixed_s1" "mixed 8 mixed_s8"; do
      set -- $ecfg; esched=$1; esub=$2; etag=$3
      out="$EVAL_ROOT/${cell}_seed${seed}_${etag}.json"
      [ -f "$out" ] && continue
      $PY scripts/ct_synth_eval.py --checkpoint "$ckpt" \
          --mechanism-kind "$mech" --schedule "$esched" --dt 1.0 --jitter 0.3 \
          --substeps "$esub" --intervention-value-scale 1.0 \
          --n-eval-batches 50 --batch-size 24 --device "$DEV" --save-json "$out" \
        >>"$EVAL_ROOT/eval_${cell}_s${seed}.log" 2>&1 \
        && echo "[$DEV]   eval $cell s$seed $etag" >>"$EVAL_ROOT/progress.log"
    done
  done
done
echo "[$DEV] worker done $(date -Is)" >>"$EVAL_ROOT/progress.log"
