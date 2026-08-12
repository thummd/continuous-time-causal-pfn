#!/usr/bin/env bash
# Phase-2: extend the reproducible ablation to 10 seeds x all 8 cells.
# Cooperative (atomic mkdir claims); resumable; skips existing artifacts.
#   bash grid_v5b.sh <cuda:N>
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
CELLS=(
  "time_naive_OU linear 1 -"        "time_fine_OU linear 8 -"
  "time_naive_neural neural 1 -"    "time_fine_neural neural 8 -"
  "pos_naive_OU linear 1 --positional-only"   "pos_fine_OU linear 8 --positional-only"
  "pos_naive_neural neural 1 --positional-only" "pos_fine_neural neural 8 --positional-only"
)
for seed in 0 1 2 3 4 5 6 7 8 9; do   # seed-major: spread cells evenly over time
 for spec in "${CELLS[@]}"; do
  set -- $spec; cell=$1; mech=$2; sub=$3; posflag=$4
  [ "$posflag" = "-" ] && posargs=() || posargs=("$posflag")
  mkdir "$CLAIM/${cell}_s${seed}" 2>/dev/null || continue
  sdir="$CKPT_ROOT/$cell/seed_$seed"
  ckpt="$sdir/continuous_do_over_time_pfn_best.pt"
  if [ ! -f "$ckpt" ]; then
    $PY scripts/ct_train.py --config "$CFG" \
        --mechanism-kind "$mech" --num-substeps "$sub" --vectorize \
        ${posargs[@]+"${posargs[@]}"} \
        --seed "$seed" --total-steps 10000 --save-dir "$sdir" --device "$DEV" \
        >>"$EVAL_ROOT/train_${cell}_s${seed}.log" 2>&1 \
      && echo "[$DEV] trained $cell s$seed $(date +%m%d-%H:%M)" >>"$EVAL_ROOT/progress.log" \
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
echo "[$DEV] v5b worker done $(date -Is)" >>"$EVAL_ROOT/progress.log"
