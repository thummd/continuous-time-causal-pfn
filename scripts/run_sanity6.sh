#!/usr/bin/env bash
# Sanity6: 4-GPU PARALLEL fast validation of intervention-signal fixes.
#
# Quick-validation config (drastically faster than original s6 plan):
#   - 1000 steps (down from 5000) — early stopping will likely cut earlier
#   - n_queries=10 (down from 100) — 10x cheaper encoder/mixer/loss
#   - sim_device=cpu (smoke-tested 25x faster than GPU sim at B=16)
#   - All 4 runs in parallel on GPUs 0, 1, 2, 3
#
# Fixes vs sanity5:
#   2a) Normalize intervention_value by observed std
#   2b) Gated residual in CrossVariableMixer
#   2c) Canonical column order [A, covariates, Y]
#   2d) Obs-only trains on Y_obs; causal trains on Y_int
#
# Usage:
#   export WANDB_API_KEY=...   (or `wandb login` once)
#   bash scripts/run_sanity6.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STEPS=1000
BATCH=16
EARLY_STOP=5
N_QUERIES=10

for cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$cb/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$cb"
        break
    fi
done
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dotime-fullscale 2>/dev/null || conda activate dotime-rocm

CTP_DIR="$(cd "$REPO_DIR/../ctp" 2>/dev/null && pwd || echo "")"
if [ -n "$CTP_DIR" ]; then
    export PYTHONPATH="${CTP_DIR}:${PYTHONPATH:-}"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set. Wandb logging may fail."
fi

echo "============================================================"
echo "Sanity6 (parallel quick-validation)"
echo "  steps=$STEPS, n_queries=$N_QUERIES, batch=$BATCH, sim=cpu"
echo "  4 runs in parallel on GPUs 0,1,2,3"
echo "============================================================"

# Common args (sim_device=cpu — GPU sim is 25x slower at B=16)
COMMON_ARGS=(
    "--config" "$REPO_DIR/configs/server.yaml"
    "--sim-device" "cpu"
    "--total-steps" "$STEPS"
    "--batch-size" "$BATCH"
    "--head-type" "quantile"
    "--target-key" "Y_true"
    "--n-queries" "$N_QUERIES"
    "--query-mode" "all_pairs"
    "--no-tscm-lag"
    "--causal-mask" "interpolation"
    "--early-stop-patience" "$EARLY_STOP"
    "--wandb-project" "do-over-time-pfn"
    "--wandb-entity" "dot-pfn"
)

run_one() {
    local tag="$1"
    local structure="$2"
    local device="$3"
    local extra="$4"
    local save="$REPO_DIR/checkpoints/s6_${tag}"
    local logdir="$REPO_DIR/results/s6_${tag}"
    mkdir -p "$save" "$logdir"
    echo "Launching s6_${tag} on $device (structure=$structure)"
    python "$REPO_DIR/scripts/train.py" "${COMMON_ARGS[@]}" \
        --device "$device" \
        --tscm-structure "$structure" \
        --save-dir "$save" \
        --wandb-run-name "s6_${tag}" \
        $extra \
        > "$logdir/train.log" 2>&1 &
    echo "  PID=$!"
}

run_one "bd_nolag_interp_causal"   "back_door"  "cuda:0" ""
run_one "bd_nolag_interp_obs"      "back_door"  "cuda:1" "--observational-only --obs-only-target Y_obs"
run_one "fd_nolag_interp_causal"   "front_door" "cuda:2" ""
run_one "fd_nolag_interp_obs"      "front_door" "cuda:3" "--observational-only --obs-only-target Y_obs"

echo ""
echo "All 4 s6 runs launched in parallel. Waiting for them to finish..."
wait
echo ""
echo "============================================================"
echo "Sanity6 complete. Checkpoints in checkpoints/s6_*"
echo "Wandb runs: https://wandb.ai/dot-pfn/do-over-time-pfn"
echo "============================================================"
