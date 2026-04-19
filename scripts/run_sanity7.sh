#!/usr/bin/env bash
# Sanity7: target positive R² on Y_int by closing 4 remaining issues from s6.
#
# Fixes on top of s6:
#   1) --intervention-source positivity_aware: clip N(0,4) to [obs_mu-3σ, obs_mu+3σ]
#      during training. Matches the new eval clipping.
#   3) --query-offset-range 0 5: sample query_offset per-query in {0..5} so the
#      mediator chain has time to reflect the intervention (addresses FD gap).
#   4) configs/server_s7.yaml: n_mixer_layers=3 (up from 1) for multi-hop reasoning.
#
#   + Budget: 5000 steps per run (5x s6); num-workers=4, prefetch=4 for overlap.
#   + 4-GPU parallel on cuda:0/1/2/3.
#
# Usage:
#   export WANDB_API_KEY=...   (or `wandb login` once)
#   bash scripts/run_sanity7.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STEPS=5000
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
echo "Sanity7 (parallel; positivity_aware + query_offset + mixer=3)"
echo "  steps=$STEPS, n_queries=$N_QUERIES, batch=$BATCH"
echo "  4 runs in parallel on GPUs 0,1,2,3"
echo "============================================================"

COMMON_ARGS=(
    "--config" "$REPO_DIR/configs/server_s7.yaml"
    "--sim-device" "cpu"
    "--total-steps" "$STEPS"
    "--batch-size" "$BATCH"
    "--head-type" "quantile"
    "--target-key" "Y_true"
    "--n-queries" "$N_QUERIES"
    "--query-mode" "all_pairs"
    "--no-tscm-lag"
    "--causal-mask" "interpolation"
    "--intervention-source" "positivity_aware"
    "--query-offset-range" "0" "5"
    "--early-stop-patience" "$EARLY_STOP"
    "--num-workers" "4"
    "--prefetch" "4"
    "--wandb-project" "do-over-time-pfn"
    "--wandb-entity" "dot-pfn"
)

run_one() {
    local tag="$1"
    local structure="$2"
    local device="$3"
    local extra="$4"
    local save="$REPO_DIR/checkpoints/s7_${tag}"
    local logdir="$REPO_DIR/results/s7_${tag}"
    mkdir -p "$save" "$logdir"
    echo "Launching s7_${tag} on $device (structure=$structure)"
    python "$REPO_DIR/scripts/train.py" "${COMMON_ARGS[@]}" \
        --device "$device" \
        --tscm-structure "$structure" \
        --save-dir "$save" \
        --wandb-run-name "s7_${tag}" \
        $extra \
        > "$logdir/train.log" 2>&1 &
    echo "  PID=$!"
}

run_one "bd_nolag_interp_causal"   "back_door"  "cuda:0" ""
run_one "bd_nolag_interp_obs"      "back_door"  "cuda:1" "--observational-only --obs-only-target Y_obs"
run_one "fd_nolag_interp_causal"   "front_door" "cuda:2" ""
run_one "fd_nolag_interp_obs"      "front_door" "cuda:3" "--observational-only --obs-only-target Y_obs"

echo ""
echo "All 4 s7 runs launched in parallel. Waiting..."
wait
echo ""
echo "============================================================"
echo "Sanity7 complete. Checkpoints in checkpoints/s7_*"
echo "Wandb: https://wandb.ai/dot-pfn/do-over-time-pfn"
echo "============================================================"
