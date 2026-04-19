#!/usr/bin/env bash
# Sanity8: per-structure query_offset + longer training trajectories.
#
# Key changes vs s7:
#   - t_range: [500, 1000] (up from [50, 200]) — SCM reaches steady state
#     before the intervention, matching collaborators' T=1000 eval protocol.
#     Encoder still truncates to context_window=200, but those 200 steps are
#     now fully settled.
#   - Per-structure query_offset_range:
#       BD: [0, 0]  — instantaneous back-door signal is strongest at t=int_time
#       FD: [1, 5]  — skip offset=0 so mediator chain has time to propagate
#
# Everything else matches s7: positivity_aware, mixer=3, Y_obs obs-only target,
# early stopping, wandb, 4-GPU parallel.
#
# Usage:
#   export WANDB_API_KEY=...
#   bash scripts/run_sanity8.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)/.."
REPO_DIR="$(cd "$REPO_DIR" && pwd)"
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
    echo "WARNING: WANDB_API_KEY not set."
fi

echo "============================================================"
echo "Sanity8 (parallel; s7 fixes + T=500..1000 + per-structure offset)"
echo "  steps=$STEPS, n_queries=$N_QUERIES, batch=$BATCH"
echo "  4 runs in parallel on GPUs 0,1,2,3"
echo "============================================================"

COMMON_ARGS=(
    "--config" "$REPO_DIR/configs/server_s8.yaml"
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
    local offset_lo="$4"
    local offset_hi="$5"
    local extra="$6"
    local save="$REPO_DIR/checkpoints/s8_${tag}"
    local logdir="$REPO_DIR/results/s8_${tag}"
    mkdir -p "$save" "$logdir"
    echo "Launching s8_${tag} on $device (structure=$structure, offset=[$offset_lo, $offset_hi])"
    python "$REPO_DIR/scripts/train.py" "${COMMON_ARGS[@]}" \
        --device "$device" \
        --tscm-structure "$structure" \
        --query-offset-range "$offset_lo" "$offset_hi" \
        --save-dir "$save" \
        --wandb-run-name "s8_${tag}" \
        $extra \
        > "$logdir/train.log" 2>&1 &
    echo "  PID=$!"
}

# BD: offset=0 (instantaneous confounding is the signal)
run_one "bd_nolag_interp_causal"   "back_door"  "cuda:0" 0 0 ""
run_one "bd_nolag_interp_obs"      "back_door"  "cuda:1" 0 0 "--observational-only --obs-only-target Y_obs"

# FD: offset in [1, 5] so mediator has time to propagate the intervention
run_one "fd_nolag_interp_causal"   "front_door" "cuda:2" 1 5 ""
run_one "fd_nolag_interp_obs"      "front_door" "cuda:3" 1 5 "--observational-only --obs-only-target Y_obs"

echo ""
echo "All 4 s8 runs launched in parallel. Waiting..."
wait
echo ""
echo "============================================================"
echo "Sanity8 complete. Checkpoints in checkpoints/s8_*"
echo "Wandb: https://wandb.ai/dot-pfn/do-over-time-pfn"
echo "============================================================"
