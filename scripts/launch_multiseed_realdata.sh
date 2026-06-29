#!/usr/bin/env bash
# Multi-seed retraining for the real-data table (tab:real).
#
# The Phase-13b/14b real-data numbers (Warfarin + Causal-Chamber-WT) come
# from a SINGLE seed (seed0).  This script trains the two paper cells
#   - p13b_pnc000_linear  (p_no_context=0.0, mechanism=linear, p_neural=0.0)
#   - p13b_pnc000_mixed   (p_no_context=0.0, mechanism=mixed,  p_neural=0.5)
# for the ADDITIONAL seeds requested via ${SEEDS} (default "1 2 3 4"), so
# that combined with the existing seed0 we have a 5-seed mean+/-std.
#
# Everything else (steps, prior, batch, optimiser) mirrors
# launch_phase13b.sh so the new checkpoints are drop-in comparable to the
# seed0 ones.  Checkpoints land in the SAME directory layout
#   checkpoints/phase13b/p13b_pnc000_<mech>_seed<S>/
# so the downstream eval/aggregation scripts treat all seeds uniformly.
#
# Usage (on the server, in an SSH session):
#   export WANDB_API_KEY=...        # or rely on ~/.netrc
#   bash scripts/launch_multiseed_realdata.sh
#   # custom seeds / GPUs:
#   SEEDS="1 2 3 4" GPUS=cuda:2,cuda:3 bash scripts/launch_multiseed_realdata.sh
#
# Wall-clock: 2 cells x 4 seeds = 8 runs @ 5000 steps (~1-2 s/step on A100,
# slower under contention).  Across 2 GPUs that is 4 waves => ~8-16 h.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)/.."
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

# Configuration knobs (override via env if desired).
STEPS="${TOTAL_STEPS:-5000}"
SEEDS="${SEEDS:-1 2 3 4}"
BATCH="${BATCH_SIZE:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-ct-cpfn-phase14c-multiseed}"
WANDB_ENTITY="${WANDB_ENTITY:-ct-cpfn}"
N_MIN="${N_MIN_PRIOR:-3}"
N_MAX_PRIOR="${N_MAX_PRIOR:-8}"
EDGE_PROB="${EDGE_PROB:-0.3}"
HIDDEN_PROB="${HIDDEN_PROB:-0.3}"
NUM_SUBSTEPS="${NUM_SUBSTEPS:-2}"

# Locate conda + activate the existing dotime-fullscale env.
for cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$cb/etc/profile.d/conda.sh" ]; then CONDA_BASE="$cb"; break; fi
done
if [ -z "${CONDA_BASE:-}" ]; then
    echo "ERROR: could not find conda installation" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dotime-fullscale

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set; relying on ~/.netrc if present."
fi

LOG_DIR="${REPO_DIR}/logs/phase14c_multiseed"
CKPT_ROOT="${REPO_DIR}/checkpoints/phase13b"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

COMMON_ARGS=(
    "--config"          "${REPO_DIR}/configs/continuous_default.yaml"
    "--prior-mode"      "random"
    "--n-min-prior"     "${N_MIN}"
    "--n-max-prior"     "${N_MAX_PRIOR}"
    "--edge-prob"       "${EDGE_PROB}"
    "--hidden-prob"     "${HIDDEN_PROB}"
    "--num-substeps"    "${NUM_SUBSTEPS}"
    "--total-steps"     "${STEPS}"
    "--batch-size"      "${BATCH}"
    "--wandb-project"   "${WANDB_PROJECT}"
    "--wandb-entity"    "${WANDB_ENTITY}"
)

echo "============================================================"
echo "Phase 14c: multi-seed real-data retraining"
echo "  seeds=[${SEEDS}], cells=pnc000_{linear,mixed}"
echo "  steps=${STEPS}, batch=${BATCH}"
echo "  prior=random N in [${N_MIN}, ${N_MAX_PRIOR}], edge_prob=${EDGE_PROB}"
echo "  hidden_prob=${HIDDEN_PROB}, num_substeps=${NUM_SUBSTEPS}"
echo "  wandb: ${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "  logs:  ${LOG_DIR}"
echo "  ckpts: ${CKPT_ROOT}"
echo "============================================================"

# Launch one (seed, cell) job on a given device.
launch_cell() {
    local seed="$1" mech="$2" pneural="$3" device="$4"
    local run_name="p13b_pnc000_${mech}_seed${seed}"
    local save="${CKPT_ROOT}/${run_name}"
    local log="${LOG_DIR}/${run_name}.log"
    mkdir -p "${save}"

    echo "  -> ${run_name} on ${device}  (mech=${mech}, p_neural=${pneural}, seed=${seed})"
    PYTHONPATH="${REPO_DIR}" python -u "${REPO_DIR}/scripts/ct_train.py" "${COMMON_ARGS[@]}" \
        --seed "${seed}" \
        --device "${device}" \
        --mechanism-kind "${mech}" \
        --p-neural "${pneural}" \
        --p-no-context "0.0" \
        --save-dir "${save}" \
        --wandb-run-name "${run_name}" \
        > "${log}" 2>&1 &
    echo "     PID=$!"
}

IFS=',' read -ra GPU_LIST <<< "${GPUS:-cuda:2,cuda:3}"
N_GPUS=${#GPU_LIST[@]}
if [ "${N_GPUS}" -lt 1 ]; then
    echo "ERROR: GPUS must list at least one device" >&2
    exit 1
fi

# Build the full (seed|mech|pneural) job list: 2 cells per seed.
JOBS=()
for seed in ${SEEDS}; do
    JOBS+=("${seed}|linear|0.0")
    JOBS+=("${seed}|mixed|0.5")
done

n_jobs=${#JOBS[@]}
n_waves=$(( (n_jobs + N_GPUS - 1) / N_GPUS ))
echo ""
echo "Plan: ${n_jobs} jobs across ${N_GPUS} GPU(s) (${GPU_LIST[*]}) in ${n_waves} wave(s)"

for ((w=0; w<n_waves; w++)); do
    echo ""
    echo "=== Wave $((w + 1))/${n_waves} ==="
    for ((g=0; g<N_GPUS; g++)); do
        idx=$((w * N_GPUS + g))
        if [ "${idx}" -ge "${n_jobs}" ]; then
            break
        fi
        IFS='|' read -r seed mech pneural <<< "${JOBS[$idx]}"
        launch_cell "${seed}" "${mech}" "${pneural}" "${GPU_LIST[$g]}"
    done
    wait
    echo "Wave $((w + 1)) complete."
done

echo ""
echo "============================================================"
echo "Phase 14c multi-seed training complete: ${n_jobs} checkpoints"
echo "  in ${CKPT_ROOT}/ (p13b_pnc000_{linear,mixed}_seed{${SEEDS// /,}})"
echo "============================================================"
