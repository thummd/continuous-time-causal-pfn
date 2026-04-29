#!/usr/bin/env bash
# Phase 13b: zero-context training augmentation grid on the GPU server.
#
# Submits a 3 x 2 = 6-cell training grid (p_no_context x mechanism_kind)
# directly via background ``python ... &`` dispatch.
#
# GPU assignment is driven by the ``GPUS`` env var (comma-separated list).
# Default targets ``cuda:0,cuda:2`` -- the two free slots on
# aidf-svr-gpu04 when the box is shared with VLLM / sinkhorn / dennis's
# own DoT-PFN runs on cuda:1 and cuda:3.  Cells are dispatched in waves
# of size ``len(GPUS)``; the number of waves is ``ceil(6 / len(GPUS))``.
#
# Logs to wandb entity ``ct-cpfn``, project ``ct-cpfn-phase13b``.  Run
# names encode the cell, e.g.\ ``p13b_pnc010_mixed_seed0``.
#
# Usage (on the server, in an SSH session):
#   export WANDB_API_KEY=...           # or `wandb login` once before running
#   bash scripts/launch_phase13b.sh
#   # or with explicit GPU subset:
#   GPUS=cuda:0,cuda:2 bash scripts/launch_phase13b.sh
#
# Wall-clock estimate: 5000 steps at ~1-2 s/step on A100 ~= 1.5-3 h per
# run; with shared-GPU contention typically 2-4 h.
#   - default 2-GPU config: 3 waves x ~3-4h = 9-12 h end-to-end
#   - 4-GPU config:         2 waves x ~2-4h = 4-8 h end-to-end
#
# Mirrors the layout of ``do-over-time-pfn/scripts/run_sanity9.sh`` so
# anyone familiar with the upstream pipeline can read this in one pass.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)/.."
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

# Configuration knobs (override via env if desired).
STEPS="${TOTAL_STEPS:-5000}"
SEED="${SEED:-0}"
BATCH="${BATCH_SIZE:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-ct-cpfn-phase13b}"
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
    echo "WARNING: WANDB_API_KEY not set in environment.  Either run"
    echo "  wandb login   (interactive, recommended)"
    echo "before launching, or export WANDB_API_KEY=... in the shell."
    echo "Continuing in case ~/.netrc already has a token..."
fi

LOG_DIR="${REPO_DIR}/logs/phase13b"
CKPT_ROOT="${REPO_DIR}/checkpoints/phase13b"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

# Common args, mirroring the structure of run_sanity9.sh.
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
    "--seed"            "${SEED}"
    "--wandb-project"   "${WANDB_PROJECT}"
    "--wandb-entity"    "${WANDB_ENTITY}"
)

echo "============================================================"
echo "Phase 13b: zero-context training augmentation grid"
echo "  steps=${STEPS}, batch=${BATCH}, seed=${SEED}"
echo "  prior=random N in [${N_MIN}, ${N_MAX_PRIOR}], edge_prob=${EDGE_PROB}"
echo "  hidden_prob=${HIDDEN_PROB}, num_substeps=${NUM_SUBSTEPS}"
echo "  wandb: ${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "  logs:  ${LOG_DIR}"
echo "  ckpts: ${CKPT_ROOT}"
echo "============================================================"

# Each cell (p_no_context, mechanism_kind, p_neural).
launch_cell() {
    local pnc="$1" mech="$2" pneural="$3" device="$4"
    local pnc_tag
    pnc_tag=$(printf "%03d" "$(awk "BEGIN{print int(${pnc}*100)}")")
    local run_name="p13b_pnc${pnc_tag}_${mech}_seed${SEED}"
    local save="${CKPT_ROOT}/${run_name}"
    local log="${LOG_DIR}/${run_name}.log"
    mkdir -p "${save}"

    echo "  -> ${run_name} on ${device}  (p_no_context=${pnc}, mech=${mech}, p_neural=${pneural})"
    # ``python -u`` flushes stdout/stderr line-by-line so per-run logs
    # show training progress without buffering the first few hundred
    # steps until eval-time -- matches do-over-time-pfn's run_sanity9 style.
    PYTHONPATH="${REPO_DIR}" python -u "${REPO_DIR}/scripts/ct_train.py" "${COMMON_ARGS[@]}" \
        --device "${device}" \
        --mechanism-kind "${mech}" \
        --p-neural "${pneural}" \
        --p-no-context "${pnc}" \
        --save-dir "${save}" \
        --wandb-run-name "${run_name}" \
        > "${log}" 2>&1 &
    echo "     PID=$!"
}

# Parse GPU list and dispatch the 6 cells round-robin in waves of len(GPUS).
IFS=',' read -ra GPU_LIST <<< "${GPUS:-cuda:0,cuda:2}"
N_GPUS=${#GPU_LIST[@]}
if [ "${N_GPUS}" -lt 1 ]; then
    echo "ERROR: GPUS must list at least one device" >&2
    exit 1
fi

# Cell definitions (p_no_context, mechanism_kind, p_neural).
CELLS=(
    "0.0|linear|0.0"
    "0.0|mixed|0.5"
    "0.1|linear|0.0"
    "0.1|mixed|0.5"
    "0.2|linear|0.0"
    "0.2|mixed|0.5"
)

n_cells=${#CELLS[@]}
n_waves=$(( (n_cells + N_GPUS - 1) / N_GPUS ))
echo ""
echo "Plan: ${n_cells} cells across ${N_GPUS} GPU(s) (${GPU_LIST[*]}) in ${n_waves} wave(s)"

for ((w=0; w<n_waves; w++)); do
    echo ""
    echo "=== Wave $((w + 1))/${n_waves} ==="
    for ((g=0; g<N_GPUS; g++)); do
        idx=$((w * N_GPUS + g))
        if [ "${idx}" -ge "${n_cells}" ]; then
            break
        fi
        IFS='|' read -r pnc mech pneural <<< "${CELLS[$idx]}"
        launch_cell "${pnc}" "${mech}" "${pneural}" "${GPU_LIST[$g]}"
    done
    wait
    echo "Wave $((w + 1)) complete."
done

echo ""
echo "============================================================"
echo "Phase 13b complete: 6 checkpoints in ${CKPT_ROOT}/"
echo "Wandb: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "============================================================"
