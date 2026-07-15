#!/usr/bin/env bash
# Train extra grid_v4 seeds for the two synthetic tables in continuous_ctp.tex.
#
# This is intentionally narrower than run_grid_experiments.sh: it trains only
# the 8 cells used by the current paper tables, then evaluates the 6 held-out
# distributions needed for later aggregation.
#
# Usage:
#   SEEDS="1 2" nohup scripts/run_grid_v4_paper_seeds.sh \
#       > experiments_v4_paper_seeds.log 2>&1 &
#
# Useful overrides:
#   SEEDS="1 2"          # checkpoint seeds to train/evaluate
#   DEVICE=mps           # mps | cuda | cpu
#   RUN_EVAL=0           # train only
#   CELLS="time_fine_OU" # optional subset for a repair run

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

SEEDS=${SEEDS:-"1 2"}
CONFIG=${CONFIG:-configs/continuous_default.yaml}
DEVICE=${DEVICE:-mps}
GRID_ROOT=${GRID_ROOT:-checkpoints/ct/grid_v4}
RESULTS_ROOT=${RESULTS_ROOT:-results/grid_v4_multiseed}
N_EVAL_BATCHES=${N_EVAL_BATCHES:-50}
EVAL_SEED=${EVAL_SEED:-99999}
RUN_EVAL=${RUN_EVAL:-1}
STEPS=${STEPS:-}
CTP_ROOT=${CTP_ROOT:-../CausalTimePrior}
DOPFN_ROOT=${DOPFN_ROOT:-${CTP_ROOT}/Do-PFN-prior}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}

export PYTHONPATH=".:${CTP_ROOT}:${DOPFN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-cache}

mkdir -p "$RESULTS_ROOT" "$MPLCONFIGDIR"

PAPER_CELLS=(
    pos_naive_OU
    pos_fine_OU
    time_naive_OU
    time_fine_OU
    pos_naive_neural
    pos_fine_neural
    time_naive_neural
    time_fine_neural
)

if [[ -n "${CELLS:-}" ]]; then
    read -r -a PAPER_CELLS <<< "$CELLS"
fi

COMMON=(
    --config "$CONFIG"
    --device "$DEVICE"
)
[[ -n "$STEPS" ]] && COMMON+=(--total-steps "$STEPS")

run_train() {
    local cell="$1"; shift
    local seed="$1"; shift
    local mech_flag="$1"; shift
    local substeps="$1"; shift
    local save_dir="${GRID_ROOT}/${cell}/seed_${seed}"
    local ckpt_path="${save_dir}/continuous_do_over_time_pfn_best.pt"

    if [[ -f "$ckpt_path" ]]; then
        echo "[skip] train ${cell} seed=${seed} (checkpoint exists at ${ckpt_path})"
        return
    fi

    local rel="${save_dir#${GRID_ROOT}/}"
    local wandb_args=()
    if [[ -n "$WANDB_PROJECT" ]]; then
        wandb_args+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "${rel//\//_}")
        [[ -n "$WANDB_ENTITY" ]] && wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi

    echo ""
    echo "========================================"
    echo "  TRAIN ${cell} seed=${seed}"
    echo "  $(date)"
    [[ -n "$WANDB_PROJECT" ]] && echo "  wandb: ${WANDB_PROJECT}/${rel//\//_}"
    echo "========================================"

    uv run python scripts/ct_train.py \
        "${COMMON[@]}" \
        --mechanism-kind "$mech_flag" \
        --substeps "$substeps" \
        --seed "$seed" \
        --save-dir "$save_dir" \
        "$@" \
        ${wandb_args[@]+"${wandb_args[@]}"}

    echo "  DONE: ${cell} seed=${seed} ($(date))"
}

cell_mechanism() {
    case "$1" in
        *_OU) echo linear ;;
        *_neural) echo neural ;;
        *) echo "unknown cell mechanism for $1" >&2; return 1 ;;
    esac
}

cell_substeps() {
    case "$1" in
        *_naive_*) echo 1 ;;
        *_fine_*) echo 8 ;;
        *) echo "unknown cell integrator for $1" >&2; return 1 ;;
    esac
}

echo "============================================================"
echo "  grid_v4 paper seeds"
echo "  seeds: ${SEEDS}"
echo "  device: ${DEVICE}"
echo "  cells: ${PAPER_CELLS[*]}"
echo "  started: $(date)"
echo "============================================================"

for seed in $SEEDS; do
    for cell in "${PAPER_CELLS[@]}"; do
        mech_flag=$(cell_mechanism "$cell")
        substeps=$(cell_substeps "$cell")
        pos_args=()
        [[ "$cell" == pos_* ]] && pos_args+=(--positional-only)

        run_train "$cell" "$seed" "$mech_flag" "$substeps" \
            ${pos_args[@]+"${pos_args[@]}"}
    done
done

if [[ "$RUN_EVAL" != "1" ]]; then
    echo ""
    echo "RUN_EVAL=${RUN_EVAL}; skipping held-out evals."
    exit 0
fi

seed_tag=${SEEDS// /_}

run_eval() {
    local mech_flag="$1"; shift
    local schedule="$1"; shift
    local substeps="$1"; shift
    local out="${RESULTS_ROOT}/eval_${mech_flag}_${schedule}_dt1.0_s${substeps}_seeds_${seed_tag}.json"

    echo ""
    echo "========================================"
    echo "  EVAL ${mech_flag} ${schedule} substeps=${substeps}"
    echo "  -> ${out}"
    echo "  $(date)"
    echo "========================================"

    uv run python scripts/eval_all_on_discrete_OU.py \
        --device "$DEVICE" \
        --n-eval-batches "$N_EVAL_BATCHES" \
        --eval-seed "$EVAL_SEED" \
        --checkpoint-seeds $SEEDS \
        --mechanism-kind "$mech_flag" \
        --schedule "$schedule" \
        --dt 1.0 \
        --substeps "$substeps" \
        --cells "${PAPER_CELLS[@]}" \
        --save-json "$out"
}

run_eval linear regular 1
run_eval neural regular 1
run_eval linear mixed 1
run_eval linear mixed 8
run_eval neural mixed 1
run_eval neural mixed 8

echo ""
echo "============================================================"
echo "  grid_v4 paper seeds complete"
echo "  finished: $(date)"
echo "  checkpoints: ${GRID_ROOT}"
echo "  eval jsons: ${RESULTS_ROOT}"
echo "============================================================"
