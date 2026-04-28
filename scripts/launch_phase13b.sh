#!/bin/bash
# Submit Phase 13b training runs (zero-context augmentation) to a Slurm
# cluster, logging to wandb under the ct-cpfn team.
#
# Usage:
#   export ACCOUNT=<slurm_account>
#   export WANDB_API_KEY=<...>          # or run `wandb login` once on the cluster
#   bash scripts/launch_phase13b.sh
#
# What it submits
# ---------------
# A small grid that varies p_no_context and mechanism_kind, with the
# rest of the prior held fixed.  Each cell is a single 5k-step training
# run on the random-graph prior; the resulting checkpoints can then be
# evaluated zero-shot on Theophylline / Warfarin / CausalChamber.
#
# Grid (3 x 2 = 6 runs)
#   p_no_context in {0.0, 0.1, 0.2}
#   mechanism_kind in {linear, mixed (p_neural=0.5)}
#
# To submit a single canonical run instead, scroll to "SINGLE_RUN" below.
#
# Wandb
# -----
# entity:  ct-cpfn   (the team you set up at https://wandb.ai/ct-cpfn)
# project: ct-cpfn-phase13b
# run name: encodes the cell, e.g. "p013b_pnc010_neural_seed0"

set -euo pipefail

PARTITION="${PARTITION:-alldlc2_gpu-l40s}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ct-cpfn-phase13b}"
WANDB_ENTITY="${WANDB_ENTITY:-ct-cpfn}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/phase13b"
CKPT_ROOT="${REPO_ROOT}/checkpoints/phase13b"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

if [[ -z "${ACCOUNT:-}" ]]; then
    echo "ERROR: export ACCOUNT=<slurm account> first." >&2
    exit 1
fi

# ---- Grid definition ----
P_NO_CONTEXTS=("0.0" "0.1" "0.2")
MECHS=("linear" "mixed")

submit_run () {
    local pnc="$1"          # p_no_context
    local mech="$2"         # mechanism_kind
    local pneural=0.0
    if [[ "${mech}" == "mixed" ]]; then
        pneural=0.5
    fi

    local pnc_tag
    pnc_tag=$(printf "%03d" "$(awk "BEGIN{print int(${pnc}*100)}")")
    local run_name="p13b_pnc${pnc_tag}_${mech}_seed${SEED}"
    local save_dir="${CKPT_ROOT}/${run_name}"
    local log_file="${LOG_DIR}/${run_name}.log"
    mkdir -p "${save_dir}"

    echo "Submitting ${run_name}  (p_no_context=${pnc}, mechanism=${mech}, p_neural=${pneural})"

    sbatch --account="${ACCOUNT}" \
           --partition="${PARTITION}" \
           --job-name="${run_name}" \
           --output="${log_file}" \
           --error="${log_file}" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G \
           --time="${TIME_LIMIT}" --gres=gpu:1 \
           --wrap="cd ${REPO_ROOT} && \
                   export PYTHONPATH=${REPO_ROOT}:\$PYTHONPATH && \
                   python scripts/ct_train.py \
                       --config configs/continuous_default.yaml \
                       --prior-mode random \
                       --n-min-prior 3 --n-max-prior 8 \
                       --edge-prob 0.3 \
                       --hidden-prob 0.3 \
                       --mechanism-kind ${mech} \
                       --p-neural ${pneural} \
                       --p-no-context ${pnc} \
                       --num-substeps 8 \
                       --total-steps ${TOTAL_STEPS} \
                       --seed ${SEED} \
                       --save-dir ${save_dir} \
                       --wandb-project ${WANDB_PROJECT} \
                       --wandb-entity ${WANDB_ENTITY} \
                       --wandb-run-name ${run_name}"
}

for pnc in "${P_NO_CONTEXTS[@]}"; do
    for mech in "${MECHS[@]}"; do
        submit_run "${pnc}" "${mech}"
    done
done

cat <<EOM

Submitted $(( ${#P_NO_CONTEXTS[@]} * ${#MECHS[@]} )) jobs.
Logs:        ${LOG_DIR}
Checkpoints: ${CKPT_ROOT}
Wandb:       https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}

# SINGLE_RUN: if you just want one canonical training run, run e.g.
#
#   sbatch --account="\${ACCOUNT}" --partition="${PARTITION}" \\
#          --gres=gpu:1 --time=${TIME_LIMIT} --cpus-per-task=4 --mem=24G \\
#          --wrap="cd ${REPO_ROOT} && PYTHONPATH=${REPO_ROOT} python scripts/ct_train.py \\
#                  --config configs/continuous_default.yaml \\
#                  --prior-mode random --n-min-prior 3 --n-max-prior 8 \\
#                  --hidden-prob 0.3 --mechanism-kind mixed --p-neural 0.5 \\
#                  --p-no-context 0.15 --num-substeps 8 \\
#                  --total-steps 5000 --seed 0 \\
#                  --save-dir ${CKPT_ROOT}/canonical \\
#                  --wandb-project ${WANDB_PROJECT} \\
#                  --wandb-entity ${WANDB_ENTITY} \\
#                  --wandb-run-name p13b_canonical"
EOM
