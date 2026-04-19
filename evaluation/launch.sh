#!/bin/bash
# Submit one sbatch job per (baseline, case_study) pair.
#
# Usage:
#   cd evaluation/
#   export ACCOUNT=<slurm_account>
#   bash launch.sh
#
# Generated configs land in eval_configs/generated/ so each job has a
# reproducible record of exactly what it ran.

PARTITION="alldlc2_gpu-l40s"
N_SAMPLES=500
T=1000

CASE_STUDIES=("back_door")

BASELINES=(
    "AR1Baseline"
    "ZeroBaseline"
    "BackDoorTabPFNInterventional"
    "BackDoorTabPFNObservational"
    "BackDoorTabPFNCausalEffect"
    "BackDoorDoTPFNCausalEffect"
    "BackDoorObsPFNCausalEffect"
    "Chronos2Observational"
)

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${EVAL_DIR}/results/logs"
CFG_DIR="${EVAL_DIR}/eval_configs/generated"
mkdir -p "${LOG_DIR}" "${CFG_DIR}"

for CASE in "${CASE_STUDIES[@]}"; do
    for BASELINE in "${BASELINES[@]}"; do

        CFG_FILE="${CFG_DIR}/${BASELINE}_${CASE}.yaml"
        LOG_FILE="${LOG_DIR}/${BASELINE}_${CASE}.log"

        # Write a minimal config for this job
        cat > "${CFG_FILE}" <<EOF
baseline: ${BASELINE}
case_study: ${CASE}
n_samples: ${N_SAMPLES}
T: ${T}
device: cuda
query_offsets: [0]
burn_in: 50
bootstrap_n: 1000
EOF

        echo "Launching ${BASELINE} / ${CASE} ..."

        sbatch --account="${ACCOUNT}" \
               --partition="${PARTITION}" \
               --job-name="eval_${BASELINE}" \
               --output="${LOG_FILE}" \
               --error="${LOG_FILE}" \
               --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
               --time=02:00:00 --gres=gpu:1 \
               --wrap="export PYTHONPATH=/work/dlclarge1/robertsj-dotpfn/CausalTimePrior:\$PYTHONPATH && \
                       cd ${EVAL_DIR} && \
                       python prior_eval.py ${CFG_FILE}"
    done
done
