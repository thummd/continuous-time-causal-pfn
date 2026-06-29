#!/usr/bin/env bash
# Multi-seed real-data evaluation driver (tab:real).
#
# For each (seed, mechanism) checkpoint trained by
# launch_multiseed_realdata.sh (plus the pre-existing seed0), run the
# zero-shot eval used in the paper:
#   - Causal-Chamber-WT: 3 query vars (rpm_in, current_in, pressure_downwind)
#                        on wt_intake_impulse_v1/load_out_0.5_osr_downwind_4
#   - Warfarin PK/PD:    dose_scale=0.01, absorption=1.0h, N=0 padding
#
# Eval is deterministic given a checkpoint, so the only seed variation
# comes from the trained model.  Results land in results/phase14c_multiseed/
# with a uniform naming scheme that aggregate_multiseed_realdata.py reads.
#
# Usage (on the server):
#   SEEDS="0 1 2 3 4" GPU=cuda:2 bash scripts/eval_multiseed_realdata.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)/.."
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

SEEDS="${SEEDS:-0 1 2 3 4}"
MECHS="${MECHS:-linear mixed}"
GPU="${GPU:-cuda:0}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_DIR}/checkpoints/phase13b}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/results/phase14c_multiseed}"
CHAMBER_QVARS="${CHAMBER_QVARS:-rpm_in current_in pressure_downwind}"
CHAMBER_EXP="load_out_0.5_osr_downwind_4"
# Paper protocol caps at the first 200 episodes (tab:real / Phase 14b);
# load_wt_episodes is otherwise uncapped (~9998 episodes) and would neither
# reproduce the committed seed0 numbers nor finish quickly.
CHAMBER_MAX_EPISODES="${CHAMBER_MAX_EPISODES:-200}"
# Persistent CausalChamber cache: the upstream library does NOT mkdir its
# download root, and /tmp gets periodically wiped -- so default to a stable
# home-dir cache and create it up front.
CHAMBER_ROOT="${CHAMBER_ROOT:-${HOME}/causalchamber_cache}"
mkdir -p "${CHAMBER_ROOT}"

for cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$cb/etc/profile.d/conda.sh" ]; then CONDA_BASE="$cb"; break; fi
done
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dotime-fullscale

mkdir -p "${OUT_DIR}"
CKPT_FILE="continuous_do_over_time_pfn_best.pt"

run_eval() {
    PYTHONPATH="${REPO_DIR}" python -u "${REPO_DIR}/scripts/ct_evaluate.py" "$@"
}

for seed in ${SEEDS}; do
    for mech in ${MECHS}; do
        ckpt="${CKPT_ROOT}/p13b_pnc000_${mech}_seed${seed}/${CKPT_FILE}"
        if [ ! -f "${ckpt}" ]; then
            echo "SKIP (missing checkpoint): ${ckpt}" >&2
            continue
        fi
        echo "==================================================="
        echo "seed=${seed} mech=${mech}  ckpt=${ckpt}"
        echo "==================================================="

        # --- Warfarin (N=0, paper protocol) ---
        # Per-eval failures are logged and skipped (not fatal) so one bad
        # cell can't abort the whole 40-eval sweep.
        wout="${OUT_DIR}/warfarin_p13b_pnc000_${mech}_seed${seed}.json"
        echo "  [warfarin] -> ${wout}"
        run_eval --checkpoint "${ckpt}" --benchmark warfarin \
            --warfarin-dose-scale 0.01 --warfarin-absorption-hours 1.0 \
            --pre-baseline-n 0 --device "${GPU}" --save-json "${wout}" \
            || echo "  !! FAILED warfarin seed=${seed} mech=${mech}" >&2

        # --- Causal-Chamber-WT (3 query vars) ---
        for qvar in ${CHAMBER_QVARS}; do
            cout="${OUT_DIR}/chamber_p13b_pnc000_${mech}_seed${seed}_${qvar}.json"
            echo "  [chamber ${qvar}] -> ${cout}"
            run_eval --checkpoint "${ckpt}" --benchmark causal_chamber \
                --chamber-rig wt --chamber-dataset wt_intake_impulse_v1 \
                --chamber-experiment "${CHAMBER_EXP}" \
                --chamber-query-var "${qvar}" --chamber-root "${CHAMBER_ROOT}" \
                --chamber-max-episodes "${CHAMBER_MAX_EPISODES}" \
                --device "${GPU}" --save-json "${cout}" \
                || echo "  !! FAILED chamber ${qvar} seed=${seed} mech=${mech}" >&2
        done
    done
done

echo ""
echo "All evals written to ${OUT_DIR}/"
