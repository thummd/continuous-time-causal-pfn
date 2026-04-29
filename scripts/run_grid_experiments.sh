#!/usr/bin/env bash
# Grid experiments for the FMSD workshop paper, EXPERIMENT_PLAN_v2 framing.
#
# Replaces run_all_experiments.sh.  Drops the schedule axis (every model
# trains on the unified ``mixed`` schedule) and replaces tiers A/B/C
# with the 2x2 architectural grid:
#
#   - positional vs time-aware encoder       (--positional-only flag)
#   - naive vs fine-grid EM integration       (--substeps 1 vs 8)
#   - x mechanism family (OU vs neural)       (--mechanism-kind)
#
# Pipeline (single overnight run, fully resumable — skips any
# checkpoint or eval JSON that already exists):
#
#   Table 1 train : 8 cells x SEEDS                ->  GRID_ROOT/{cell}/seed_N/
#   Table 1 eval  : synth eval per checkpoint      ->  RESULTS_ROOT/synth/{cell}_seed_N.json
#   Table 2 train : 4 cells x SEEDS (hidden conf.) ->  GRID_ROOT/table2/{tag}/seed_N/
#   Table 2 eval  : synth eval per checkpoint      ->  RESULTS_ROOT/synth/table2_{tag}_seed_N.json
#   Table 3 eval  : chamber eval on contrast pair  ->  RESULTS_ROOT/chamber/{cell}_seed_N.json
#
# Usage:
#   chmod +x scripts/run_grid_experiments.sh
#   nohup scripts/run_grid_experiments.sh > experiments.log 2>&1 &
#
# Env-var overrides:
#   SEEDS="0 1 2"     # space-separated seed list (default: "0")
#   STEPS=5000        # training steps per cell (default 5000)
#   DEVICE=mps        # mps | cuda | cpu (default: mps)
#   GRID_ROOT=...     # checkpoint root (default checkpoints/ct/grid)
#   RESULTS_ROOT=...  # eval-json root (default results/grid)

set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env into the script's own shell so WANDB_PROJECT / WANDB_ENTITY
# (and anything else) can live there alongside WANDB_API_KEY instead of
# being threaded through the launch command.  set -a auto-exports every
# variable assignment until set +a.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

SEEDS=${SEEDS:-"0"}
CONFIG=${CONFIG:-configs/continuous_default.yaml}
DEVICE=${DEVICE:-mps}
STEPS=${STEPS:-}                  # Empty by default; YAML's training.total_steps wins.
GRID_ROOT=${GRID_ROOT:-checkpoints/ct/grid}
RESULTS_ROOT=${RESULTS_ROOT:-results/grid}
N_EVAL_BATCHES=${N_EVAL_BATCHES:-50}

# Wandb (optional).  WANDB_API_KEY is read from .env via uv.  Leave
# WANDB_PROJECT empty to disable wandb logging entirely.
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}

mkdir -p "${RESULTS_ROOT}/synth" "${RESULTS_ROOT}/chamber"

# Per-run flags (config + device).  Everything else -- prior-mode,
# n_max_prior, edge_prob, schedule, vectorize, batch_size, lr,
# total_steps, embed_size, etc. -- comes from the YAML.  Edit the
# config to change any of those.
COMMON=(
    --config "$CONFIG"
    --device "$DEVICE"
)
# Only inject --total-steps if the env var is explicitly set; otherwise
# the YAML's training.total_steps wins.
[[ -n "$STEPS" ]] && COMMON+=(--total-steps "$STEPS")

run_train() {
    local tag="$1"; shift
    local save_dir="$1"; shift
    local ckpt_path="${save_dir}/continuous_do_over_time_pfn_best.pt"
    if [[ -f "$ckpt_path" ]]; then
        echo "[skip] train $tag (checkpoint exists at $ckpt_path)"
        return
    fi
    # Build wandb-friendly run name from save_dir relative to GRID_ROOT.
    # e.g. checkpoints/ct/grid_v4/pos_naive_OU/seed_0  ->  pos_naive_OU_seed_0
    #      checkpoints/ct/grid_v4/table2/hidden0.0_OU/seed_0  ->  table2_hidden0.0_OU_seed_0
    local rel="${save_dir#${GRID_ROOT}/}"
    local wandb_args=()
    if [[ -n "$WANDB_PROJECT" ]]; then
        wandb_args+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "${rel//\//_}")
        [[ -n "$WANDB_ENTITY" ]] && wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
    echo ""
    echo "========================================"
    echo "  TRAIN $tag"
    echo "  $(date)"
    [[ -n "$WANDB_PROJECT" ]] && echo "  wandb: ${WANDB_PROJECT}/${rel//\//_}"
    echo "========================================"
    uv run --env-file .env python scripts/ct_train.py \
        "${COMMON[@]}" "$@" --save-dir "$save_dir" \
        ${wandb_args[@]+"${wandb_args[@]}"}
    echo "  DONE: $tag ($(date))"
}

run_synth_eval() {
    local tag="$1"; shift
    local ckpt="$1"; shift
    local mech_flag="$1"; shift
    local out="$1"; shift
    local extra=("$@")
    if [[ ! -f "$ckpt" ]]; then
        echo "[skip] synth $tag (no checkpoint at $ckpt)"
        return
    fi
    if [[ -f "$out" ]]; then
        echo "[skip] synth $tag (json exists at $out)"
        return
    fi
    echo ""
    echo "========================================"
    echo "  SYNTH-EVAL $tag  ->  $out"
    echo "  $(date)"
    echo "========================================"
    uv run --env-file .env python scripts/ct_synth_eval.py \
        --checkpoint "$ckpt" \
        --mechanism-kind "$mech_flag" \
        --schedule mixed \
        --n-eval-batches "$N_EVAL_BATCHES" \
        --device "$DEVICE" \
        --save-json "$out" \
        ${extra[@]+"${extra[@]}"}
}

run_chamber_eval() {
    local tag="$1"; shift
    local ckpt="$1"; shift
    local out="$1"; shift
    if [[ ! -f "$ckpt" ]]; then
        echo "[skip] chamber $tag (no checkpoint at $ckpt)"
        return
    fi
    if [[ -f "$out" ]]; then
        echo "[skip] chamber $tag (json exists at $out)"
        return
    fi
    echo ""
    echo "========================================"
    echo "  CHAMBER-EVAL $tag  ->  $out"
    echo "  $(date)"
    echo "========================================"
    uv run --env-file .env python scripts/ct_evaluate.py \
        --checkpoint "$ckpt" \
        --benchmark causal_chamber \
        --chamber-dataset lt_walks_v1 \
        --chamber-experiment actuators_white \
        --chamber-max-episodes 50 \
        --device "$DEVICE" \
        --save-json "$out"
}

# ============================================================
# TABLE 1: 2x2 architectural grid x mechanism, on the mixed schedule.
# Cells:  pos_naive, pos_fine, time_naive, time_fine
# Mechs:  OU (linear), neural, mixed (p_neural=0.5 => per-variable Bernoulli)
# Total:  3 mechs * 2 enc * 2 substeps = 12 cells x SEEDS.
# ============================================================
P_NEURAL_MIXED=${P_NEURAL_MIXED:-0.5}

echo "============================================================"
echo "  TABLE 1: architectural grid (12 cells x ${SEEDS})"
echo "  Started: $(date)"
echo "============================================================"

for seed in $SEEDS; do
    for mech in OU neural mixed; do
        case "$mech" in
            OU)     mech_flag="linear"; mech_extra=() ;;
            neural) mech_flag="neural"; mech_extra=() ;;
            mixed)  mech_flag="mixed";  mech_extra=(--p-neural "$P_NEURAL_MIXED") ;;
        esac

        for posmode in pos time; do
            case "$posmode" in
                pos)  pos_flag=(--positional-only) ;;
                time) pos_flag=() ;;
            esac

            for steps_kind in naive fine; do
                case "$steps_kind" in
                    naive) substeps=1 ;;
                    fine)  substeps=8 ;;
                esac

                cell="${posmode}_${steps_kind}_${mech}"
                save_dir="${GRID_ROOT}/${cell}/seed_${seed}"

                run_train "T1 $cell seed=$seed" "$save_dir" \
                    --mechanism-kind "$mech_flag" \
                    ${mech_extra[@]+"${mech_extra[@]}"} \
                    --substeps "$substeps" \
                    ${pos_flag[@]+"${pos_flag[@]}"} \
                    --seed "$seed"
            done
        done
    done
done

# Synthetic eval for Table 1 (one JSON per cell x seed).
echo ""
echo "============================================================"
echo "  TABLE 1: synthetic eval"
echo "  Started: $(date)"
echo "============================================================"

for seed in $SEEDS; do
    for mech in OU neural mixed; do
        case "$mech" in
            OU)     mech_flag="linear"; mech_extra=() ;;
            neural) mech_flag="neural"; mech_extra=() ;;
            mixed)  mech_flag="mixed";  mech_extra=(--p-neural "$P_NEURAL_MIXED") ;;
        esac
        for posmode in pos time; do
            for steps_kind in naive fine; do
                cell="${posmode}_${steps_kind}_${mech}"
                ckpt="${GRID_ROOT}/${cell}/seed_${seed}/continuous_do_over_time_pfn_best.pt"
                out="${RESULTS_ROOT}/synth/${cell}_seed_${seed}.json"
                run_synth_eval "T1 $cell seed=$seed" "$ckpt" "$mech_flag" "$out" \
                    ${mech_extra[@]+"${mech_extra[@]}"}
            done
        done
    done
done

# ============================================================
# TABLE 2: hidden confounders on the time_fine row.
# 3 mechs x 2 hidden_prob in {0.0, 0.3}.  hidden_0 cells overlap with
# Table 1 time_fine cells but the train step is idempotent (skips if
# the checkpoint already exists).
# ============================================================

echo ""
echo "============================================================"
echo "  TABLE 2: hidden confounders (time_fine row)"
echo "  Started: $(date)"
echo "============================================================"

for seed in $SEEDS; do
    for mech in OU neural mixed; do
        case "$mech" in
            OU)     mech_flag="linear"; mech_extra=() ;;
            neural) mech_flag="neural"; mech_extra=() ;;
            mixed)  mech_flag="mixed";  mech_extra=(--p-neural "$P_NEURAL_MIXED") ;;
        esac

        for hp in 0.0 0.3; do
            tag="hidden${hp}_${mech}"
            save_dir="${GRID_ROOT}/table2/${tag}/seed_${seed}"

            run_train "T2 $tag seed=$seed" "$save_dir" \
                --mechanism-kind "$mech_flag" \
                ${mech_extra[@]+"${mech_extra[@]}"} \
                --substeps 8 \
                --hidden-prob "$hp" \
                --seed "$seed"
        done
    done
done

# Synthetic eval for Table 2 (passes --hidden-prob to the eval prior so
# it matches the training distribution).
echo ""
echo "============================================================"
echo "  TABLE 2: synthetic eval"
echo "  Started: $(date)"
echo "============================================================"

for seed in $SEEDS; do
    for mech in OU neural mixed; do
        case "$mech" in
            OU)     mech_flag="linear"; mech_extra=() ;;
            neural) mech_flag="neural"; mech_extra=() ;;
            mixed)  mech_flag="mixed";  mech_extra=(--p-neural "$P_NEURAL_MIXED") ;;
        esac
        for hp in 0.0 0.3; do
            tag="hidden${hp}_${mech}"
            ckpt="${GRID_ROOT}/table2/${tag}/seed_${seed}/continuous_do_over_time_pfn_best.pt"
            out="${RESULTS_ROOT}/synth/table2_${tag}_seed_${seed}.json"
            run_synth_eval "T2 $tag seed=$seed" "$ckpt" "$mech_flag" "$out" \
                ${mech_extra[@]+"${mech_extra[@]}"} \
                --hidden-prob "$hp"
        done
    done
done

# ============================================================
# TABLE 3: chamber zero-shot eval.
# Contrast rows: pos_naive_neural (worst tier), time_fine_neural and
# time_fine_mixed (best tier with single vs heterogeneous mechanisms).
# ============================================================

echo ""
echo "============================================================"
echo "  TABLE 3: chamber zero-shot eval"
echo "  Started: $(date)"
echo "============================================================"

for seed in $SEEDS; do
    for cell in pos_naive_neural time_fine_neural time_fine_mixed; do
        ckpt="${GRID_ROOT}/${cell}/seed_${seed}/continuous_do_over_time_pfn_best.pt"
        out="${RESULTS_ROOT}/chamber/${cell}_seed_${seed}.json"
        run_chamber_eval "T3 $cell seed=$seed" "$ckpt" "$out"
    done
done

echo ""
echo "============================================================"
echo "  GRID COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
echo ""
echo "  Checkpoints:    ${GRID_ROOT}"
echo "  Synth metrics:  ${RESULTS_ROOT}/synth/"
echo "  Chamber metrics:${RESULTS_ROOT}/chamber/"
