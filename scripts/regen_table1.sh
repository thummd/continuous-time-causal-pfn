#!/usr/bin/env bash
# Regenerate the synthetic ablation table (tab:main) from the grid_v4
# checkpoints, plus the encoder cells on the mixed schedule reported in
# the appendix.
#
# Checkpoints come from Hugging Face:
#     python scripts/download_checkpoints.py --subset grid_v4 --out checkpoints
#
# The eval prior is the *common* prior of configs/continuous_ablation_grid.yaml
# (intervention_value_scale 1.0, matching the published table), so every
# checkpoint is scored against the same distribution rather than its own
# training prior.  Aggregate the resulting JSONs with:
#     python scripts/agg_finegrid.py --dir results/grid_v4_synth_finegrid_iv1
#
# Usage:
#     bash scripts/regen_table1.sh                  # single GPU, all cells
#     SHARDS=4 bash scripts/regen_table1.sh         # split across 4 GPUs
set -u

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${REPO_DIR}/checkpoints/grid_v4}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/results/grid_v4_synth_finegrid_iv1}"
CONFIG="${CONFIG:-${REPO_DIR}/configs/continuous_ablation_grid.yaml}"
N_EVAL_BATCHES="${N_EVAL_BATCHES:-50}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SHARDS="${SHARDS:-1}"
SHARD="${SHARD:-0}"          # set automatically when SHARDS > 1
PY="${PY:-python}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"

# Fan out across GPUs by re-invoking this script once per shard.
if [ "$SHARDS" -gt 1 ] && [ "${_SHARDED:-0}" = "0" ]; then
    for s in $(seq 0 $((SHARDS - 1))); do
        _SHARDED=1 SHARD="$s" SHARDS="$SHARDS" DEVICE="cuda:$s" \
            bash "$0" &
    done
    wait
    echo "all shards done"
    exit 0
fi
DEVICE="${DEVICE:-cuda:0}"

i=0
for enc in time pos; do
  for sim in naive fine; do
    for mech_cell in OU:linear neural:neural; do
      cell_mech="${mech_cell%%:*}"; mech="${mech_cell##*:}"
      for seed in 0 1 2; do
        # tab:main(a) is the regular unit-gap eval; (b) is the mixed
        # schedule at two eval resolutions.  The pos x mixed cells are
        # the appendix's encoder comparison.
        for job in "regular 1 reg_s1" "mixed 1 mixed_s1" "mixed 8 mixed_s8"; do
          set -- $job; sched="$1"; sub="$2"; tag="$3"
          if [ $((i % SHARDS)) -eq "$SHARD" ]; then
            cell="${enc}_${sim}_${cell_mech}"
            ckpt="${CKPT_ROOT}/${cell}/seed_${seed}/continuous_do_over_time_pfn_best.pt"
            out="${OUT_DIR}/${cell}_seed${seed}_${tag}.json"
            if [ -f "$out" ]; then
              echo "skip  $(basename "$out")"
            elif [ ! -f "$ckpt" ]; then
              echo "MISSING checkpoint: $ckpt" >&2
            else
              $PY scripts/ct_synth_eval.py \
                  --checkpoint "$ckpt" \
                  --mechanism-kind "$mech" \
                  --schedule "$sched" --dt 1.0 --jitter 0.3 \
                  --substeps "$sub" \
                  --n-eval-batches "$N_EVAL_BATCHES" \
                  --batch-size "$BATCH_SIZE" \
                  --device "$DEVICE" \
                  --save-json "$out" \
                && echo "ok    $(basename "$out")" \
                || echo "FAIL  $(basename "$out")" >&2
            fi
          fi
          i=$((i + 1))
        done
      done
    done
  done
done
echo "shard ${SHARD} done"
