#!/usr/bin/env bash
# Full-scale pinball comparison: 100K steps on GPU server.
#
# Usage:
#   ssh 10.230.252.6
#   cd /path/to/do-over-time-pfn
#   bash scripts/run_full_scale.sh [cuda:1] [--skip-install] [128]
#
# Prerequisites on server:
#   - conda with Python 3.12
#   - The ctp repo at ../ctp (sibling directory)
#   - NVIDIA GPU with CUDA (or adapt torch install for ROCm)

set -euo pipefail

DEVICE="${1:-cuda:1}"
SKIP_INSTALL="${2:-}"
BATCH_SIZE="${3:-128}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CTP_DIR="$(cd "$REPO_DIR/../ctp" 2>/dev/null && pwd || echo "")"
ENV_NAME="dotime-fullscale"
RESULTS_DIR="$REPO_DIR/results/full_scale"
CKPT_DIR="$REPO_DIR/checkpoints/full_scale"

echo "============================================================"
echo "Do-Over-Time-PFN: Full-Scale Pinball Comparison"
echo "============================================================"
echo "Repo:    $REPO_DIR"
echo "CTP:     ${CTP_DIR:-NOT FOUND}"
echo "Device:  $DEVICE"
echo "Results: $RESULTS_DIR"
echo ""

# --- Environment setup ---
if [ "$SKIP_INSTALL" != "--skip-install" ]; then
    echo "[1/5] Setting up conda environment..."

    # Source conda
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    # Create env if it doesn't exist
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        conda create -n "$ENV_NAME" python=3.12 -y
    fi
    conda activate "$ENV_NAME"

    # Install PyTorch (CUDA by default; change for ROCm)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

    # Install pfns from PyPI
    pip install pfns

    # Install dopfnprior from local ctp repo
    if [ -n "$CTP_DIR" ] && [ -d "$CTP_DIR/Do-PFN-prior" ]; then
        pip install -e "$CTP_DIR/Do-PFN-prior"
    else
        echo "WARNING: ctp/Do-PFN-prior not found. Install dopfnprior manually."
    fi

    # Install dotime
    pip install -e "$REPO_DIR"
else
    echo "[1/5] Skipping install (--skip-install)"
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

# Set PYTHONPATH for causal_time_prior (no setup.py)
if [ -n "$CTP_DIR" ]; then
    export PYTHONPATH="${CTP_DIR}:${PYTHONPATH:-}"
fi

# --- Verify imports ---
echo "[2/5] Verifying imports..."
python -c "
import torch, pfns, dopfnprior, causal_time_prior, dotime
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print('  All imports OK')
"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR"

# --- Train baseline (pinball_weight=0) ---
echo ""
echo "[3/5] Training BASELINE model (pinball_weight=0, 100K steps)..."
python "$REPO_DIR/scripts/train.py" \
    --config "$REPO_DIR/configs/default.yaml" \
    --device "$DEVICE" \
    --pinball-weight 0.0 \
    --batch-size "$BATCH_SIZE" \
    --save-dir "$CKPT_DIR/baseline_pw0" \
    2>&1 | tee "$RESULTS_DIR/train_baseline.log"

# --- Train pinball model (pinball_weight=0.1) ---
echo ""
echo "[4/5] Training PINBALL model (pinball_weight=0.1, 100K steps)..."
python "$REPO_DIR/scripts/train.py" \
    --config "$REPO_DIR/configs/default.yaml" \
    --device "$DEVICE" \
    --pinball-weight 0.1 \
    --batch-size "$BATCH_SIZE" \
    --save-dir "$CKPT_DIR/pinball_pw0.1" \
    2>&1 | tee "$RESULTS_DIR/train_pinball.log"

# --- Evaluate both on TSCM identifiability ---
echo ""
echo "[5/5] Running TSCM identifiability case studies..."

echo "--- Baseline ---"
python "$REPO_DIR/scripts/tscm_identifiability.py" \
    --checkpoint "$CKPT_DIR/baseline_pw0/do_over_time_pfn_best.pt" \
    --device "$DEVICE" \
    2>&1 | tee "$RESULTS_DIR/tscm_baseline.log"

echo ""
echo "--- Pinball ---"
python "$REPO_DIR/scripts/tscm_identifiability.py" \
    --checkpoint "$CKPT_DIR/pinball_pw0.1/do_over_time_pfn_best.pt" \
    --device "$DEVICE" \
    2>&1 | tee "$RESULTS_DIR/tscm_pinball.log"

echo ""
echo "============================================================"
echo "DONE. Results saved to: $RESULTS_DIR"
echo "  - train_baseline.log"
echo "  - train_pinball.log"
echo "  - tscm_baseline.log"
echo "  - tscm_pinball.log"
echo "============================================================"
