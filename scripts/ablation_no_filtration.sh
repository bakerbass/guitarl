#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Ablation study: NO FILTRATION
#
# Layer 1 (physics gate) is completely bypassed.  Every action — regardless
# of torque range, fret distance, or silence — is fed directly to the CNN
# classifier (Layer 2) for a reward signal.
#
# Purpose: quantify how much the physics gate contributes to sample efficiency.
# If the agent learns equally well without it, the gate is unnecessary overhead.
# If it learns slower or not at all, the gate is doing meaningful work.
#
# Compare run output against scripts/ablation_no_audio.sh and the full baseline.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-../HarmonicsClassifier/models/best_model.pt}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
STRING_INDICES="${STRING_INDICES:-0 2 4}"
CURRICULUM="${CURRICULUM:-easy_to_hard}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/ablation_no_filtration}"

echo "=========================================="
echo " Ablation: NO FILTRATION (Layer 2 only)"
echo "=========================================="
echo "  model:      $MODEL_PATH"
echo "  timesteps:  $TOTAL_TIMESTEPS"
echo "  strings:    $STRING_INDICES"
echo "  curriculum: $CURRICULUM"
echo "  output:     $OUTPUT_DIR"
echo ""

cd "$REPO_ROOT"

# shellcheck disable=SC2086
conda run -n guitaRL python train.py \
    --model-path "$MODEL_PATH" \
    --reward-mode no_filtration \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --string-indices $STRING_INDICES \
    --curriculum "$CURRICULUM" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
