#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Ablation study: NO AUDIO (fret + torque shaping only)
#
# The CNN classifier is never invoked.  Layer 1 (filtration) still runs so
# the agent is penalised for silence, extreme torque, and badly placed frets.
# Layer 2 uses only the fret-accuracy and torque-optimality shaping terms,
# rebalanced to equal weights (0.5 fret / 0.5 torque).
#
# Audio is NOT captured in this mode, making each step significantly faster.
#
# Purpose: measure the marginal contribution of the harmonic classifier.
# If the policy converges to reasonable mechanics without audio feedback,
# the CNN may be providing only fine-grained refinement rather than coarse
# behavioural shaping.
#
# Compare run output against scripts/ablation_no_filtration.sh and the baseline.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-../HarmonicsClassifier/models/best_model.pt}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
STRING_INDICES="${STRING_INDICES:-0 2 4}"
CURRICULUM="${CURRICULUM:-easy_to_hard}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/ablation_no_audio}"

echo "=========================================="
echo " Ablation: NO AUDIO (fret+torque only)"
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
    --reward-mode no_audio \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --string-indices $STRING_INDICES \
    --curriculum "$CURRICULUM" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
