#!/bin/bash
# Example: the complete MatterGen fine-tuning workflow on a toy dataset.
# Run it from any directory; its outputs go to the current directory.
# Requires a GPU and a MatterGen checkout (see the skill's prerequisites).

set -e  # Exit on error

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$(dirname "$HERE")")"
RUN="${SKILL_DIR}/../../venv/run"

echo "=== MatterGen Fine-Tuning Example ==="
echo

# Configuration
PROPERTY_NAME="formation_energy"
BASE_MODEL="mattergen_base"
EPOCHS=2  # Use 2 for quick testing, 100+ for real fine-tuning
OUTPUT_DIR="./example_finetuned_model"

# Step 1: Prepare training data
echo "Step 1: Preparing training data..."
"$RUN" mattergen python "${HERE}/make_dummy.py"   # writes example_training_data.json
"$RUN" mattergen python "${SKILL_DIR}/scripts/prepare_training_data.py" \
  --structures-json example_training_data.json \
  --property-name "$PROPERTY_NAME" \
  --output training_data.csv

echo "✓ Training data prepared"
echo

# Step 2: Run fine-tuning
echo "Step 2: Running fine-tuning (${EPOCHS} epochs)..."
"$RUN" mattergen python "${SKILL_DIR}/scripts/run_finetuning.py" \
  --training-data training_data.csv \
  --property-name "$PROPERTY_NAME" \
  --base-model "$BASE_MODEL" \
  --epochs "$EPOCHS" \
  --output-dir "$OUTPUT_DIR"

echo
echo "=== Fine-Tuning Complete ==="
echo "Model saved to: $OUTPUT_DIR"
echo
echo "To use the fine-tuned model:"
echo "  1. Load it in your Python code using MatterGen's API"
echo "  2. Generate structures conditioned on $PROPERTY_NAME"
