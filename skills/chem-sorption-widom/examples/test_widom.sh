#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # repository root, for venv/run
cd $(dirname "$0")

# Run Widom insertion using UMA
export PYTHONPATH=$(dirname $(dirname $(dirname $(dirname "$PWD"))))
"$ROOT/venv/run" fairchem python ../scripts/run_widom.py \
  --structure test_structure_supercell.relaxed.cif \
  --name test_structure \
  --calculator fairchem \
  --model-name uma-s-1p1 \
  --task-name omol \
  --gas CO2 \
  --temperature 298 \
  --num-insertions 5000 \
  --output-dir .

echo "Widom insertion test complete!"
