#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # repository root, for venv/run
cd $(dirname "$0")

# Build supercell based on a minimum interplanar distance of 12.0 Å
"$ROOT/venv/run" cpu python ../scripts/build_supercell.py \
    --structure test_structure.cif \
    --min-plane-dist 12.0 \
    --output-cif test_structure_supercell.cif

echo "Supercell built successfully!"

export PYTHONPATH=$(dirname $(dirname $(dirname $(dirname "$PWD"))))
"$ROOT/venv/run" fairchem python ../scripts/relax_structure.py \
    --structure test_structure_supercell.cif \
    --name test_structure_supercell \
    --calculator fairchem \
    --model-name uma-s-1p1 \
    --task-name omol \
    --steps 50 \
    --output-dir .

echo "Relaxation complete!"
