#!/bin/bash
# Example: Retrieve crystal structures by material ID from Materials Project
# This demonstrates both single and batch structure retrieval

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # repository root, for venv/run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"

echo "======================================"
echo "Materials Project Structure Retrieval"
echo "======================================"
echo ""

# Example 1: Retrieve a single structure (Silicon)
echo "Example 1: Retrieving Silicon (mp-149)..."
"$ROOT/venv/run" cpu python ../../scripts/get_structure_by_id.py mp-149 \
    --output "${OUTPUT_DIR}/mp-149_Si.cif"
echo ""

# Example 2: Retrieve multiple structures in batch mode
echo "Example 2: Batch retrieval of common materials..."
"$ROOT/venv/run" cpu python ../../scripts/get_structure_by_id.py \
    mp-149 \
    mp-19017 \
    mp-1143 \
    --output_dir "${OUTPUT_DIR}"
echo ""

echo "======================================"
echo "Examples completed!"
echo "Output files saved to: ${OUTPUT_DIR}"
echo "======================================"
