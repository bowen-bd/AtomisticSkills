---
name: chem-react-ot
description: Generate transition state structures for chemical reactions using React-OT.
metadata:
  category: [chemistry]
  venv: [reactot]
---

# `chem-react-ot` — React-OT Transition State Generation

## Goal

Generate transition state (TS) structures given reactant and product structures using the React-OT model (Optimal Transport). React-OT is a generative model that predicts TS geometries directly without requiring an initial guess path (like NEB).

**Category:** `chemistry`
**Environment:** `react-ot-agent`

## Key Features

- **Generative TS Prediction:** Predicts 3D transition state structures from 3D reactants and products.
- **Fast Inference:** Uses an ODE solver for generation, typically much faster than DFT-based NEB.
- **No Path Guess Required:** Directly generates the TS structure.

## Usage

### 1. Fetch React-OT and the model

React-OT runs in its own uv environment, `reactot` (torch 2.2.1, created on first
use). Its source is not a package: fetch the pinned version, with the fixes it
needs, once, then download the pre-trained weights:

```bash
${CLAUDE_SKILL_DIR}/../../venv/run reactot python ${CLAUDE_SKILL_DIR}/scripts/setup_react_ot.py
${CLAUDE_SKILL_DIR}/../../venv/run reactot python ${CLAUDE_SKILL_DIR}/scripts/download_models.py
```

The source goes to `~/.cache/atomisticskills/react-ot` (or `$REACT_OT_DIR`).

The checkpoint is saved to `~/.cache/react-ot/checkpoints/sb-pretrained.ckpt`.

### 2. Generate Transition State

Run the generation script with reactant and product files (xyz, cif, pdb, etc. - anything ASE reads).

```bash
${CLAUDE_SKILL_DIR}/../../venv/run reactot python ${CLAUDE_SKILL_DIR}/scripts/generate_ts.py \
    --reactants reactant.xyz \
    --products product.xyz \
    --output_dir results/ts_search
```

**Arguments:**

- `--reactants`: Path to reactant structure file(s). Can be a single file with multiple molecules or a list of files.
- `--products`: Path to product structure file(s).
- `--output_dir`: Directory to save the generated TS structure (`ts_generated.xyz`) and trajectory (`generation_traj.xyz`).
- `--nfe`: Number of function evaluations for the ODE solver (default: 10). Higher values might be more accurate but slower.
- `--checkpoint`: Path to custom model checkpoint (optional, defaults to downloaded one).

## Example

```bash
${CLAUDE_SKILL_DIR}/../../venv/run reactot python ${CLAUDE_SKILL_DIR}/scripts/generate_ts.py \
    --reactants ${CLAUDE_SKILL_DIR}/examples/oxadiazole_isomerization/reactant.xyz \
    --products ${CLAUDE_SKILL_DIR}/examples/oxadiazole_isomerization/product.xyz \
    --output_dir ${CLAUDE_SKILL_DIR}/examples/oxadiazole_isomerization/output
```

## Constraints

- **Environment**: `reactot` (x86_64: CUDA 12.1 build of torch 2.2.1, driver >= 525; aarch64: CPU). On aarch64 the PyG extensions are compiled on first use, which needs a C++ compiler.
- **Input Format**: Reactant and product structures must be in any format readable by ASE (XYZ, CIF, PDB, etc.).
- **Atom Ordering**: Reactant and product structures must have the same number of atoms with consistent atom ordering.
- **Model Checkpoint**: The pre-trained checkpoint must be downloaded before first use (see step 1).

## References

- [React-OT GitHub](https://github.com/deepprinciple/react-ot)
- Duan, C., Liu, G.-H., Du, Y. et al., "Optimal transport for generating transition states in chemical reactions", *Nature Machine Intelligence*, 2025. [DOI](https://doi.org/10.1038/s42256-025-01010-0)

---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
