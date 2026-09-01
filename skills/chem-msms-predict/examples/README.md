# chem-msms-predict examples

One worked example demonstrating LC-MS/MS spectrum prediction via ICEBERG for a small organic molecule.

## Prerequisites

- `ms-gen` conda environment activated (see `conda-envs/msms-agent/install.sh`)
- ICEBERG checkpoints downloaded to `downloads/`:
  - `downloads/iceberg_dag_gen_msg_best.ckpt` (stage 1: fragment DAG generator)
  - `downloads/iceberg_dag_inten_msg_best.ckpt` (stage 2: intensity predictor)

## Example: 2-Aminoethyl benzoate

**Script**: `predict_smiles.py`
**Molecule**: 2-aminoethyl benzoate — `c1ccccc1C(=O)OCCN`
**Precursor**: `[M+H]+` ≈ 166.087 Da

Runs inference at collision energies 20 eV and 40 eV and writes results to `.agents/test/msms_example/`.

```bash
# Env: ms-gen
python skills/chem-msms-predict/examples/predict_smiles.py \
    --gen_ckpt downloads/iceberg_dag_gen_msg_best.ckpt \
    --inten_ckpt downloads/iceberg_dag_inten_msg_best.ckpt \
    --output_dir .agents/test/msms_example
```

**Outputs**:

| File | Description |
|------|-------------|
| `spectrum.png` | Two-panel stem plot (20 eV + 40 eV) |
| `fragments.json` | Fragment SMILES per peak, sorted by intensity |
| `input_configs.yaml` | Full run parameters for reproducibility |

## Extending to other molecules

Adapt `predict_smiles.py` or call the underlying script directly:

```bash
# Env: ms-gen
python skills/chem-msms-predict/scripts/predict_msms.py \
    --smiles "<your SMILES>" \
    --gen_ckpt downloads/iceberg_dag_gen_msg_best.ckpt \
    --inten_ckpt downloads/iceberg_dag_inten_msg_best.ckpt \
    --collision_energies 20 40 60 \
    --adduct "[M+H]+" \
    --output_dir results/my_compound
```

See [SKILL.md](../SKILL.md) for full parameter reference and constraints.
