# chem-msms-predict examples

One worked example demonstrating LC-MS/MS spectrum prediction via ICEBERG for a small organic molecule.

## Prerequisites

- The `msms` environment (`venv/run msms ...`; x86_64 Linux), created on first use.
- ICEBERG 2.1 checkpoints in `downloads/iceberg_msg_all/`, fetched by
  `venv/run msms python skills/chem-msms-predict/scripts/download_weights.py`:
  - `gen/best.ckpt` (stage 1: fragment DAG generator)
  - `inten_contr/best.ckpt` (stage 2: intensity predictor)

## Example: 2-Aminoethyl benzoate

**Script**: `predict_smiles.py`
**Molecule**: 2-aminoethyl benzoate — `c1ccccc1C(=O)OCCN`
**Precursor**: `[M+H]+` ≈ 166.087 Da

Runs inference at collision energies 20 eV and 40 eV and writes results to `.agents/test/msms_example/`.

```bash
venv/run msms python skills/chem-msms-predict/examples/predict_smiles.py \
    --gen_ckpt downloads/iceberg_msg_all/gen/best.ckpt \
    --inten_ckpt downloads/iceberg_msg_all/inten_contr/best.ckpt \
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
venv/run msms python skills/chem-msms-predict/scripts/predict_msms.py \
    --smiles "<your SMILES>" \
    --gen_ckpt downloads/iceberg_msg_all/gen/best.ckpt \
    --inten_ckpt downloads/iceberg_msg_all/inten_contr/best.ckpt \
    --collision_energies 20 40 60 \
    --adduct "[M+H]+" \
    --output_dir results/my_compound
```

See [SKILL.md](../SKILL.md) for full parameter reference and constraints.
