---
name: chem-msms-predict
description: Predict LC-MS/MS (MS2, tandem mass spectra) from SMILES via ICEBERG, a two-stage deep neural network. Outputs predicted m/z vs intensity spectrum, fragment ion SMILES, and a spectrum plot.
metadata:
  category: [chemistry, drug-discovery]
---

# LC-MS/MS Spectrum Prediction

## Goal

Predict the LC-MS/MS (tandem mass) spectrum of a molecule given its SMILES string using ICEBERG — a two-stage GNN that first generates a fragmentation DAG (fragment ions) and then predicts their intensities. Output is a predicted spectrum (m/z, intensity) with optional fragment SMILES assignments per peak.

## When to Use This Skill

- A SMILES string is known and a predicted LC-MS/MS spectrum (m/z vs intensity) is needed.
- Fragment ion assignments (SMILES per peak) are required.
- No reference spectrum exists, or comparison to a predicted spectrum is desired.
- Companion skill `chem-spectrum-matcher` can compare predicted vs experimental spectra.

## When NOT to Use This Skill

- **Experimental spectrum already available** — use it directly; no prediction needed.
- **Only compound name known** — first resolve to SMILES via `drug-db-pubchem`, then call this skill.
- **GC-MS or other MS types** — ICEBERG is trained on LC-MS/MS only; flag a warning before proceeding.
- **Organometallics or MW > 1000** — predictions may be unreliable or fail due to unsupported element types.

## Prerequisites

### 1. Download ICEBERG checkpoints

Download from [coleygroup/ms-pred releases](https://github.com/coleygroup/ms-pred) and place in `downloads/`:

```
downloads/
├── iceberg_dag_gen_msg_best.ckpt     # generator (stage 1)
└── iceberg_dag_inten_msg_best.ckpt   # intensity predictor (stage 2)
```

**Flag error and stop** if either checkpoint is missing.

### 2. Set up the conda environment

```bash
bash conda-envs/msms-agent/install.sh
```

The `ms_pred` Python package is installed from GitHub automatically by the install script.

## Instructions

### Step 1 — Run inference and generate spectrum

```bash
# Venv: venv/mlip
uv run --project venv/mlip python skills/chem-msms-predict/scripts/predict_msms.py \
    --smiles "c1ccccc1C(=O)OCCN" \
    --gen_ckpt downloads/iceberg_dag_gen_msg_best.ckpt \
    --inten_ckpt downloads/iceberg_dag_inten_msg_best.ckpt \
    --collision_energies 20 40 \
    --adduct "[M+H]+" \
    --instrument "Orbitrap" \
    --output_dir results/msms_prediction
```

**Key parameters:**
- `--smiles` — input molecule as SMILES string
- `--gen_ckpt` / `--inten_ckpt` — paths to ICEBERG checkpoints
- `--collision_energies` — one or more collision energies in eV (e.g. `20 40 60`); model was trained on absolute eV values
- `--adduct` — supported adducts: `[M+H]+`, `[M-H]-`, `[M+Na]+`, `[M+NH4]+`, and others from `ms_pred.common.ion2mass`
- `--instrument` — instrument type for intensity prediction (e.g. `"Orbitrap"`, `"QTOF"`)
- `--threshold` — confidence cutoff for DAG fragment generator (default `0.1`; lower = more fragments)
- `--sparse_k` — maximum number of peaks returned (default `100`)
- `--cuda_devices` — GPU device IDs (e.g. `"0"` or `"0,1"`); omit or set to `None` for CPU

**Outputs written to `--output_dir`:**
| File | Description |
|------|-------------|
| `spectrum.png` | Stem plot of predicted spectrum, one panel per collision energy |
| `fragments.json` | JSON list per CE: `{mz, intensity, fragment_smiles}` sorted by intensity |
| `input_configs.yaml` | All run parameters for reproducibility |

### Step 2 — Inspect fragment assignments (optional)

`fragments.json` maps each predicted peak to the fragment ion SMILES responsible for it:

```json
{
  "20": [
    {"mz": 122.0600, "intensity": 1.0, "fragment_smiles": "c1ccccc1C=O"},
    ...
  ]
}
```

Use this to rationalize which bonds fragment at which energy.

### Step 3 — Compare with experimental spectrum (optional)

If an experimental spectrum is available, use the companion skill:

→ [`chem-spectrum-matcher`](../chem-spectrum-matcher/SKILL.md)

## Examples

### 2-Aminoethyl benzoate (`c1ccccc1C(=O)OCCN`)

```bash
# Venv: venv/mlip
uv run --project venv/mlip python skills/chem-msms-predict/examples/predict_smiles.py \
    --gen_ckpt downloads/iceberg_dag_gen_msg_best.ckpt \
    --inten_ckpt downloads/iceberg_dag_inten_msg_best.ckpt \
    --output_dir .agents/test/msms_example
```

Expected output:
- `spectrum.png` — two-panel spectrum (20 eV + 40 eV)
- `fragments.json` — fragment assignments for both energies
- Precursor `[M+H]+` ≈ 166.087 Da

## Constraints

- **Environment**: All scripts require the `ms-gen` conda environment. `ms_pred` is installed automatically from GitHub by `conda-envs/msms-agent/install.sh`.
- **Checkpoints required**: Script raises `FileNotFoundError` if `--gen_ckpt` or `--inten_ckpt` are missing.
- **Collision energy units**: Use absolute eV values. To convert NCE → eV, set `nce=True` in `iceberg_prediction()` directly.
- **Non-binned output only**: This skill uses `binned_out=False` (high-precision m/z). Binned output disables fragment assignment.
- **Single-compound inference**: Provide one SMILES per call. For batch prediction, loop over SMILES and use separate output dirs.
- **Unsupported elements**: Molecules containing metals, lanthanides, or rare main-group elements may fail or produce low-quality predictions.
- **MW limit**: ICEBERG is unreliable for MW > 1000 Da.

## References

- Alberts, M. et al., "Artificial intelligence for context-aware mass spectrometry", *Nature Methods*, 2025. [DOI:10.1038/s41592-025-02658-z](https://doi.org/10.1038/s41592-025-02658-z)
- ICEBERG source code: [github.com/coleygroup/ms-pred](https://github.com/coleygroup/ms-pred)

---

**Author:** Magdalena Lederbauer
**Contact:** [GitHub @mlederbauer](https://github.com/mlederbauer)
