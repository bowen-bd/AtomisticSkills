---
name: chem-msms-predict
description: Predict LC-MS/MS (MS2, tandem mass spectra) from SMILES via ICEBERG, a two-stage deep neural network. Outputs predicted m/z vs intensity spectrum, fragment ion SMILES, and a spectrum plot.
metadata:
  category: [chemistry, drug-discovery]
  venv: [msms]
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

Scripts run in the `msms` environment, created on first use by `venv/run`. It
installs ICEBERG 2.1 (`ms-pred`, pinned to a commit) on a CPU torch build, and
is x86_64 Linux only: DGL publishes no aarch64 wheels.

### Download the ICEBERG 2.1 checkpoints

```bash
${CLAUDE_SKILL_DIR}/../../venv/run msms python ${CLAUDE_SKILL_DIR}/scripts/download_weights.py
```

This fetches the public weights trained on MassSpecGym (`msg_all`, ~80 MB) from
the ms-pred authors:

```
downloads/iceberg_msg_all/
├── gen/best.ckpt           # generator (stage 1)
└── inten_contr/best.ckpt   # intensity predictor (stage 2)
```

Weights trained on NIST are available from the ms-pred authors on proof of a
NIST license. ICEBERG 2.0 checkpoints do not load: 2.1 adds instrument types.
**Flag error and stop** if either checkpoint is missing.

## Instructions

### Step 1 — Run inference and generate spectrum

```bash
${CLAUDE_SKILL_DIR}/../../venv/run msms python ${CLAUDE_SKILL_DIR}/scripts/predict_msms.py \
    --smiles "c1ccccc1C(=O)OCCN" \
    --gen_ckpt downloads/iceberg_msg_all/gen/best.ckpt \
    --inten_ckpt downloads/iceberg_msg_all/inten_contr/best.ckpt \
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
- `--num_workers` — parallel CPU workers (default `0`: serial); inference runs on the CPU

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
${CLAUDE_SKILL_DIR}/../../venv/run msms python ${CLAUDE_SKILL_DIR}/examples/predict_smiles.py \
    --gen_ckpt downloads/iceberg_msg_all/gen/best.ckpt \
    --inten_ckpt downloads/iceberg_msg_all/inten_contr/best.ckpt \
    --output_dir .agents/test/msms_example
```

Expected output:
- `spectrum.png` — two-panel spectrum (20 eV + 40 eV)
- `fragments.json` — fragment assignments for both energies
- Precursor `[M+H]+` ≈ 166.087 Da

## Constraints

- **Environment**: All scripts run in the `msms` environment (`venv/run msms`, x86_64 Linux only), on the CPU.
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
