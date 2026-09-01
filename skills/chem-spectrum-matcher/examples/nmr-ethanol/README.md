# Example: 1H NMR Matching — Ethanol

This example demonstrates the full predict → register → match cycle for ethanol (CCO, $C_2H_6O$). Ethanol has two distinct 1H signals: CH₃ triplet (~1.17 ppm) and CH₂ quartet (~3.69 ppm), making it a clean minimal test case.

The same workflow applies to IR: replace `--modality nmr_1h` with `--modality ir` and supply IR `.jdx` files from `chem-db-spectra`.

---

## Step 1 — Predict 1H NMR Reference Spectrum

```bash
# Env: nmr-agent
python skills/chem-nmr-predict/scripts/predict_nmr.py \
  --smiles "CCO" \
  --names "ethanol" \
  --field_mhz 400 \
  --output_dir research/nmr_predictions/
```

Outputs in `research/nmr_predictions/`:
- `ethanol.xy`, `ethanol_signals.csv`
- `predictions.json`

## Step 2 — Register into Local Catalog

```bash
# Env: nmr-agent
python skills/chem-spectrum-matcher/scripts/register_spectrum.py \
  --source_dir research/nmr_predictions/ \
  --modality nmr_1h \
  --catalog_dir research/spectrum_catalog/
```

Expected output:
```
  Registered ethanol (CCO) [nmr_1h]
Done: 1 registered, 0 skipped.
```

## Step 3 — Match an Experimental Spectrum

Replace `<query>.xy` with a real two-column spectrum file (ppm vs intensity). For a self-consistency test, use the predicted spectrum itself — expect score ≈ 1.0.

```bash
# Env: nmr-agent
python skills/chem-spectrum-matcher/scripts/match_spectrum.py \
  --query <query>.xy \
  --smiles "CCO" \
  --names "ethanol" \
  --modality nmr_1h \
  --metric l2 \
  --catalog_dir research/spectrum_catalog/ \
  --output_dir research/spectrum_match/ \
  --plot
```

### Expected `match_results.json`

```json
{
  "query": "experimental.xy",
  "modality": "nmr_1h",
  "metric": "l2",
  "candidates": [
    {
      "name": "ethanol",
      "smiles": "CCO",
      "canonical_smiles": "CCO",
      "score": 0.951,
      "source": "local_catalog",
      "spectrum_path": "research/spectrum_match/ethanol_ref.xy"
    }
  ]
}
```

Score > 0.90 → strong match. Agent reports ethanol as the identified compound.

---

## Literature Validation

**Ethanol 1H NMR** (400 MHz, CDCl₃): δ 3.69 (2H, q, J = 7.0 Hz, CH₂), 2.61 (1H, s, OH), 1.17 (3H, t, J = 7.0 Hz, CH₃).

SPINUS correctly predicts the CH₂/CH₃ chemical shift separation and the triplet/quartet splitting pattern.

---

## Extending to IR

To match against IR instead of NMR, fetch references from NIST WebBook and register them:

```bash
# Env: base-agent
python skills/chem-db-spectra/scripts/query_spectra.py C2H6O research/ir_refs/ --type IR

# Env: nmr-agent  (or base-agent)
python skills/chem-spectrum-matcher/scripts/register_spectrum.py \
  --source_dir research/ir_refs/ \
  --modality ir \
  --smiles "CCO" \
  --names "ethanol" \
  --catalog_dir research/spectrum_catalog/

python skills/chem-spectrum-matcher/scripts/match_spectrum.py \
  --query experimental_ir.jdx \
  --smiles "CCO" \
  --names "ethanol" \
  --modality ir \
  --metric cosine \
  --catalog_dir research/spectrum_catalog/ \
  --output_dir research/ir_match/ \
  --plot
```
