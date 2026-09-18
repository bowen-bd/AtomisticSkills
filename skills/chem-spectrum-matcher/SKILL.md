---
name: chem-spectrum-matcher
description: Match an experimental spectrum (1H NMR, 13C NMR, IR) against predicted or database reference spectra for candidate ranking and structure confirmation. Supports local catalog lookup, public database fallback, and pluggable similarity metrics.
metadata:
  category: [chemistry, drug-discovery]
---

# Spectrum Matcher

## Goal
To retrieve or generate reference spectra for a set of candidate molecules and rank them by similarity to an experimental query spectrum. The skill abstracts a common three-component pattern:

1. **Prediction** — generate reference spectra from structure (SMILES) via empirical predictors or QM.
2. **Reference DB** — cache computed spectra locally; fall back to public databases for known compounds.
3. **Similarity metric** — score each candidate against the query and rank.

This pattern applies to any spectral modality: 1H NMR, 13C NMR, IR, mass spectrometry, UV-Vis, Raman. The concrete implementation here covers **1H NMR** and **IR**, with the NMR path fully implemented and IR sketched for extension.

## When to Use This Skill

- Confirm a proposed structure against an experimental spectrum.
- Screen a shortlist of candidates and rank by spectral similarity.
- Avoid re-running expensive predictions by retrieving cached spectra from the local catalog.
- Fetch experimental reference spectra from public databases (NMRShiftDB2, NIST WebBook) before committing to a prediction run.

## When NOT to Use This Skill

- **Unknown structure elucidation from scratch** — this skill requires a candidate list. For open-ended structure identification, use `general-query-literature-database` first.
- **Mixture deconvolution** — use `chem-nmr-analysis` (Wasserstein deconvolution) for quantifying component ratios.
- **13C, 19F, 31P NMR prediction** — `chem-nmr-predict` (SPINUS) covers 1H only. Extension needed.
- **Mass spectrometry** — not yet implemented. Scaffold is in place; add a predictor and similarity metric.

---

## Architecture

```
SMILES  ──► [Predictor]  ──► predicted spectrum (.xy / .jdx)
                                        │
                                        ▼
                              [Local Catalog]  ◄──  register_spectrum.py
                                        │
        [Public DB fallback]  ──────────┤  (NMRShiftDB2, NIST WebBook)
                                        │
                                        ▼
            Experimental query  ──► [match_spectrum.py]  ──► ranked candidates
```

### Modality–Predictor–Metric table

| Modality | Predictor skill | Public DB | Similarity metric |
|---|---|---|---|
| 1H NMR | `chem-nmr-predict` (SPINUS + nmrsim) | NMRShiftDB2 | L2 / Wasserstein |
| IR | `chem-db-spectra` (NIST) or ORCA DFT | NIST WebBook | Cosine |
| 13C NMR | *(not yet implemented)* | NMRShiftDB2 | L2 |
| Mass spec | *(not yet implemented)* | NIST WebBook | Dot product |

---

## Workflow

### Step 1 — Prepare Candidate Reference Spectra

#### Option A: Retrieve from catalog or public DB (fast, no prediction)

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-spectrum-matcher/scripts/match_spectrum.py \
  --query experimental_spectrum.xy \
  --smiles "CCO" \
  --names "ethanol" \
  --modality nmr_1h \
  --catalog_dir research/spectrum_catalog/ \
  --output_dir <research_dir>/spectrum_match/ \
  --fallback_public_db
```

#### Option B: Predict first, then match

Run the appropriate predictor for the modality, then register outputs into the catalog, then match.

**1H NMR:**
```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-nmr-predict/scripts/predict_nmr.py \
  --smiles "CCO" \
  --names "ethanol" \
  --field_mhz 400 \
  --output_dir <research_dir>/nmr_predictions/

# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-spectrum-matcher/scripts/register_spectrum.py \
  --source_dir <research_dir>/nmr_predictions/ \
  --modality nmr_1h \
  --catalog_dir research/spectrum_catalog/
```

**IR (from NIST WebBook):**
```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-db-spectra/scripts/query_spectra.py \
  C10H18O <research_dir>/ir_references/ --type IR

# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-spectrum-matcher/scripts/register_spectrum.py \
  --source_dir <research_dir>/ir_references/ \
  --modality ir \
  --catalog_dir research/spectrum_catalog/
```

**IR (QM-backed, high accuracy):**
```bash
# Venv: venv/cpu
# Run ORCA frequency calculation → extract IR spectrum → register
# See conda-envs/orca-agent/ for ORCA setup.
# After ORCA run, convert output with src/utils/dft/orca_utils.py
# then call register_spectrum.py --modality ir
```

### Step 2 — Match and Rank

`match_spectrum.py` retrieves reference spectra (catalog → public DB fallback) and computes similarity scores between the query and each candidate.

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-spectrum-matcher/scripts/match_spectrum.py \
  --query experimental_spectrum.xy \
  --smiles "CCO" \
  --names "ethanol" \
  --modality nmr_1h \
  --metric l2 \
  --catalog_dir research/spectrum_catalog/ \
  --output_dir <research_dir>/spectrum_match/ \
  --plot
```

**Arguments:**
- `--query`: experimental spectrum file (two-column, ppm/wavenumber vs intensity; `.xy`, `.csv`, `.jdx`).
- `--smiles`: candidate SMILES strings.
- `--names`: human-readable labels matching SMILES order.
- `--modality`: `nmr_1h`, `nmr_13c`, `ir`. Controls which catalog partition and public DB to query.
- `--metric`: similarity metric — `l2` (default), `cosine`, `wasserstein`. Choose based on modality (see table above).
- `--catalog_dir`: local spectrum catalog directory.
- `--fallback_public_db`: query NMRShiftDB2 or NIST WebBook for any candidate not in catalog.
- `--field_mhz`: spectrometer field (NMR only, default 400). Must match experimental spectrum.
- `--plot`: emit overlay plot (`match_plot.png`) with query and top-3 candidates.
- `--output_dir`: directory for outputs.

**Outputs:**
- `match_results.json` — ranked candidates with similarity scores, source (catalog/public_db/predicted), and spectrum paths.
- `match_plot.png` — overlay of query vs ranked references (if `--plot`).
- `input_configs.yaml` — all parameters for reproducibility.

### Step 3 — Interpret Results

Read `match_results.json`. Candidates are ranked by descending similarity score (1.0 = perfect match, 0.0 = no overlap).

**If/Then rules:**

| Score (L2 / cosine) | Interpretation | Agent action |
|---|---|---|
| > 0.90 | Strong match | Report top candidate with confidence. |
| 0.70–0.90 | Plausible match | Report with caveat; check overlay plot for unmatched peaks. |
| < 0.70 | Poor match | Likely wrong candidate or missing structure. Expand candidate list or re-examine experimental spectrum. |

After reading scores, the agent must:
1. **Inspect `match_plot.png`** — verify visual agreement, check for systematic shifts.
2. **Cross-check with signal table** (NMR) — compare predicted multiplicity/coupling to experimental assignments.
3. **If top candidate has score < 0.70** — consider running QM-backed prediction (ORCA IR, or higher-level NMR) rather than empirical.

---

## Failure Modes

| Failure | Symptom | Agent action |
|---|---|---|
| Catalog miss, no public DB hit | `match_results.json` candidate marked `missed` | Run appropriate predictor then `register_spectrum.py`. |
| Ppm/wavenumber axis mismatch | Similarity scores all near 0 | Query and reference use different x-axis. Check `--field_mhz` or unit convention. |
| SMILES canonicalization fails | RDKit error | SMILES invalid. Verify with RDKit before retry. |
| NMRShiftDB2 / NIST timeout | HTTP error during public DB query | Retry once; if persistent, disable `--fallback_public_db` and predict locally. |
| ORCA IR prediction unavailable | `ORCA_BINARY_PATH` not set | Set env var per `conda-envs/orca-agent/README.md` or fall back to NIST WebBook IR. |

---

## Relationship to Other Skills

```
drug-db-pubchem          → resolve compound name to SMILES
chem-nmr-predict         → 1H NMR prediction (SPINUS + nmrsim)
chem-db-spectra          → experimental IR/MS from NIST WebBook
chem-nmr-analysis        → mixture deconvolution (Wasserstein)
chem-spectrum-matcher    → this skill: catalog + retrieval + similarity ranking
```

---

## Environment

**Primary (NMR matching):**
```bash
# Venv: venv/cpu
```
Required packages: `numpy`, `scipy`, `rdkit`, `requests`, `matplotlib`.

**IR prediction via QM (optional):**
```bash
# Venv: venv/cpu
```
Requires `ORCA_BINARY_PATH` environment variable. See `conda-envs/orca-agent/README.md`.

---

## Constraints
- **Catalog format**: `catalog.json` keyed by `(canonical_smiles, modality)`. Do not edit manually.
- **Spectrum format**: `.xy` files are two-column tab-separated (x-axis descending, intensity). `.jdx` files are parsed via the `jcamp` package.
- **Modality isolation**: NMR and IR spectra are stored in separate catalog partitions; cross-modality lookup is not supported.
- **Field strength**: NMR catalog entries store the field (MHz) used during prediction. Retrieval warns if query field differs.
- **Stereochemistry**: Diastereomers stored separately by canonical SMILES. Enantiomers produce identical achiral NMR/IR spectra but are stored separately for traceability.

---

## References
- Steinbeck, C. et al., "NMRShiftDB — constructing a free chemical information system with open-source components", *J. Chem. Inf. Comput. Sci.*, 2003. [DOI](https://doi.org/10.1021/ci0341363)
- Linstrom, P.J. and Mallard, W.G., Eds., *NIST Chemistry WebBook*, NIST Standard Reference Database 69. [URL](https://webbook.nist.gov)
- Banfi, D. & Patiny, L., "www.nmrdb.org: Resurrecting and processing NMR spectra on-line", *Chimia*, 2008.
- Landrum, G. et al., RDKit: Open-Source Cheminformatics. [URL](https://www.rdkit.org)

---

**Author:** Magdalena Lederbauer
**Contact:** [GitHub @mlederbauer](https://github.com/mlederbauer)
