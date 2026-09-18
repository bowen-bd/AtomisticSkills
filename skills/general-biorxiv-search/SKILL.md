---
name: general-biorxiv-search
description: Search and retrieve preprint metadata from bioRxiv and medRxiv APIs for biological and medical research.
metadata:
  category: [general]
---

# bioRxiv / medRxiv Search

## Goal
Search and retrieve preprint metadata (title, authors, abstract, DOI, category, etc.) from bioRxiv and medRxiv using the official REST API. Supports keyword filtering, subject category filtering, date-range search, and DOI lookup.

## Instructions

### 1. Keyword Search (Recent Papers)
Search last 30 days for papers matching keywords in title/abstract.

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/general-biorxiv-search/scripts/biorxiv_search.py "protein folding" --max_results 5 --output protein_folding.json
```

### 2. Date Range Search
Retrieve preprints posted in a specific date range, with optional keyword/category filter.

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/general-biorxiv-search/scripts/biorxiv_search.py "CRISPR" --start 2025-01-01 --end 2025-06-01 --max_results 10 --output crispr_2025.json
```

### 3. Category Filter
Filter by subject category (use shortcuts or full names).

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/general-biorxiv-search/scripts/biorxiv_search.py --category neuroscience --days 14 --max_results 10
```

**Supported Category Shortcuts:**
- `biochemistry`, `bioinformatics`, `biophysics`
- `cancer` → cancer_biology
- `cell-bio` → cell_biology
- `dev-bio` → developmental_biology
- `ecology`, `epidemiology`, `evolution`
- `genetics`, `genomics`, `immunology`
- `microbiology`, `molecular-bio` → molecular_biology
- `neuroscience`, `pharmacology`, `physiology`
- `plant-bio` → plant_biology
- `synthetic-bio` → synthetic_biology
- `systems-bio` → systems_biology

### 4. Search medRxiv
Use `--server medrxiv` for medical preprints.

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/general-biorxiv-search/scripts/biorxiv_search.py "COVID vaccine efficacy" --server medrxiv --max_results 5 --output covid_vaccine.json
```

### 5. DOI Lookup
Retrieve metadata for a specific preprint by DOI.

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/general-biorxiv-search/scripts/biorxiv_search.py --doi 10.1101/2021.01.01.000000 --output paper.json
```

## Examples

### Recent Neuroscience Papers
```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/general-biorxiv-search/scripts/biorxiv_search.py --category neuroscience --days 7 --max_results 5
```

### Genomics Papers with Keyword Filter
```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/general-biorxiv-search/scripts/biorxiv_search.py "single cell sequencing" --category genomics --start 2024-01-01 --end 2024-12-31 --max_results 10 --output scrna_2024.json
```

## Constraints
- **No official rate limit** specified by bioRxiv API; script includes 0.5s delay per request as courtesy.
- **Pagination**: API returns 30 results per page; script paginates automatically up to `--max_results`.
- **Keyword filtering**: Done client-side (API lacks full-text search); date range must be broad enough to surface relevant papers.
- **Environment**: Requires `base-agent` conda environment.
- **Dependencies**: Uses `requests`.
- **Metadata fields**: DOI, URL, title, authors, abstract, date, category, server, version, license, published journal (if available).

---

**Author:** mlederbauer
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
