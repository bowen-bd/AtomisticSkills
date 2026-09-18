---
name: chem-hazard-toxicity
description: Extract explicit safety warnings, GHS classifications, LD50 toxicity profiles, and acute oral toxicity triage from PubChem PUG VIEW.
metadata:
  category: [chemistry, drug-discovery]
---

# Chemical Hazard and Toxicity Profiling

## Goal
To programmatically extract critical safety information from the PubChem PUG-VIEW API. This skill pulls GHS Classifications, Hazard Classes, and Toxicological properties (like LD50/LC50 experimental animal records) for a given compound based on its CID, and performs GHS Acute Toxicity Category assignment and hazard statement consistency analysis.

## Instructions

### 1. Extract Safety Profile by CID
Provide the precise CID of the molecule to query safety metadata.

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-hazard-toxicity/scripts/get_safety_data.py \
  --cid 2519 \
  --outdir research/caffeine_safety \
  --output safety_caffeine.json
```

### 2. GHS Acute Oral Toxicity Triage Helper
Use `--triage` to extract consensus GHS statements ($\ge 50\%$), lowest rat oral LD50, GHS acute oral category (1–5 or `unclassified`), and check oral hazard code consistency:

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/chem-hazard-toxicity/scripts/get_safety_data.py \
  --cid 2519 \
  --triage \
  --outdir research/caffeine_safety \
  --output triage_caffeine.json
```

### Python API Integration
Import helper functions directly into python scripts:

```python
from scripts.get_safety_data import (
    extract_consensus_ghs_codes,
    extract_lowest_rat_oral_ld50,
    assign_acute_oral_category,
    check_oral_code_consistency,
    profile_compound,
)

profile = profile_compound(2519, threshold_percent=50.0)
# Returns dict with cid, consensus_ghs_codes, oral_rat_ld50_mg_kg, oral_rat_ld50_evidence, acute_oral_category, ghs_oral_code_consistent
```

> **GHS Category & Code Consistency Rule**:
> - Category 1 ($\le 5\text{ mg/kg}$) & Category 2 ($5 < \text{LD}_{50} \le 50\text{ mg/kg}$) $\rightarrow$ expected GHS oral code `H300` (*Fatal if swallowed*).
> - Category 3 ($50 < \text{LD}_{50} \le 300\text{ mg/kg}$) $\rightarrow$ `H301` (*Toxic if swallowed*).
> - Category 4 ($300 < \text{LD}_{50} \le 2000\text{ mg/kg}$) $\rightarrow$ `H302` (*Harmful if swallowed*).
> - Category 5 ($2000 < \text{LD}_{50} \le 5000\text{ mg/kg}$) $\rightarrow$ `H303` (*May be harmful if swallowed*).
> - `unclassified` ($> 5000\text{ mg/kg}$) $\rightarrow$ empty set (`{}`).

## Constraints
- **Data Availability**: Relies on experimental or reported data listed in PubChem.
- **Network Limits**: Automatically handles standard `HTTP 503/429` rate limiting via exponential backoff.

---
---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
