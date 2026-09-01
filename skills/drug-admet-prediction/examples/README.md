# ADMET Prediction Example

This example computes molecular descriptors and drug-likeness heuristics for three common drugs: caffeine, aspirin, and ibuprofen.

## Files

- `compounds.smi`: SMILES file with 3 compounds
- `compounds_admet.json`: Full output with descriptors, Ro5, Veber, and QED scores

## How to reproduce

Using the drugdisc MCP tool:

```python
mcp_drugdisc_compute_molecular_descriptors(
    smiles_file="skills/drug-admet-prediction/examples/compounds.smi",
    output_file="compounds_admet.json"
)
```

## Results Summary

All 3 molecules passed Lipinski's Rule of Five and Veber criteria:

| Molecule | MW | LogP | TPSA | HBD | `hba` | `hba_lipinski` | Ro5 | Veber | QED |
|----------|------|------|------|-----|-----|-----|-----|-------|-----|
| Caffeine | 194.19 | -1.03 | 61.82 | 0 | 6 | 6 | ✅ | ✅ | 0.54 |
| Aspirin | 180.16 | 1.31 | 63.60 | 1 | 3 | 4 | ✅ | ✅ | 0.55 |
| Ibuprofen | 206.28 | 3.07 | 37.30 | 1 | 1 | 2 | ✅ | ✅ | 0.82 |

Ibuprofen shows the highest QED (drug-likeness) score of 0.82.

**On the two HBA columns.** `hba_lipinski` is the raw N+O count
(`CalcNumLipinskiHBA`) that Ro5 is scored on; it is stable across rdkit versions.
`hba` is `CalcNumHBA`, and it is **rdkit-version-dependent**: for caffeine it returns
**6** on rdkit <= 2025.09.4 and **3** on >= 2025.09.6, where it became a strict SMARTS
acceptor count that excludes amide and pyrrole-type N with delocalised lone pairs.

The table above was generated on **rdkit 2025.09.4** (the version this repo's
`drugdisc-agent` environment pins), so caffeine's `hba` reads 6. On a newer rdkit the
same code reads 3 -- that is the library change, not a discrepancy, and 3 is the
chemically correct acceptor count for caffeine: only its two carbonyl O and its
imidazole `=N-` accept. `hba_lipinski` and the Ro5 verdict are identical on both.

Author: Matthew Cox
Contact: github username <mcox3406>
