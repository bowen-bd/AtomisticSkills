---
name: drug-admet-prediction
description: Compute RDKit physicochemical descriptors and rule-based drug-likeness heuristics (Ro5, Veber, QED) from SMILES.
metadata:
  category: [drug-discovery]
  venv: [cpu]
---

# admet-prediction

<!-- mcp-tools-note -->
> [!NOTE]
> Steps written `server.tool` are MCP tool calls: `drugdisc.compute_molecular_descriptors` is the `compute_molecular_descriptors`
> tool of the `drugdisc` server (`mcp__drugdisc__compute_molecular_descriptors`, or
> `mcp__plugin_atomistic-skills_drugdisc__compute_molecular_descriptors` when installed as a plugin).
> Without a connected server, run the same tools from the shell. Tools named in
> one command share a process, so a model loaded by `load_model` stays loaded:
>
> ```bash
> ${CLAUDE_SKILL_DIR}/../../venv/run cpu python -m src.mcp_server.cli drugdisc compute_molecular_descriptors key=value
> ```

## Goal
Compute ADMET-relevant **physicochemical descriptors** and **rule-based drug-likeness heuristics** from SMILES strings using RDKit.

This skill reports:
- Core descriptors: molecular weight (average and exact), Wildman-Crippen cLogP, TPSA, HBD/HBA, rotatable bonds, ring counts, aromatic rings, heavy atoms, fractionCSP3, molar refractivity.
- Heuristics:
  - **Lipinski Rule of Five (Ro5)** compliance (≤ 1 violation) as a permeability/absorption triage heuristic.
  - **Veber** oral bioavailability heuristic (RB ≤ 10 and TPSA ≤ 140 Å²; plus reporting the alternative HBD+HBA ≤ 12 condition).
  - **QED** (Quantitative Estimate of Drug-likeness) score.

> Note: This does **not** predict experimental ADMET endpoints (e.g., clearance, CYP inhibition, hERG, Ames, etc.). It is an early-stage physchem/heuristics screen.

## Instructions

The drugdisc MCP server provides a `compute_molecular_descriptors` tool that can be called directly:

**Single molecule analysis:**
```bash
drugdisc.compute_molecular_descriptors(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    output_file="aspirin_admet.json"
)
```

**Batch analysis from a SMILES file:**
```bash
drugdisc.compute_molecular_descriptors(
    smiles_file="${CLAUDE_SKILL_DIR}/examples/compounds.smi",
    output_file="batch_admet.json"
)
```

**With S/P-inclusive TPSA:**
```bash
drugdisc.compute_molecular_descriptors(
    smiles="OC(=O)P(=O)(O)O",
    include_sandp_tpsa=True,
    output_file="foscarnet_admet.json"
)
```

## Examples

Example `compounds.smi`:
```text
CN1C=NC2=C1C(=O)N(C(=O)N2C)C	caffeine
CC(=O)Oc1ccccc1C(=O)O	aspirin
CC(C)Cc1ccc(cc1)C(C)C(=O)O	ibuprofen
```

Run:
```bash
drugdisc.compute_molecular_descriptors(
    smiles_file="${CLAUDE_SKILL_DIR}/examples/compounds.smi",
    output_file="drug_admet.json"
)
```

## Constraints
- **MCP Server**: Requires `drugdisc` MCP server
- **Dependencies**: RDKit (Chem, Descriptors, Lipinski, Crippen, QED)
- **Scope**: Outputs physchem descriptors + rule-based heuristics only; not ML/experimental ADMET prediction
- **Ro5 interpretation**: A "pass" is defined here as **≤ 1 violation** (common industry convention)
- **Veber interpretation**: Primary check uses **TPSA ≤ 140 Å² and rotatable bonds ≤ 10**, and additionally reports the alternative **(HBD + HBA ≤ 12)** criterion
- **Standardization**: If SMILES contains multiple fragments (e.g., salts, "."), results are reported but flagged with a warning; consider desalting/neutralization upstream for library triage
- **TPSA option**: By default, TPSA uses RDKit's default behavior (no S/P); `include_sandp_tpsa=True` includes S/P contributions
- **Two HBA definitions, both reported**: `hba` is `rdMolDescriptors.CalcNumHBA`, the
  strict SMARTS acceptor count that excludes amide and pyrrole-type N with delocalised
  lone pairs (caffeine = 3: two carbonyl O plus one imidazole `=N-`). `hba_lipinski` is
  `rdMolDescriptors.CalcNumLipinskiHBA`, the raw N+O count Lipinski 1997 specified
  (caffeine = 6). **Ro5 is scored on `hba_lipinski`**, per the original paper.
  Do not call the `Lipinski.NumHAcceptors` alias: its meaning changed between rdkit
  2025.09.4 and 2025.09.6 (caffeine 6 -> 3), so results computed through it are not
  comparable across environments. `hba` inherits that library change and will read 6
  on rdkit <= 2025.09.4 and 3 on >= 2025.09.6; `hba_lipinski` is stable on both.
---

**Author:** Matthew Cox
**Contact:** [GitHub @mcox3406](https://github.com/mcox3406)
