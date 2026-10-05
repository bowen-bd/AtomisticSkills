---
name: drug-ligand-prep
description: Prepare small-molecule ligands for docking and analysis via optional state enumeration, 3D conformer generation, MMFF/UFF minimization, and export to SDF + AutoDock PDBQT.
metadata:
  category: [drug-discovery]
  venv: [cpu]
---

# Ligand Preparation

<!-- mcp-tools-note -->
> [!NOTE]
> Steps written `server.tool` are MCP tool calls: `drugdisc.convert_to_pdbqt` is the `convert_to_pdbqt`
> tool of the `drugdisc` server (`mcp__drugdisc__convert_to_pdbqt`, or
> `mcp__plugin_atomistic-skills_drugdisc__convert_to_pdbqt` when installed as a plugin).
> Without a connected server, run the same tools from the shell. Tools named in
> one command share a process, so a model loaded by `load_model` stays loaded:
>
> ```bash
> ${CLAUDE_SKILL_DIR}/../../venv/run cpu python -m src.mcp_server.cli drugdisc convert_to_pdbqt key=value
> ```

## Goal
To prepare small-molecule ligands for molecular docking and downstream analysis by:
1) optionally enumerating relevant ligand ionization states and tautomers,
2) generating 3D conformers with RDKit ETKDG (via MCP),
3) minimizing with MMFF94/UFF (via MCP),
4) exporting a docking-ready **PDBQT** (AutoDock-Vina) and an optimized **SDF** (via MCP).

This skill combines script-based state enumeration with MCP-based 3D generation to ensure reproducibility.

## Instructions

### 1. Enumerate States (Optional Batch Processing)

Use the script to process SMILES/SDF files and enumerate protonation/tautomer states. This outputs 2D SDFs.

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/prepare_ligand.py \
  --smiles_file ligands.smi \
  --enumerate_protomers \
  --output_dir ligand_states/
```

### 2. Generate 3D Conformer and PDBQT (using MCP)

Use the `drugdisc.convert_to_pdbqt` tool to generate the final 3D docking input.

**From a single SMILES:**
```bash
drugdisc.convert_to_pdbqt(
    input_data="CC(=O)Oc1ccccc1C(=O)O",
    input_type="smiles",
    output_path="aspirin.pdbqt",
    num_confs=50
)
```

**From an SDF (e.g. output of Step 1):**
```bash
drugdisc.convert_to_pdbqt(
    input_data="ligand_states/ligand_001.sdf",
    input_type="sdf",
    output_path="ligand_001.pdbqt",
    num_confs=20
)
```

## Examples

### Prepare Ibuprofen

1. Enumerate inputs (if needed):
   ```bash
   ${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/prepare_ligand.py \
     --smiles "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O" \
     --name ibuprofen \
     --output_dir prep_stages/
   ```

2. Generate PDBQT:
   ```bash
   drugdisc.convert_to_pdbqt(
       input_data="prep_stages/ibuprofen.sdf",
       input_type="sdf",
       output_path="prep_stages/ibuprofen.pdbqt",
       num_confs=50
   )
   ```

## Constraints

* **Environment**: Requires `drugdisc-agent`.
* **3D/PDBQT**: Delegated to `drugdisc.convert_to_pdbqt` (Meeko/RDKit).
* **State Enumeration**: The script handles batch enumeration of protonation/tautomer states, but 3D generation is done by the MCP tool.
---

**Author:** Matthew Cox
**Contact:** [GitHub @mcox3406](https://github.com/mcox3406)
