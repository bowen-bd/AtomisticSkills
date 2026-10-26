---
name: ml-foundation-potentials
description: Guide for selecting the most appropriate foundation MLIP model based on simulation requirements.
metadata:
  category: [machine-learning, materials, chemistry, drug-discovery]
  venv: [cpu]
---

# Foundation Potentials Selection

<!-- mcp-tools-note -->
> [!NOTE]
> Steps written `server.tool` are MCP tool calls: `base.search_model_registry` is the `search_model_registry`
> tool of the `base` server (`mcp__base__search_model_registry`, or
> `mcp__plugin_atomistic-skills_base__search_model_registry` when installed as a plugin).
> Without a connected server, run the same tools from the shell. Tools named in
> one command share a process, so a model loaded by `load_model` stays loaded:
>
> ```bash
> ${CLAUDE_SKILL_DIR}/../../venv/run cpu python -m src.mcp_server.cli base search_model_registry key=value
> ```

## Goal
Select the appropriate machine learning interatomic potential (MLIP) for a given atomistic simulation task, balancing accuracy, computational cost, and material composition.

## Model Selection Guide

> [!NOTE]
> This list is not exhaustive. For a full list of available pre-trained checkpoints, refer to the `load_model` function documentation for each respective MCP server.

### MatGL Models
**Environment:** `mlip` (MatGL >= 4, PyTorch Geometric)

- **CHGNet-PES-MatPES-PBE-1M-2026.9** (the default CHGNet):
  - Use for PBE-level inorganic materials simulation.
  - Recommended when charge information and magnetic moments are involved (e.g., calculating transition metal valence states).
- **CHGNet-PES-MatPES-r2SCAN-1M-2026.9**:
  - Use for r2SCAN-level inorganic materials simulation, with the same strengths.
- **TensorNet-PES-MatPES-r2SCAN-2025.2** / **TensorNet-PES-MatPES-PBE-2025.2**:
  - Use for r2SCAN- or PBE-level inorganic materials simulation.
  - Smaller and faster than CHGNet, suitable for dynamic simulations (MD, NEB, phonons).

### FAIRCHEM Models
**Environment:** `fairchem`

- **uma-s-1p2p1** (default; UMA checkpoints are gated on Hugging Face: request access at https://huggingface.co/facebook/UMA and set `HF_TOKEN`):
  - Use for organic and inorganic simulations.
  - **Speed:** this is the one checkpoint UMA Nitro serves (fairchem-core >= 2.24, the `fairchem` env), which is what makes UMA viable for dynamics. The gain is largest on small systems — several-fold versus the pre-Nitro paths at tens of atoms — and narrows to about 1.5x by a thousand atoms. Nitro computes in TF32, so energies and forces differ from the FP32 path at the meV level; use `high-precision` when you need FP32.
  - Nitro needs a CUDA GPU; check the `load_model` result, which reports whether Nitro or a fallback is running. Off CUDA, on other UMA checkpoints, or for batched input it silently falls back and you lose most of the speed.
- **uma-s-1p2p1-ft-r2scan** / **uma-s-1p2p1-ft-d4rev**:
  - Single-task 1.2.1 fine-tunes at higher levels of theory, both served by Nitro.
  - `-ft-r2scan` (omat, r2SCAN): materials. Not spin-polarised — do not use for magnetic systems or cohesive energies; add dispersion with `DFTD3Calculator(functional="r2scan")`.
  - `-ft-d4rev` (omol, wB97M-D4rev): molecules and non-covalent interactions. Dispersion is already included; do not add D3.
- **uma-m-1p1**:
  - Use for organic and inorganic simulations with <100 atoms.
- **esen-md-direct-all-omol**:
  - Use for organic ionic relaxation (ground state calculations).

### MACE Models
**Environment:** `mlip`

- **MACE-MH-1**:
  - Latest multi-head foundation model. Use as default for most tasks.
  - `omat_pbe` head (default): General materials, balanced performance.
  - `matpes_r2scan` head: High-accuracy materials simulation.
  - `omol` head: Molecular systems, organic chemistry, organometallics.
  - `spice_wB97M` head: Molecular systems and organic chemistry.
  - `oc20_usemppbe` head: Surface catalysis, adsorbates.
- **MACE-MATPES-r2SCAN-0**:
  - Specialized for r2SCAN-level inorganic systems.
- **MACE-OMAT-0-small**:
  - Small, efficient model for materials.

## Selection Criteria

Prioritize criteria in the following order:

### 0. Check the Local Model Registry (Always First)
Before selecting any foundation model, call `search_model_registry` to check whether a fine-tuned checkpoint already exists for the target chemical system:

```bash
base.search_model_registry(
    chemical_system="Li-Fe-P-O",   # elements of interest
    max_energy_mae=5.0,            # optional accuracy filter (meV/atom)
)
```

- If a match is found **and** `checkpoint_exists = True`, use that model directly — no foundation model selection or fine-tuning is needed.
- If a match is found but `checkpoint_exists = False` (file missing), fall through to the criteria below and plan a new fine-tuning run.
- If no match is found, continue with the criteria below to select the best foundation model.

> [!TIP]
> After completing any fine-tuning, always register the new model with `register_model` so it can be reused in future tasks.

### 1. User Explicit Request
If the user explicitly mentions a model name or framework (e.g., "MACE model", "fine-tuned MACE", "CHGNet", "UMA"), use that model/framework.
- Detect frameworks from keywords like: "MACE", "CHGNet", "TensorNet", "UMA", "ESEN", "FAIRCHEM", "MatGL".

### 2. Calculation Expense
If the simulation involves dynamic or expensive calculations (Molecular Dynamics, NEB, Phonons, Diffusion, Melting Temperature):
- **Prioritize smaller/cheaper models:** structure
  - `TensorNet-MatPES-r2SCAN-v2025.1-PES`
  - `MACE-MATPES-r2SCAN-0` (or MACE small variants)
- **Avoid UMA models** for dynamic simulations due to higher cost, unless the system is very small.
  - Exception: `uma-s-1p2p1` under UMA Nitro (see the FAIRCHEM section) is
    competitive for MD up to a few hundred atoms. Still prefer a smaller model
    for thousands of atoms, where Nitro's advantage disappears.

### 3. System Composition
Consider the chemical elements present in the system:
- **Organic (C, H, N, O, P, S)**:
  - Use **UMA models** or **MACE-MH-1** with `omol` head.
- **Inorganic**:
  - Use **MatGL**, **MACE models**, or **UMA** with `omat` head.
- **For Phase Diagrams & Thermodynamic Stability**:
  - It is highly recommended to use **MatPES-r2SCAN** trained checkpoints (e.g., `CHGNet-MatPES-r2SCAN`, `MACE-MATPES-r2SCAN`). These offer superior energy accuracy for phase stability and bypass messy energy compatibility corrections in GGA (see [mat-dft-mixing-functionals](../mat-dft-mixing-functionals/SKILL.md)).

### 4. Default
For general materials where no specific constraints apply:
- Use **MACE-MH-1** with `omat_pbe` head.

## Performance Benchmark

For detailed inference speed and memory usage of various MLIPs, refer to the dedicated **[ml-mlip-speed](../ml-mlip-speed/SKILL.md)** skill. This skill provides automatic benchmarks to help you choose the most efficient model for your simulation scale.
---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
