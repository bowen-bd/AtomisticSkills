---
trigger: always_on
---

# MCP Server and Environment Runtime Rules

AtomisticSkills runs all skill scripts and MCP servers through the unified launcher `venv/run`. The runtime uses three shared `uv` projects under `venv/` plus one pinned project per research stack, with pre-built container images as an automatic fallback when host requirements are not met.

## 1. Environments Overview

| Environment | Primary Packages | GPU | MCP Servers |
| :--- | :--- | :--- | :--- |
| `cpu` | ASE, pymatgen, RDKit, pycalphad, MDAnalysis (no torch) | No | `base`, `atomate2`, `drugdisc`, `smol` |
| `mlip` | PyTorch 2.14.1, MACE-torch 0.3.16, MatGL 4.1.0, nvalchemi-toolkit 0.2.0 | Yes | `mace`, `matgl` |
| `fairchem` | PyTorch 2.13.0, fairchem-core 2.23.0, nvalchemi-toolkit 0.2.0 | Yes | `fairchem` |

### Research Stacks

Pinned to the versions they were verified with, rather than tracking the latest releases:

| Environment | Used by | Platforms |
| :--- | :--- | :--- |
| `adit`, `diffcsp`, `mattergen` | the generative MCP servers and skills | x86_64 natively (CUDA 12.6 or 13 by driver); aarch64 through the `generative` image |
| `msms` | `chem-msms-predict` (ICEBERG 2.1, CPU) | x86_64 only |
| `reactot` | `chem-react-ot` | x86_64 and aarch64 |
| `scd` | `ml-property-predict-scd` | x86_64 only |

### Why Three Shared Environments?
1. **Package conflicts**: `mace-torch` pins `e3nn==0.4.4`, whereas `fairchem-core` requires `e3nn>=0.5`.
2. **PyTorch versions**: `fairchem-core` 2.23 requires `torch~=2.13`, while `mlip` runs PyTorch 2.14.1.
3. **NumPy constraints**: `nvalchemi-toolkit` 0.2 requires `numpy<2.4`, so GPU environments use NumPy 2.3.5 while `cpu` uses NumPy 2.5+.

### Optional Extras
Environments support optional extras specified as `<venv>+<extra>` (e.g., `cpu+openmm`, `cpu+docking`):
- `openmm`: OpenMM and PDBFixer (requires glibc ≥ 2.34).
- `pymol`: PyMOL open-source (x86_64 only).
- `docking`: AutoDock Vina (builds against Boost on aarch64).
- `void`: VOID guest docking (installed from git).
- `transport`: AMSET and BoltzTraP2 (requires git and cmake).

## 2. Server Runtime Mapping

| MCP Server | venv | Image | GPU | Description |
| :--- | :--- | :--- | :--- | :--- |
| `base` | `cpu` | `cpu` | No | Materials Project query, structure manipulation, literature |
| `atomate2` | `cpu` | `cpu` | No | Atomate2 and Jobflow calculation workflows |
| `drugdisc` | `cpu` | `cpu` | No | Small-molecule standardization, descriptors, fingerprints |
| `smol` | `cpu` | `cpu` | No | Cluster expansion and Monte Carlo simulations |
| `mace` | `mlip` | `mlip` | Yes | MACE foundation models (relax, MD, features) |
| `matgl` | `mlip` | `mlip` | Yes | MatGL models (CHGNet, TensorNet, M3GNet) |
| `fairchem` | `fairchem` | `fairchem` | Yes | FairChem models (UMA, eSEN) |
| `adit` | `adit` | `generative` | Yes | ADiT all-atom diffusion transformer |
| `diffcsp` | `diffcsp` | `generative` | Yes | DiffCSP++ crystal structure generation |
| `mattergen` | `mattergen` | `generative` | Yes | MatterGen generative diffusion model |

The generative servers run from their uv projects on x86_64 hosts with glibc ≥ 2.32 (PyG's wheels need it); elsewhere `venv/run` uses the `generative` container image (linux/amd64 and linux/arm64; on arm64 it compiles the PyG extensions from source, since PyG publishes no aarch64 wheels).

## 3. No Conda

Since 2.0.0 nothing runs from conda. Stacks that cannot share an environment get their own pinned uv project (the research stacks above); host builds use a uv extra (`mlip+lammps` provides what `mat-lammps-md` needs to compile LAMMPS against the `mlip` environment, `fairchem+lammps` the LAMMPS wheel and `fairchem-lammps`).

## 4. Runtime Selection and Launcher Backend

All skills and servers execute through the `venv/run` launcher:
- `venv/run <venv>[+<extra>] <command> [args...]`
- `venv/run --server <name>`
- `venv/run --setup` (prepares environments)
- `venv/run --doctor` (checks runtime status and diagnostics)

The runtime backend is controlled by `ATOMISTIC_RUNTIME` (or `~/.config/atomistic_skills.yaml`):
- `auto` (default): Uses native `uv` if the machine is Linux x86_64/aarch64 with compatible glibc (per `venv/platforms.tsv`) and a C compiler (`gcc`). Otherwise, automatically falls back to container images (`ghcr.io/learningmatter-mit/atomisticskills-<name>:2.0.0`).
- `uv`: Enforces host `uv` execution.
- `docker`, `podman`, `apptainer`, `singularity`: Enforces container execution with host paths mounted at identical locations.

## 5. Shell CLI Fallback for MCP Tools

When an MCP server is not connected via stdio, any MCP tool can be executed directly from the shell:
```bash
venv/run <venv> python -m src.mcp_server.cli <server> <tool> key=value ...
```
- Pass `--list` to view all available tools for a server.
- Multiple tool calls chained in one command share process memory (e.g., `load_model` followed by `relax_structure`).
