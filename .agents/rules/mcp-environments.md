---
trigger: always_on
---

# MCP Server and Environment Runtime Rules

AtomisticSkills runs all skill scripts and MCP servers through the unified launcher `venv/run`. The runtime uses three primary `uv` projects under `venv/`, with pre-built container images as an automatic fallback when host requirements are not met.

## 1. Environments Overview

| Environment | Primary Packages | GPU | MCP Servers |
| :--- | :--- | :--- | :--- |
| `cpu` | ASE, pymatgen, RDKit, pycalphad, MDAnalysis (no torch) | No | `base`, `atomate2`, `drugdisc`, `smol` |
| `mlip` | PyTorch 2.14.1, MACE-torch 0.3.16, MatGL 4.1.0, nvalchemi-toolkit 0.2.0 | Yes | `mace`, `matgl` |
| `fairchem` | PyTorch 2.13.0, fairchem-core 2.23.0, nvalchemi-toolkit 0.2.0 | Yes | `fairchem` |

### Why Three Environments?
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
| `adit` | — | `generative` | Yes | ADiT all-atom diffusion transformer |
| `diffcsp` | — | `generative` | Yes | DiffCSP++ crystal structure generation |
| `mattergen` | — | `generative` | Yes | MatterGen generative diffusion model |

The generative servers (`adit`, `diffcsp`, `mattergen`) have no uv project. Through the plugin they run from the `generative` container image (linux/arm64 only); `configure_mcp.py` instead points them at their conda environments (`<conda>/envs/<name>-agent`) when it finds them, on either architecture.

## 3. Standalone Conda Stacks (Legacy / Special)

A few specialized skills remain on isolated Conda environments declared via `metadata.conda_env` in their `SKILL.md`:
- `adit-agent`, `diffcsp-agent`, `mattergen-agent`: the generative servers and skills, when not run from the `generative` image.
- `ms-gen`: LC-MS/MS prediction via ICEBERG (`chem-msms-predict`).
- `react-ot-agent`: Reaction transition state generation (`chem-react-ot`).
- `mace-agent`, `matgl-agent`, `fairchem-agent`: LAMMPS with MLIP plugins (`mat-lammps-md`).

All other directories under `conda-envs/*` are retained for legacy reference only.

## 4. Runtime Selection and Launcher Backend

All skills and servers execute through the `venv/run` launcher:
- `venv/run <venv>[+<extra>] <command> [args...]`
- `venv/run --server <name>`
- `venv/run --setup` (prepares environments)
- `venv/run --doctor` (checks runtime status and diagnostics)

The runtime backend is controlled by `ATOMISTIC_RUNTIME` (or `~/.config/atomistic_skills.yaml`):
- `auto` (default): Uses native `uv` if the machine is Linux x86_64/aarch64 with compatible glibc (per `venv/platforms.tsv`) and a C compiler (`gcc`). Otherwise, automatically falls back to container images (`ghcr.io/learningmatter-mit/atomisticskills-<name>:1.5.0`).
- `uv`: Enforces host `uv` execution.
- `docker`, `podman`, `apptainer`, `singularity`: Enforces container execution with host paths mounted at identical locations.

## 5. Shell CLI Fallback for MCP Tools

When an MCP server is not connected via stdio, any MCP tool can be executed directly from the shell:
```bash
venv/run <venv> python -m src.mcp_server.cli <server> <tool> key=value ...
```
- Pass `--list` to view all available tools for a server.
- Multiple tool calls chained in one command share process memory (e.g., `load_model` followed by `relax_structure`).
