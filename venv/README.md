# Python Environments

AtomisticSkills runs all skill commands and MCP servers through the unified launcher `venv/run`. Three uv projects under `venv/` replace the ~20 legacy conda environments for everything except a few specialized stacks. Each project has its own `pyproject.toml`, committed `uv.lock`, and isolated resolver boundary.

| uv project | Contents | Accelerator | MCP Servers |
| :--- | :--- | :--- | :--- |
| `venv/cpu` | Materials, chemistry, drug discovery and analysis stack (no torch) | none | `base`, `atomate2`, `drugdisc`, `smol` |
| `venv/mlip` | MACE and MatGL (PyG only) on top of the CPU stack | torch 2.14, CUDA | `mace`, `matgl` |
| `venv/fairchem` | FairChem (UMA, eSEN) on top of the CPU stack | torch 2.13, CUDA | `fairchem` |

## Running Commands and MCP Servers

Run commands through the `venv/run` launcher without manual environment activation:

```bash
# Run a skill script
venv/run cpu python skills/<skill>/scripts/<script>.py ...
venv/run mlip python skills/<skill>/scripts/<script>.py ...

# Run with an optional extra
venv/run cpu+openmm python ...

# Start an MCP server over stdio
venv/run --server mace

# Setup and diagnostics
venv/run --setup     # syncs all three projects ahead of time
venv/run --doctor    # reports host compatibility and runtime readiness
```

## Versions and Packages

Floors (not strict pins) keep the scientific packages current while respecting fundamental conflicts:

| Package | mlip | fairchem | cpu |
| :--- | :--- | :--- | :--- |
| torch | 2.14.1 (cu130 on aarch64) | 2.13.0 | — |
| mace-torch / e3nn | 0.3.16 / 0.4.4 | — | — |
| matgl | 4.1.0 (PyG only; DGL removed) | — | — |
| fairchem-core / e3nn | — | 2.23.0 / 0.6.0 | — |
| nvalchemi-toolkit | 0.2.0 | 0.2.0 | — |
| numpy | 2.3.5 | 2.3.5 | 2.5.3 |
| pandas | 3.0.6 | 3.0.6 | 3.0.6 |

### Conflicts That Force Separate Environments

1. **`mace-torch` vs `fairchem-core`**: `mace-torch` pins `e3nn==0.4.4`, whereas `fairchem-core` requires `e3nn>=0.5`. `mlip` and `fairchem` cannot merge into a single environment on any platform.
2. **PyTorch versions**: `fairchem-core` 2.23 requires `torch~=2.13`, while `mlip` takes torch 2.14.1.
3. **NumPy constraints**: `nvalchemi-toolkit` 0.2 requires `numpy<2.4`, so the GPU projects use numpy 2.3.5 while `cpu` uses numpy 2.5+.

### Pins That Remain

- `mcp<2`: All ten servers use the 1.x `FastMCP` API; porting to mcp 2.x is a planned follow-up.
- `pymol-open-source==3.2.0a0`: The only release providing Python 3.12 wheels.

## Optional Extras

System-dependent or heavy dependencies are isolated in optional extras:

- `openmm`: OpenMM and PDBFixer (requires glibc ≥ 2.34).
- `pymol`: PyMOL open-source (x86_64 only).
- `docking`: AutoDock Vina (builds against Boost and SWIG on aarch64).
- `void`: VOID guest docking (installed from git).
- `transport`: AMSET and BoltzTraP2 (requires git and cmake at build time).

Use them as `venv/run cpu+openmm ...` or `venv/run cpu+docking ...`.

## Environment Mapping (Former Conda vs uv Projects)

| Former conda env | Current Runtime / venv | Note |
| :--- | :--- | :--- |
| `base-agent`, `drugdisc-agent`, `smol-agent`, `atomate2-agent` | `cpu` | |
| `nmr-agent`, `phasefield-agent`, `calphad-agent`, `xrd-agent` | `cpu` | |
| `drugmd-agent` | `cpu+openmm` | `pymol` extra is x86_64 only |
| `orca-agent` | `cpu` | x86_64 only (SCINE wheels); requires external ORCA binary |
| `atomistic-agent` | `cpu+void` | VOID guest docking |
| `mace-agent`, `matgl-agent` | `mlip` | |
| `scd-agent` | `mlip` | |
| `fairchem-agent` | `fairchem` | |
| `adit-agent`, `diffcsp-agent`, `mattergen-agent` | `generative` image / conda | Generative models |
| `ms-gen` | `ms-gen` conda env | ICEBERG (`chem-msms-predict`) |
| `react-ot-agent` | `react-ot-agent` conda env | React-OT (`chem-react-ot`) |
| `mace-agent`, `matgl-agent`, `fairchem-agent` | conda envs | `mat-lammps-md` with ML plugins |

## Architectures and Container Fallback

All three uv projects resolve for both `linux/x86_64` and `linux/aarch64`.
- Host requirements: Linux, compatible glibc (≥ 2.28 for cpu, ≥ 2.34 for aarch64 GPU stacks per `venv/platforms.tsv`), and a C compiler (`gcc`).
- For GPU acceleration: NVIDIA driver ≥ 580 on aarch64 (CUDA 13) or standard drivers on x86_64 (CUDA 12.8).
- Hosts that lack these (older clusters, macOS, or machines without a compiler) automatically run via container images (`docker`, `podman`, or `apptainer`) when `ATOMISTIC_RUNTIME=auto` (the default). Host paths are mounted at identical locations so outputs land in the active workspace.
