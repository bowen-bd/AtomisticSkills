# Conda Environments in AtomisticSkills

With version 1.5.0, AtomisticSkills has transitioned from maintaining ~20 isolated Conda environments to **three unified uv projects** (`venv/cpu`, `venv/mlip`, and `venv/fairchem`) launched via `venv/run`.

Most directories in `conda-envs/` are now retained for **historical reference only**. Only a few specialized environments remain active for skills that cannot yet be represented in a modern `uv` project.

## Active Conda Environments

The following environments are still actively referenced by skills that declare `metadata.conda_env` in their `SKILL.md`:

| Environment | Skill(s) | Description | Why It Remains Conda |
| :--- | :--- | :--- | :--- |
| `adit-agent` | `ml-generative-adit` | ADiT all-atom diffusion transformer | Not published on PyPI; compiled extensions. Runs via `generative` container image on aarch64, or this conda env on amd64. |
| `diffcsp-agent` | `ml-generative-diffcsp` | DiffCSP++ structure generation | Not published on PyPI; complex PyG dependencies. Runs via `generative` container image on aarch64, or this conda env on amd64. |
| `mattergen-agent` | `ml-generative-mattergen` | MatterGen crystal generation | Hard-pins torch 2.2.1+cu118 and source-compiled PyG extensions. Runs via `generative` container on aarch64, or this conda env on amd64. |
| `ms-gen` | `chem-msms-predict` | ICEBERG LC-MS/MS spectral prediction | Legacy PyTorch and package stack (`ms-pred`). |
| `react-ot-agent` | `chem-react-ot` | React-OT transition state generation | Legacy PyTorch / torch-geometric requirements. |
| `mace-agent`, `matgl-agent`, `fairchem-agent` | `mat-lammps-md` | LAMMPS molecular dynamics with MLIP plugins | Standalone LAMMPS binaries built with MLIP C++ libraries to avoid Python/Torch runtime collisions. |

## Legacy Environments (Replaced by uv Projects)

All other environments listed below have been migrated to the three `uv` projects under `venv/`. Their specifications under `conda-envs/<name>/` remain for reference:

| Legacy Conda Environment | Migrated To | Notes |
| :--- | :--- | :--- |
| `base-agent` | `venv/cpu` | Core materials, pymatgen, ASE, MP API |
| `atomate2-agent` | `venv/cpu` | Atomate2, Jobflow |
| `drugdisc-agent` | `venv/cpu` | RDKit, descriptors, fingerprints |
| `smol-agent` | `venv/cpu` | Cluster expansion, Monte Carlo |
| `nmr-agent` | `venv/cpu` | NMR prediction and Wasserstein deconvolution |
| `phasefield-agent` | `venv/cpu` | FiPy, phase-field simulations |
| `calphad-agent` | `venv/cpu` | pycalphad, CALPHAD diagrams |
| `xrd-agent` | `venv/cpu` | DARA, XRD refinement and search |
| `orca-agent` | `venv/cpu` | SCINE / ReaDuct wrapper (x86_64 only) |
| `drugmd-agent` | `venv/cpu+openmm` | OpenMM, PDBFixer (optional extra) |
| `void-agent` | `venv/cpu+void` | VOID guest docking (optional extra) |
| `scd-agent` | `venv/mlip` | SelfConditionedDenoisingAtoms models |
| `mace-agent` (standard) | `venv/mlip` | MACE foundation models |
| `matgl-agent` (standard) | `venv/mlip` | MatGL (CHGNet, TensorNet, M3GNet) |
| `fairchem-agent` (standard) | `venv/fairchem` | FairChem (UMA, eSEN) |
