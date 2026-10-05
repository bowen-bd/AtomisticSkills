---
name: mat-lammps-md
description: Build and run LAMMPS molecular dynamics with isolated MLIP-specific binaries (MACE, MatGL/CHGNet, FairChem) to avoid Python and Torch stack conflicts.
metadata:
  category: [materials]
  venv: [fairchem, mlip]
---

# LAMMPS Molecular Dynamics with MLIPs

## Goal
Run GPU-accelerated LAMMPS molecular dynamics with MLIP backends using three isolated binaries (MACE, MatGL/CHGNet, FairChem) so Python embedding through `ML-IAP`/`mliappy` remains stable and reproducible.

## Instructions

1. **Select the MLIP backend and model family first** using the foundation-potential guide:
   - [ml-foundation-potentials](../ml-foundation-potentials/SKILL.md)
   - This determines which environment (`mlip` for MACE and MatGL, `fairchem` for FairChem) and which LAMMPS binary you must use.

2. **Check system prerequisites**.
```bash
nvidia-smi
nvcc --version
g++ --version
cmake --version
mpicxx --version
```

3. **Identify GPU compute capability and set Kokkos arch flag**.
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```
- Example mapping:
  - `8.0` -> `Kokkos_ARCH_AMPERE80`
  - `8.6` -> `Kokkos_ARCH_AMPERE86`
  - `8.9` -> `Kokkos_ARCH_ADA89`
  - `9.0` -> `Kokkos_ARCH_HOPPER90`
  - `10.0` -> `Kokkos_ARCH_BLACKWELL100`
  - `12.0`, `12.1` (e.g. GB10 / DGX Spark) -> `Kokkos_ARCH_BLACKWELL120` (Kokkos 4.6 has no 12.1 target; it runs, with a performance warning)

4. **Build the environment-matched LAMMPS binary** (choose one of the three paths below). Builds go to `$LAMMPS_ROOT` (default `~/.cache/atomisticskills/lammps`) and need the environment to run natively on the host.

   **Path A: MACE** (ACEsuit's LAMMPS fork with `ML-MACE`, linked against the `mlip` environment's libtorch)
```bash
bash ${CLAUDE_SKILL_DIR}/scripts/build_lammps_mace.sh
```
   - Binary: `~/.cache/atomisticskills/lammps/mace/lmp`
   - Runtime env: `mlip+lammps`
   - Needs a CUDA toolkit (`CUDA_HOME`) at least as new as the environment's torch build (12.6 for cu126, 13.0 for cu130); PyTorch's CMake config refuses an older one.

   **Path B: MatGL/CHGNet** (Kokkos with CUDA, ML-IAP with the Python coupling, embedding the `mlip` environment's Python)
```bash
KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80 \
bash ${CLAUDE_SKILL_DIR}/scripts/build_lammps_matgl.sh
```
   - Binary: `~/.cache/atomisticskills/lammps/matgl/lmp`
   - Runtime env: `mlip+lammps`
   - `LAMMPS_REF` defaults to `stable_22Jul2025_update4`, whose Kokkos knows current GPU architectures.

   **Path C: FairChem** (no build: the `lammps` extra installs the LAMMPS wheel and `fairchem-lammps`)
```bash
${CLAUDE_SKILL_DIR}/../../venv/run fairchem+lammps lmp_fc --help
```
   - Binaries: `lmp` and `lmp_fc` in the `fairchem+lammps` environment

5. **Run the selected binary in its matching environment**.
```bash
# (example; switch environment/binary pair as needed)
${CLAUDE_SKILL_DIR}/../../venv/run mlip+lammps ~/.cache/atomisticskills/lammps/mace/lmp -h
```

6. **Launch MD with the same binary-environment pair used during build**; do not cross-run binaries between MLIP stacks.

## Examples

See [scripts/three-backends-build-check/README.md](scripts/three-backends-build-check/README.md) for a minimal build/verification matrix across MACE, MatGL, and FairChem.
See the respective README.md files under [examples/mace/](examples/mace/), [examples/matgl/](examples/matgl/), and [examples/fairchem/](examples/fairchem/) for model-specific run scripts.

## Constraints
- **Strict binary-env pairing**: each LAMMPS binary must run only in the environment it was built against (`venv/run mlip+lammps` or `venv/run fairchem+lammps`).
- **No stack mixing**: never run the MACE or MatGL binary in the `fairchem` environment, or `lmp_fc` in `mlip`.
- **GPU arch alignment**: choose `KOKKOS_ARCH_*` from actual `compute_cap` output.
- **Python-coupled mode**: this workflow targets `ML-IAP`/`mliappy` usage.

## References
- Thompson et al., "LAMMPS - A flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales", *Computer Physics Communications*, 2022. [DOI](https://doi.org/10.1016/j.cpc.2021.108171)
- LAMMPS Manual, ML-IAP package documentation. [Link](https://docs.lammps.org/Packages_details.html#pkg-ml-iap)
- Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields". [arXiv](https://arxiv.org/abs/2206.07697)
- Deng et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling". [arXiv](https://arxiv.org/abs/2302.14231)
- FairChem documentation and model zoo. [Link](https://fair-chem.github.io/)

---

**Author:** Jurģis Ruža
**Contact:** [GitHub @JurgisR](https://github.com/JurgisR)
