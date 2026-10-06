# LAMMPS MLIP Backend Matrix Check

## Goal
Verify that each MLIP stack has its own working LAMMPS binary, run in the
environment it was built against.

## Steps

Run from the AtomisticSkills checkout. The builds need git, cmake, g++, an MPI
compiler wrapper (`mpicxx`) and, for Kokkos, nvcc; binaries go to
`$LAMMPS_ROOT` (default `~/.cache/atomisticskills/lammps`).

1. Build the MACE-linked binary:
```bash
bash skills/mat-lammps-md/scripts/build_lammps_mace.sh
```

2. Build the MatGL-linked binary (Kokkos arch from `nvidia-smi --query-gpu=compute_cap`):
```bash
KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80 \
bash skills/mat-lammps-md/scripts/build_lammps_matgl.sh
```

3. FairChem needs no build: the `lammps` extra of the fairchem environment
   installs the LAMMPS wheel and `fairchem-lammps`.

4. Verify each binary in its environment:
```bash
venv/run mlip+lammps ~/.cache/atomisticskills/lammps/mace/lmp -h | grep ML-MACE
venv/run mlip+lammps ~/.cache/atomisticskills/lammps/matgl/lmp -h | grep -E "KOKKOS|ML-IAP|PYTHON"
venv/run fairchem+lammps lmp -h | head -5
venv/run fairchem+lammps lmp_fc --help
```

## Expected Output
- All three binaries start without Python embedding errors.
- The MACE binary lists ML-MACE; the MatGL binary lists KOKKOS, ML-IAP and PYTHON.
- Each binary runs only in the environment it was built against.
