---
name: mat-elemental-energies
description: A library of ground-state element structures and their energies calculated from MLIPs. Used to calculate formation energies of compounds.
metadata:
  category: [materials]
---

# Elemental Energies

## Goal
To provide a centralized library of the most stable phases (ground states) for each element and their corresponding reference energies calculated using different Machine Learning Interatomic Potentials (MLIPs). This avoids redundant relaxations and ensure consistency in thermodynamics calculations (e.g., formation energy, stability).

## Instructions

### 1. Retrieve Elemental Energies
To get the energies for a list of elements from a specific checkpoint:
```bash
# Env: base-agent
python skills/mat-elemental-energies/scripts/get_elemental_energies.py --elements Li Fe O --checkpoint mace-mp-medium
```


## Library Status
As of **2026-01-30**, the library is fully expanded and contains **89 elements** (Ground states from H to Lr) across **36 MLIP variants**, including:
- **MACE**: MACE-MP (S/M/L), MACE-MH-0/1 (all heads), MACE-OMAT (S/M), MACE-MATPES (PBE/R2SCAN).
- **MatGL**: CHGNet (MPtrj, MatPES), M3GNet (MP, MatPES), TensorNet (MatPES).
- **FairChem**: UMA (S/M) with all heads (omat, omol, oc20).

## Constraints
- **Materials Project**: Structures must be queried from Materials Project to ensure they represent the true ground-state phases.
- **Naming**: Library files must follow the `<checkpoint_name>_energies.json` format.
- **Units**: Energies are stored as **eV/atom** (total potential energy of the relaxed structure divided by the number of atoms).
- **Relaxation**: The stored value comes from relaxing **both cell and coordinates**
  (`relax_cell=True`). A positions-only relaxation leaves the reference above the
  potential's own minimum and biases every formation energy that uses it.
- **Molecular-crystal elements are sensitive**: for O, N, F, Cl, H the MP ground state
  is a van der Waals-bound molecular crystal whose MLIP-relaxed cell differs a lot from
  the DFT cell, so the cell relaxation matters more here than for metals. With
  `TensorNet-PES-MatPES-PBE-2025.2`, mp-12957 (O2) gives -5.118 eV/atom fully relaxed
  versus -4.968 positions-only. More than one MP entry can also tie at
  `energy_above_hull = 0` for these elements (O2: mp-12957 and mp-1524462), so pin the
  `material_id` rather than relying on a formula lookup returning a stable order.
---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
