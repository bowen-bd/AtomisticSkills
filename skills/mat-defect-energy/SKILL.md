---
name: mat-defect-energy
description: Calculate point-defect formation energies (vacancies, substitutions, interstitials) using MLIPs.
metadata:
  category: [materials]
  venv: [cpu, mlip]
---

# Point-Defect Formation Energy (MLIP)

<!-- mcp-tools-note -->
> [!NOTE]
> Steps written `server.tool` are MCP tool calls: `base.search_materials_project_by_formula` is the `search_materials_project_by_formula`
> tool of the `base` server (`mcp__base__search_materials_project_by_formula`, or
> `mcp__plugin_atomistic-skills_base__search_materials_project_by_formula` when installed as a plugin).
> Without a connected server, run the same tools from the shell. Tools named in
> one command share a process, so a model loaded by `load_model` stays loaded:
>
> ```bash
> ${CLAUDE_SKILL_DIR}/../../venv/run cpu python -m src.mcp_server.cli base search_materials_project_by_formula key=value
> ${CLAUDE_SKILL_DIR}/../../venv/run mlip python -m src.mcp_server.cli mace load_model key=value relax_structure key=value
> ```

## Goal
To calculate the formation energy ($E_f$) of neutral point defects (vacancies, substitutions, and interstitials) using Machine Learning Interatomic Potentials (MLIPs). Formation energy is defined as:

$$E_f = E_\mathrm{defect} - \frac{n_\mathrm{defect}}{n_\mathrm{bulk}} E_\mathrm{bulk} + \sum_i \Delta n_i \mu_i$$

where $E_\mathrm{defect}$ and $E_\mathrm{bulk}$ are the total energies of the defective and pristine supercells, $n$ is the number of atoms, $\Delta n_i$ is the change in number of species $i$, and $\mu_i$ is the chemical potential of species $i$.

## Instructions

### 1. Select Level of Theory
Choose an MLIP model. See [ml-foundation-potentials](../ml-foundation-potentials/SKILL.md) for guidance.
- **Recommended**: r2SCAN-level potentials for inorganic defects (e.g., `MACE-MH-1` with `matpes_r2scan` head).
- Use the **same model** for bulk and defect calculations.

### 2. Obtain Bulk Structure
Start with a relaxed bulk primitive cell. You can retrieve one from Materials Project:
```bash
base.search_materials_project_by_formula(formula="MgO", save_to_file="MgO.cif")
```

### 3. Relax Bulk Structure
Relax the bulk unit cell to get the reference energy:
```bash
mace.load_model(model_name="MACE-MH-1", task_name="matpes_r2scan")
mace.relax_structure(
    structure_data="MgO.cif",
    relax_cell=True,
    fmax=0.01,
    output_dir="bulk_relaxation/"
)
```
Record the final **energy per atom** from the output.

### 4. Generate Defect Supercells
Use the defect generation script with `pymatgen-analysis-defects`:
```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/generate_defects.py \
    --bulk bulk_relaxation/relaxed_structure.cif \
    --supercell_size 2 2 2 \
    --defect_type vacancy \
    --output defect_structures/
```

**Options for `--defect_type`:**
- `vacancy` — removes each symmetry-unique atom
- `substitution` — replaces atoms with `--substitute_element` at each unique site
- `interstitial` — inserts `--interstitial_element` at Voronoi interstitial sites
- `all` — generates all vacancy types

### 5. Relax Defect Supercells
Relax **without cell relaxation** (fixed supercell volume). This applies to the defect
supercells only -- the bulk cell in step 3 and the elemental references in step 6 are
both relaxed with `relax_cell=True`:
```bash
mace.relax_structure(
    structure_data="defect_structures/",
    relax_cell=False,  # Fixed cell for defect calculations
    fmax=0.02,
    output_dir="defect_relaxations/"
)
```

### 6. Calculate Formation Energies
Compute defect formation energies:
```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/calculate_defect_energy.py \
    --bulk_dir bulk_relaxation/ \
    --defect_dir defect_relaxations/ \
    --supercell_size 2 2 2 \
    --output defect_energies.json
```
The script automatically:
- Parses bulk and defect relaxation results
- Determines removed/added species and computes $\Delta n_i$
- Uses elemental energies from [mat-elemental-energies](../mat-elemental-energies/SKILL.md) as default chemical potentials (metal-rich limit)
- Reports formation energies in eV

If you derive $\mu_i$ yourself instead of reading the library, relax the elemental
reference **cell and coordinates** (`relax_cell=True`) with the same potential. A
chemical potential is only meaningful at the reference phase's own minimum for that
potential: evaluating an MP structure at its DFT geometry leaves it above the
potential's minimum and that error passes straight into every formation energy.
For O with `TensorNet-PES-MatPES-PBE-2025.2`, positions-only relaxation of mp-12957
gives -4.968 eV/atom versus -5.118 fully relaxed -- a 0.15 eV/atom error.

## Examples

### Oxygen Vacancy in MgO
```bash
# 1. Get MgO structure
base.search_materials_project_by_formula(formula="MgO")

# 2. Relax bulk
mace.load_model(model_name="MACE-MH-1", task_name="matpes_r2scan")
mace.relax_structure(structure_data="MgO.cif", relax_cell=True, fmax=0.01, output_dir="bulk/")

# 3. Generate O vacancy supercells
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/generate_defects.py \
    --bulk bulk/relaxed_structure.cif --supercell_size 3 3 3 --defect_type vacancy --output vacancies/

# 4. Relax defect structures
mace.relax_structure(structure_data="vacancies/", relax_cell=False, fmax=0.02, output_dir="vac_relax/")

# 5. Compute formation energies
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/calculate_defect_energy.py \
    --bulk_dir bulk/ --defect_dir vac_relax/ --supercell_size 3 3 3 --output vac_energies.json
```
Expected: O vacancy formation energy ~6–8 eV (DFT reference: ~7.2 eV for neutral O vacancy in MgO).

## Constraints
- **Neutral defects only**: This skill does NOT handle charged defects. For charged defects with finite-size corrections, use [mat-defect-energy-dft](../mat-defect-energy-dft/SKILL.md).
- **Fixed cell**: Do NOT relax the unit cell during defect relaxation — the supercell must remain fixed to be commensurate with the bulk reference. This constraint is scoped to the defect supercell: the bulk cell and the elemental chemical-potential references are both fully relaxed (cell and coordinates).
- **Supercell size**: Use at least 3×3×3 for cubic systems to minimize periodic image interactions. Formation energies converge with supercell size.
- **Chemical potential**: Default uses metal-rich limit (elemental energies). For environment-specific stability, manually provide chemical potentials.
- **Environments**: Defect generation and energy calculation scripts require `cpu`.

---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
