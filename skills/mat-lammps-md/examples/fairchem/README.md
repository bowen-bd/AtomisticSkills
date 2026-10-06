# FairChem Example: CO Adsorption on Cu(111)

This example runs adsorption energy with FAIR-Chem `lmp_fc`:

1. builds a clean Cu(111) slab, isolated CO molecule, and adsorbed CO/Cu(111);
2. evaluates each with the same FairChem setup;
3. computes adsorption energy as:
   `E_ads = E(CO/Cu111) - E(Cu111) - E(CO)`.

```bash
venv/run fairchem+lammps bash skills/mat-lammps-md/examples/fairchem/run_fairchem_co_cu111_adsorption.sh
```

## Output
- `./out-fairchem-co-cu111/cu111_clean.data`
- `./out-fairchem-co-cu111/co_gas.data`
- `./out-fairchem-co-cu111/co_on_cu111.data`
- `./out-fairchem-co-cu111/energies.json`
- `./out-fairchem-co-cu111/adsorption_summary.txt`

## Expected result

With the default UMA task `oc20` (adsorbates on surfaces) and these unrelaxed
geometries, `E_adsorption_eV` is about -0.39 eV (CO carbon-down on a Cu top
site; experiment about -0.5 eV). Other tasks are trained on other chemistry:
`TASK_NAME=omat` (bulk materials) overbinds here at about -1.35 eV.
