---
name: mat-epw-mobility
description: Compute phonon-limited carrier mobility and mode-resolved electron-phonon coupling in 2D materials from first principles with Quantum ESPRESSO and EPW.
metadata:
  category: [materials]
---

# mat-epw-mobility

## Goal
To compute the intrinsic phonon-limited carrier mobility $\mu(T)$ of a 2D
semiconductor in the Self-Energy Relaxation Time Approximation (SERTA), together
with the mode-resolved electron-phonon coupling matrix elements $|g(k, q, \nu)|$,
from first principles using the Quantum ESPRESSO + EPW pipeline. The route
interpolates the electron-phonon vertex onto dense fine grids via maximally
localized Wannier functions and applies the 2D Frohlich long-range kernel needed
for polar monolayers.

## Background
Carrier mobility limited by phonon scattering requires the electron-phonon
matrix elements $g_{mn\nu}(k, q)$ on grids far denser than any tractable DFPT
calculation. EPW solves this by computing $g$ on a coarse q-grid with DFPT,
transforming to a maximally localized Wannier basis, and interpolating to fine
k/q meshes. For 2D polar materials, the long-range Frohlich part of $g$ diverges
as $1/q$ near the zone centre and must be treated with a 2D-truncated Coulomb
kernel (`lpolar`, `system_2d`), otherwise $\mu$ collapses by a factor of ~3.

Unlike the DFT+AMSET route in
[mat-dft-electronic-transport](../mat-dft-electronic-transport/SKILL.md), which
uses a momentum-relaxation-time approximation on VASP band structures, this skill
computes the full first-principles electron-phonon vertex. It is complementary to
[mat-dft-electron-phonon](../mat-dft-electron-phonon/SKILL.md) (which targets
temperature-dependent bandgap renormalization) and to
[mat-phonon](../mat-phonon/SKILL.md) (MLIP phonons).

The pipeline is a DAG. Steps 3 and 5 are optional (a standalone dielectric check
and a DFPT $|g|$ benchmark); the transport path is 1 -> 2 -> 4 -> 6 -> 7 -> 9.

```
                       1 SCF
           +-------------+--------------+
           v             v              v
      3 DFPT-Gamma   4 DFPT          2 NSCF
      (eps_inf, Z*)  uniform-q       (explicit k)
      (optional)         |               |
                         v               |
                    6 pp.py gather       |
                         |    \          |
                         |     v         v
                         |   7 Wannierize
                         |     |     |
                         v     v     v
                    9 EPW    8 EPW  (8 also <- 5 DFPT single-q
                    SERTA    prtgkk      benchmark, optional)
                    mu(T)    |g|
```

The worked example below runs the whole pipeline end-to-end on a ZrS2 monolayer
and validates EPW-interpolated $|g|$ against the DFPT reference to 0.05% on the
gauge-invariant sum.

## Instructions

> [!IMPORTANT]
> Quantum ESPRESSO (`pw.x`, `ph.x`) and EPW (`epw.x`, `pp.py`) are external
> compiled MPI binaries, exactly as VASP is in
> [mat-dft-vasp](../mat-dft-vasp/SKILL.md). This skill was validated on a source
> build of **QE 7.4.1 / EPW 5.8.1** with **ONCV SG15 PBE v1.2** pseudopotentials.
> The `# Env: base-agent` annotations apply only to the Python parser scripts.
> Reference input decks for every step are in
> [`resources/inputs/`](resources/inputs/).

### 1. Ground-state SCF
Compute the charge density and Kohn-Sham orbitals every later step consumes. For
2D materials, `assume_isolated = '2D'` truncates the out-of-plane Coulomb tail and
**must** be mirrored in `ph.x` and EPW. Use the deck
[`resources/inputs/scf.in`](resources/inputs/scf.in).

```bash
# Requires: Quantum ESPRESSO 7.4.1 (external MPI build)
mpirun -np <ranks> pw.x -in scf.in > scf.out
```

Confirm `convergence has been achieved`. For ZrS2 the total energy is
-136.1241 Ry and the VBM (HOMO) is -6.4435 eV.

### 2. NSCF on an explicit uniform k-grid
EPW folds Bloch orbitals on an **explicit** uniform k-grid into Wannier
functions. Use `K_POINTS crystal` with the full list, never `automatic` (any
compression desynchronises EPW's k-indexing). Generate the list and stage the
NSCF into its own `tmp/` so it does not overwrite the SCF save tree that DFPT
needs. Deck: [`resources/inputs/nscf.in`](resources/inputs/nscf.in).

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/mat-epw-mobility/scripts/gen_kpoints.py 12 12 1
```

```bash
# Requires: Quantum ESPRESSO 7.4.1 (external MPI build)
mpirun -np <ranks> pw.x -in nscf.in > nscf.out
```

`nbnd` must cover the disentanglement window (30 for ZrS2 = 6 occupied + 24
empty). For ZrS2 the gap is 1.1724 eV (CBM -5.2711 eV).

### 3. (Optional) Gamma DFPT for dielectric tensor and Born charges
Compute $\varepsilon_\infty$ and Born effective charges $Z^*$ as a fast
standalone sanity check (EPW itself reads them from step 4's Gamma irrep). A
non-symmetric or non-positive-definite $\varepsilon_\infty$ signals an upstream
error. Deck: [`resources/inputs/ph_gamma.in`](resources/inputs/ph_gamma.in).

```bash
# Requires: Quantum ESPRESSO 7.4.1 (external MPI build)
mpirun -np <ranks> ph.x -in ph_gamma.in > ph_gamma.out
```

For ZrS2: $\varepsilon_\infty^{xx}$ = 2.973, $Z^*_{xx}(\mathrm{Zr})$ = +7.003.

### 4. DFPT on a coarse uniform q-grid
The dynamical matrices and self-consistent perturbation potentials (`dvscf`) EPW
interpolates from. This is the most expensive step; set `recover = .true.` so a
timeout only loses the in-progress q. `fildvscf` is **required** or EPW has no
long-range kernel data. `nq1/nq2/nq3` must match the coarse `nk*` EPW reads
later. Deck: [`resources/inputs/ph_uniform.in`](resources/inputs/ph_uniform.in).

```bash
# Requires: Quantum ESPRESSO 7.4.1 (external MPI build)
mpirun -np <ranks> ph.x -in ph_uniform.in > ph_uniform.out
```

For ZrS2 (6x6x1) this yields 7 irreducible q and phonons in 16.9-337.6 cm^-1
with no imaginary modes.

### 5. (Optional) Single-q DFPT reference for |g|
The ground-truth $|g(k, q, \nu)|$ at one finite q, against which the EPW
interpolation (step 8) is benchmarked. **q must not be (0, 0, 0)** (the Frohlich
cusp diverges); use e.g. crystal (0.005, 0, 0). Stage the NSCF `tmp/` (not SCF)
so the CBM band is present. Deck:
[`resources/inputs/ph_single_q.in`](resources/inputs/ph_single_q.in).

```bash
# Requires: Quantum ESPRESSO 7.4.1 (external MPI build)
mpirun -np <ranks> ph.x -in ph_single_q.in > ph_q.out
```

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/mat-epw-mobility/scripts/parse_prt.py ph_q.out --fermi -5.655843
```

Compare on rank-sorted $|g|$ or the gauge-invariant $\sum |g|^2$ over the
degenerate LO/TO quartet, never on the raw mode index.

### 6. Gather the EPW save/ tree
Collect the distributed `dvscf` and `phsave` files from step 4 into the `save/`
layout EPW expects, using `pp.py` (bundled with EPW). `pp.py` reads the prefix
from stdin.

```bash
# Requires: EPW pp.py (bundled with Quantum ESPRESSO 7.4.1)
printf 'zrs2\n' | python3 pp.py
```

This produces `save/zrs2.dvscf_q*`, `save/zrs2.dyn_q*`, and `save/zrs2.phsave/`.

### 7. Wannierization (EPW write stage)
Build maximally localized Wannier functions and write the EPW archive the
interpolation reads. This is the single strongest predictor of downstream
quality. **`guiding_centres = .true.` is required** in long-vacuum 2D cells, or WF
centres fold to a wrong z-image and the total spread explodes ~100x. The
projections, `nbndsub`, and `exclude_bands` are material-specific. Deck:
[`resources/inputs/epw_write.in`](resources/inputs/epw_write.in).

```bash
# Requires: EPW 5.8.1 (external MPI build)
mpirun -np <ranks> epw.x -npool <ranks> -in epw_write.in > epw_write.out
```

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/mat-epw-mobility/scripts/parse_wout.py zrs2.wout
```

`-npool N` with N = ranks is **mandatory** (EPW aborts in < 1 s otherwise). For
ZrS2, `num_wann` = 9, $\Omega_I$ = 18.5 A^2, $\Omega_{total}/\Omega_I$ ~ 1.05.

### 8. (Optional) Mode-resolved interpolated |g|
Print EPW-interpolated $|g(k, q, \nu)|$ on an explicit (k, q) list and compare to
the step-5 DFPT reference. Use explicit `filkf` + `filqf` (not a uniform fine grid
with an arbitrary q, which aborts in `kpmq_map`). Note that the coordinate list files
(`kpoints.dat` and `qpoints.dat`) require a header line specifying coordinate type and count
(e.g., `1 crystal` followed by coordinate lines), otherwise the Fortran parser fails. Deck:
[`resources/inputs/epw_prtgkk.in`](resources/inputs/epw_prtgkk.in).

```bash
# Requires: EPW 5.8.1 (external MPI build)
mpirun -np <ranks> epw.x -npool <ranks> -in epw_prtgkk.in > epw_prtgkk.out
```

```bash
# Venv: venv/cpu
uv run --project venv/cpu python skills/mat-epw-mobility/scripts/parse_epw_prtgkk.py epw_prtgkk.out --fermi -5.655843
```

### 9. SERTA carrier mobility
Compute $\mu(T)$ in the semiconductor SERTA branch on production fine meshes.
`lpolar = .true.` and `system_2d` must match step 7 exactly.

> [!WARNING]
> Setting `etf_mem = 3` turns on `lfast_kmesh = .true.` to save memory, which
> **silently bypasses** the standard SERTA drift mobility print statements.
> For printing the drift mobility table and producing `zrs2_elcond_e`, use `etf_mem = 1`.
> If you encounter Out-Of-Memory (OOM) errors on large grids, run on a single core
> (to avoid MPI pool k-point splitting problems) using `etf_mem = 0` and
> `epmatkqread = .true.` to read pre-computed binary `.epb` files.

Deck: [`resources/inputs/epw_mob.in`](resources/inputs/epw_mob.in).

```bash
# Requires: EPW 5.8.1 (external MPI build)
mpirun -np <ranks> epw.x -npool <ranks> -in epw_mob.in > epw_mob.out
```

Read $\mu_{xx}$ from the last column of `zrs2_elcond_e`. Converge in order of
impact: `nkf` (= `nqf`) first, then `fsthick` (wide enough to cover the band-edge
carriers), then `degaussw` (fixed-smearing branch only). For ZrS2 at 1e13 cm^-2
electron doping and 300 K, $\mu$ lands near 195 cm^2/V/s at a 100x100 fine mesh.

## Examples
See [`examples/zrs2-monolayer/`](examples/zrs2-monolayer/README.md) for the
full end-to-end run on a ZrS2 1T monolayer, including expected values at every
step and the DFPT-vs-EPW $|g|$ validation against the gauge-invariant reference.

## Constraints
- **External codes**: requires an MPI build of Quantum ESPRESSO 7.4.1 with EPW
  and Wannier90; the Python parsers run in `base-agent`. Validated with ONCV
  SG15 PBE v1.2 pseudopotentials. Version pins reflect what was tested, not a
  claim that other versions are broken.
- **2D truncation**: `assume_isolated = '2D'` in `pw.x` and `ph.x`, and
  `lpolar = .true.` with `system_2d` in EPW, must all be set and consistent, or
  the 2D Frohlich kernel is dropped and $\mu$ collapses by ~3x.
- **No q = (0, 0, 0)** for electron-phonon coupling in polar 2D systems: the
  Frohlich $1/q$ cusp diverges. Use q >= (0.005, 0, 0) crystal (steps 5, 8).
- **`fildvscf` is required** in `ph.x` (steps 4, 5), even for single-q `prt`;
  omitting it causes a silent MPI abort at high rank count.
- **`guiding_centres = .true.` is required** for Wannierization in long-vacuum
  2D cells (step 7).
- **`-npool N` (N = rank count) is mandatory** for every EPW run.
- **MPI layout is machine-dependent**: on the validated build, DFT/DFPT ran at
  `-np 32` (higher counts aborted DFPT silently). If a DFPT or EPW step aborts via
  MPI with no Fortran trace, re-run at a lower rank count to expose the error.
- **Material-specific inputs**: `ecutwfc`, `nbnd`, projections, `nbndsub`, and
  `exclude_bands` are set for ZrS2 in the decks and must be adapted to your
  material and band manifold.
- **Gauge invariance**: benchmark $|g|$ on rank-sorted values or
  $\sum |g|^2$ over the degenerate LO/TO quartet, never on the raw mode index.
- **Disk**: `etf_mem = 3` checkpoints interpolated $|g|$ to disk; for a
  100x100 x 100x100 grid this can exceed 100 GB.

## References
- S. Ponce, E. R. Margine, C. Verdi, F. Giustino, "EPW: Electron-phonon coupling,
  transport and superconducting properties using maximally localized Wannier
  functions", *Comput. Phys. Commun.* **209**, 116 (2016).
  [DOI](https://doi.org/10.1016/j.cpc.2016.07.028)
- H. Lee, S. Ponce, et al., "Electron-phonon physics from first principles using
  the EPW code", *npj Comput. Mater.* **9**, 156 (2023).
  [DOI](https://doi.org/10.1038/s41524-023-01107-3)
- C. Verdi, F. Giustino, "Frohlich Electron-Phonon Vertex from First Principles",
  *Phys. Rev. Lett.* **115**, 176401 (2015).
  [DOI](https://doi.org/10.1103/PhysRevLett.115.176401)
- T. Sohier, M. Calandra, F. Mauri, "Two-dimensional Frohlich interaction in
  transition-metal dichalcogenide monolayers: Theoretical modeling and
  first-principles calculations", *Phys. Rev. B* **94**, 085415 (2016).
  [DOI](https://doi.org/10.1103/PhysRevB.94.085415)
- G. Pizzi, et al., "Wannier90 as a community code: new features and
  applications", *J. Phys. Condens. Matter* **32**, 165902 (2020).
  [DOI](https://doi.org/10.1088/1361-648X/ab51ff)
- P. Giannozzi, et al., "Advanced capabilities for materials modelling with
  Quantum ESPRESSO", *J. Phys. Condens. Matter* **29**, 465901 (2017).
  [DOI](https://doi.org/10.1088/1361-648X/aa8f79)

---

**Author:** cz2014
**Contact:** [GitHub @cz2014](https://github.com/cz2014)
