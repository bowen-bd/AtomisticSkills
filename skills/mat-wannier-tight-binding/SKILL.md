---
name: mat-wannier-tight-binding
description: Construct and validate Wannier tight-binding models, inspect orbital localization and hopping amplitudes, and interpolate electronic bands from DFT or existing Wannier90 files.
metadata:
  category: [materials]
  venv: [cpu]
---

# mat-wannier-tight-binding

## Goal

Construct maximally localized Wannier functions (MLWFs) and a real-space Hamiltonian from DFT, or analyze an existing Wannier90 model. Validate numerical interpolation separately from its accuracy against independent DFT. This skill covers Quantum ESPRESSO → Wannier90 and Python analysis; the bundled examples reuse official tutorial DFT matrices.

## Background

The standard Marzari–Vanderbilt method minimizes

$$\Omega=\sum_n(\langle r^2\rangle_n-|\langle\mathbf r\rangle_n|^2)
=\Omega_I+\Omega_D+\Omega_{OD}.$$

At fixed subspace, $\Omega_I$ is gauge invariant; both diagonal and off-diagonal gauge-dependent parts are minimized by localization. $\Omega_D$ is **not** a displacement from inversion centers. For entangled bands, the Souza–Marzari–Vanderbilt step first selects a smooth subspace. If $V(\mathbf k)=U^{dis}(\mathbf k)U(\mathbf k)$, where $U^{dis}$ is rectangular, then

$$H(\mathbf R)=\frac1{N_k}\sum_{\mathbf k}e^{-i2\pi\mathbf k\cdot\mathbf R}
 V^\dagger(\mathbf k)\,\mathrm{diag}(\epsilon_{j\mathbf k})\,V(\mathbf k).$$

The scripts read standard **folded** Wannier90 v4.0.3 exports. Interpolation applies both the lattice degeneracy $N_R$ and orbital-pair shifts $\mathbf T$ from `_wsvec.dat`:

$$H_{mn}(\mathbf k)=\sum_\mathbf R\frac{H_{mn}(\mathbf R)}{N_R}
 \frac1{N_{T,mn\mathbf R}}\sum_\mathbf T e^{i2\pi\mathbf k\cdot(\mathbf R+\mathbf T)}.$$

Wannier90 has enabled `use_ws_distance=true` by default since v3.0. Preserve the sidecar with the Hamiltonian. The unshifted expression is supported only for explicitly identified legacy output. See the [official interpolation notes](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/notes_interpolations/) and [method definitions](https://wannier90.readthedocs.io/en/stable/user_guide/wannier90/methodology/).

## Instructions

### 1. Select the starting point and runtime

- Existing `_hr.dat`, `_wsvec.dat`, `.win` and `.wout`: start at step 5.
- A reproducible teaching example: use [GaAs](examples/gaas-valence/README.md) or [silicon](examples/silicon-sp3/README.md); neither requires QE or a pseudopotential download.
- New DFT calculation: use steps 2–4. Obtain a relaxed/converged structure and choose the target energy manifold first; [mat-dft-vasp](../mat-dft-vasp/SKILL.md) covers the alternative VASP route, whose interface settings differ.

`${CLAUDE_SKILL_DIR}` denotes this skill's absolute directory. The launcher and script paths below work from any working directory; calculation paths are relative to the current directory. The example READMEs also provide commands for use from the repository root. `wannier90.x`, `pw.x` and `pw2wannier90.x` are external executables, not installed by the `cpu` environment. Use an existing working Wannier90 installation, or build the pinned release with existing compiler/BLAS/LAPACK dependencies:

```bash
bash ${CLAUDE_SKILL_DIR}/scripts/build_wannier90.sh
export PATH="${CLAUDE_SKILL_DIR}/../../.agents/test/wannier90-build/install/bin:$PATH"
wannier90.x -v
```

The builder defaults to an isolated installation under `.agents/test`, verifies the requested source tag, installs through CMake and checks execution. `WANNIER_BUILD_DIR`, `WANNIER_PREFIX`, `WANNIER_REF`, `BUILD_JOBS`, `FC`, `BLAS_LIBRARIES` and `LAPACK_LIBRARIES` are optional overrides. No system package installation is performed.

### 2. Prepare consistent SCF, NSCF and Wannier inputs

[Templates](resources/templates/) provide a silicon starting point with identical explicit cells and atoms, plus all 64 NSCF k-points. They are **not** a reproduction of the bundled tutorial matrices: the pseudopotential, cutoff, lattice constant and energy zero are separate choices.

Copy `template_scf.in`, `template_nscf.in`, `template.win` and `template.pw2wan` into the calculation directory as `scf.in`, `nscf.in`, `silicon.win` and `silicon.pw2wan`. Supply the named pseudopotential in `pseudo/` and record its source/hash, XC functional, cutoffs, occupations, spin/SOC settings and mesh convergence.

The NSCF calculation must provide a full uniform mesh, with reciprocal basis, shift and ordering matching `.win`. Explicit points with `nosym=true` and `noinv=true` are the supplied QE convention. Automatic grids can also work if their **actual full output mesh and ordering** are matched; the keyword itself is not prohibited. Shifted meshes are valid when both sides use the same shift.

To change the mesh consistently:

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/generate_kmesh.py \
  --grid 6 6 6 --shift 0 0 0 --output-dir research/wannier/mesh
```

Replace the complete NSCF `K_POINTS` card and Wannier `mp_grid`/`kpoints` block using the two generated files. The SCF mesh may differ from the NSCF Wannier mesh. See [QE input conventions](https://www.quantum-espresso.org/Doc/INPUT_PW.html).

### 3. Choose projections and energy windows

Choose `num_wann` and initial projectors for the intended orbitals. Set `num_bands >= num_wann`; disentanglement needs additional bands. Frozen/outer energies are in eV on the **DFT eigenvalue energy zero**, not automatically relative to the Fermi energy. Replace the templates' provisional 6.4/17.0 eV windows after inspecting the new eigenvalues.

At every mesh point require

$$N_{\rm frozen}(\mathbf k)\le N_{\rm wann}\le N_{\rm outer}(\mathbf k).$$

Freeze states needed for the target observable. Not every model must include all occupied bands. Record lower as well as upper window bounds when excluding deep states. Check orbital character and sensitivity to windows/projections; small spreads alone do not guarantee a physically useful subspace. For spinor calculations use consistent noncollinear/SOC wavefunctions, `spinors=true` and spinor projector counts; the provided examples are nonmagnetic scalar calculations.

### 4. Generate matrices, localize and export

Run in the calculation directory, adjusting MPI ranks and executables for the installation:

```bash
pw.x -in scf.in > scf.out
pw.x -in nscf.in > nscf.out
wannier90.x -pp silicon
pw2wannier90.x -in silicon.pw2wan > pw2wan.out
wannier90.x silicon
```

Check each program's termination before proceeding. The interface writes `.mmn` (overlaps), `.amn` (trial projections) and `.eig` (eigenvalues). Eigenvalues are written automatically; `write_eig` is not an [accepted QE interface input](https://www.quantum-espresso.org/Doc/INPUT_pw2wannier90.html). Keep the DFT save directory and matching `.nnkp` through matrix generation. Changing cell, mesh, bands, spin settings or trial projectors requires regenerating the affected matrices.

Use `write_hr=true`, `use_ws_distance=true`, `bands_plot=true` and a defined `kpoint_path` for Hamiltonian and path export. Set explicit localization and disentanglement convergence tolerances/windows, with sufficient iteration limits. `write_hr` is the supported keyword; its replacement of `hr_plot` predates v4.

### 5. Check final-state completeness and iterative convergence

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/parse_wout.py \
  research/wannier/silicon.wout --require-converged --output-dir research/wannier/analysis
```

The parser requires a complete final state, consistent spread sums and normal termination. It distinguishes localization and disentanglement convergence. `--max-omega-tot` and `--max-omega-d` are optional **system-specific localization bounds**, not convergence tests. For historical fixed-iteration examples, omit `--require-converged` only deliberately; the report then says `CONVERGENCE_NOT_ESTABLISHED` if native convergence is not documented.

### 6. Inspect Hamiltonian elements and distances

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/parse_hr.py \
  research/wannier/silicon_hr.dat --win research/wannier/silicon.win \
  --wout research/wannier/silicon.wout --threshold 1e-4 \
  --output-dir research/wannier/analysis
```

Outputs are `tb_hamiltonian.json` and `tb_hoppings.csv`. On-site energies and raw folded matrix elements are gauge/energy-zero dependent. The CSV retains Fourier degeneracies separately; it is not an already-expanded sparse Hamiltonian. With cell and centres, distances use $|(\mathbf R+\mathbf T)A+\tau_n-\tau_m|$ in Å. With cell alone they are explicitly labeled lattice-translation distances. Without a cell, no spatial decay profile is claimed. The threshold filters the table only; it does not truncate the interpolated model.

### 7. Validate the interpolation implementation

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/interpolate_bands.py \
  research/wannier/silicon_hr.dat --kpoints research/wannier/silicon_band.kpt \
  --ref-bands research/wannier/silicon_band.dat --reference-kind wannier90 \
  --max-error 5e-5 --output-dir research/wannier/analysis
```

The sidecar and reference `.kpt` are discovered beside their corresponding inputs; `--wsvec` and `--ref-kpoints` override their paths. Missing/invalid files, mismatched point ordering, band counts or non-Hermitian matrices fail with nonzero exit. `band_comparison.json` records MAE/RMSE/maximum error, convention, scope and input hash; exceeding `--max-error` also returns nonzero.

The exported `tb_interpolated_bands.dat` uses standard two-column, band-separated distance/energy blocks and has a matching `.kpt`. Without reference bands, pass `--win` for a reciprocal-distance axis along a continuous list; otherwise the axis is explicitly an index. A path with disconnected or symmetry-equivalent segment endpoints should retain the reference path distances.

A comparison with `_band.dat` from the **same run** is a numerical regression check. It does not measure interpolation error against DFT.

### 8. Validate scientific accuracy and plot

For scientific validation, calculate independent DFT eigenvalues at off-mesh k-points using the same electronic-structure settings. Export them as band-separated distance/energy blocks plus a matching fractional `.kpt`. Select a corresponding manifold (`--ref-band-indices`, one-based), specify any known reference energy alignment (`--ref-energy-shift`, added to the reference), and restrict comparison to the target range if appropriate (`--energy-window`). Use `--reference-kind dft`; this label records the caller's declared provenance and cannot authenticate it.

Converge mesh, windows, projector choice, number of bands and DFT cutoffs for the intended observable. Do not promise sub-meV accuracy from a single coarse-mesh example. Outside a frozen subspace, disentangled eigenvalues may not correspond one-to-one to DFT bands; compare the target manifold and character deliberately. Band agreement alone does not validate transport or topology.

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/plot_tb_bands.py \
  research/wannier/analysis/tb_interpolated_bands.dat \
  --ref-bands research/wannier/silicon_band.dat \
  --labelinfo research/wannier/silicon_band.labelinfo.dat \
  --prefix silicon_tb_bands --output-dir research/wannier/analysis
```

Produces PNG and SVG with high-symmetry labels. Supply `--fermi` only when the energy zero is known; use `--reference-label DFT` and matching energy shift for an independent DFT overlay. Every script preserves all CLI defaults/options under its own stage in `input_configs.yaml`.

## Examples

- [GaAs valence MLWFs](examples/gaas-valence/README.md): four orbitals on a 2×2×2 mesh. Reproduces the official tutorial, while explicitly reporting its difference from Marzari–Vanderbilt Table II.
- [Silicon disentangled sp3 model](examples/silicon-sp3/README.md): eight orbitals from twelve bands on a 4×4×4 mesh. Exercises default Wigner–Seitz interpolation and compares it with the native Wannier90 result.

Both contain versioned input provenance and measured execution results. Regenerate outputs in a fresh directory with:

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python ${CLAUDE_SKILL_DIR}/scripts/run_example.py \
  silicon-sp3 --wannier90 wannier90.x --output-dir research/wannier/silicon-example --plot
```

## Constraints

- Supported export: standard folded `*_hr.dat` paired with `*_wsvec.dat`, tested with Wannier90 v4.0.3. Do not apply a folded sidecar to newer already-expanded/weight-applied exports. Mismatched mappings are rejected.
- `--legacy` permits a missing sidecar only when the generating calculation is known to have `use_ws_distance=false`. It conflicts with a sidecar declaring `true`.
- The interpolation checks Hermiticity before removing roundoff; it does not silently repair an invalid model.
- The examples reuse precomputed DFT matrices and do not establish DFT mesh convergence, experimental accuracy or transferability to another pseudopotential.
- Full ab initio Berry/optical responses may require additional position, velocity or spin matrices. Diagonalizing `_hr.dat` alone does not provide all such information.

## References

- N. Marzari and D. Vanderbilt, "Maximally localized generalized Wannier functions for composite energy bands", *Phys. Rev. B* **56**, 12847 (1997). [DOI](https://doi.org/10.1103/PhysRevB.56.12847); [open preprint](https://arxiv.org/abs/cond-mat/9707145).
- I. Souza, N. Marzari, and D. Vanderbilt, "Maximally localized Wannier functions for entangled energy bands", *Phys. Rev. B* **65**, 035109 (2001). [DOI](https://doi.org/10.1103/PhysRevB.65.035109); [open preprint](https://arxiv.org/abs/cond-mat/0108084).
- G. Pizzi et al., "Wannier90 as a community code: new features and applications", *J. Phys.: Condens. Matter* **32**, 165902 (2020). [DOI](https://doi.org/10.1088/1361-648X/ab51ff); [open preprint](https://arxiv.org/abs/1907.09788).

---

**Author:** bowen-bd
**Contact:** [GitHub @bowen-bd](https://github.com/bowen-bd)
