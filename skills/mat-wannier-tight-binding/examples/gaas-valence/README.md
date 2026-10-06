# GaAs valence MLWFs

## Goal and provenance

Reproduce the four valence Wannier functions of official Wannier90 tutorial 1 using its precomputed DFT overlaps and projections. This is a tutorial reproduction, not a new DFT calculation or a mesh-converged prediction.

The unmodified `.amn` and `.mmn` files come from [Wannier90 v4.0.3 tutorial01](https://github.com/wannier-developers/wannier90/tree/v4.0.3/tutorials/tutorial01), commit `971c8b64bfe703c747a52655e1d59936597fe8ea`. [provenance.json](provenance.json) records their SHA-256 hashes and the execution settings. Upstream licensing is retained in [LICENSE.wannier90](../LICENSE.wannier90).

- Zincblende GaAs, FCC primitive cell; actual conventional lattice constant **5.680188 Å** (10.734 bohr).
- Four occupied bands and four Wannier functions; no disentanglement.
- Full Gamma-centered 2×2×2 mesh, eight points, no shift.
- The upstream `.win` declares `As:sp3`; this example retains the supplied precomputed `.amn`. Editing `.win` projectors alone does not regenerate the projections. The online tutorial describes its starting guess as bond-centered Gaussians; use the archived matrix provenance when reproducing this dataset.
- We enable localization convergence explicitly: `conv_window=3`, `conv_tol=1e-10`, `num_iter=200` as an upper limit. The original tutorial uses 20 fixed iterations.

## Run

From the repository root, build Wannier90 if needed as described in [the skill](../../SKILL.md), then use an empty output directory:

```bash
venv/run cpu python skills/mat-wannier-tight-binding/scripts/run_example.py gaas-valence \
  --wannier90 "$PWD/.agents/test/wannier90-build/install/bin/wannier90.x" \
  --output-dir "$PWD/.agents/test/gaas-wannier-example"
```

For an existing installation, replace `--wannier90` with its executable path. The runner copies the inputs, executes Wannier90 and parses the final state with `--require-converged`. Inspect `gaas.wout`, `analysis/wout_summary.json`, `execution.json` and `input_configs.yaml`. No external DFT executable is needed. The fixture [gaas.wout](gaas.wout) records the verified v4.0.3 run.

## What the result validates

| Quantity | Marzari–Vanderbilt Table II, 2×2×2 | Official v4.0.3 regression output | Verified run |
|---|---:|---:|---:|
| Total spread, Å² | 4.409 | 4.466881009 | 4.466881009 |
| Omega I, Å² | 3.898 | 3.956862987 | 3.956862987 |
| Omega OD, Å² | 0.503 | 0.501987973 | 0.501987973 |
| Omega D, Å² | 0.0078 | 0.008030049 | 0.008030049 |
| Bond-center fraction beta | 0.602 | approximately 0.610 | approximately 0.610 |

The native convergence criterion is satisfied. Each final WF has spread approximately **1.11672025 Å²**. For the bond from Ga at the origin, beta is the Ga-to-center distance divided by the Ga–As bond distance: approximately `0.866253 / 1.420047 = 0.610`.

The total spread agrees with the current tutorial/regression dataset, but differs from the original paper by **0.057881009 Å² (1.31%)**. These are distinct numerical references. Matching all original DFT settings would be necessary to claim reproduction of the paper's value; no such claim is made here. The paper's mesh-dependence table also shows why a 2×2×2 spread should not be presented as mesh-converged.

## References

1. N. Marzari and D. Vanderbilt, *Phys. Rev. B* **56**, 12847 (1997), Table II. [DOI](https://doi.org/10.1103/PhysRevB.56.12847); [open paper](https://arxiv.org/pdf/cond-mat/9707145).
2. [Official tutorial 1 solution](https://wannier90.readthedocs.io/en/latest/tutorial_solutions/tutorial_solution_1/), with the rounded values and centre interpretation.
3. [Versioned v4.0.3 regression output](https://raw.githubusercontent.com/wannier-developers/wannier90/v4.0.3/test-suite/tests/testw90_example01/benchmark/gaas.wout), source of the full-precision benchmark column.
4. G. Pizzi et al., *J. Phys.: Condens. Matter* **32**, 165902 (2020), software reference. [DOI](https://doi.org/10.1088/1361-648X/ab51ff).
