# Example: ZrS2 1T monolayer

End-to-end run of the `mat-epw-mobility` pipeline on a ZrS2 monolayer (1T
phase), the system the skill was validated on. It exercises every step, reports
the expected value at each, and validates the EPW-interpolated electron-phonon
coupling against the DFPT ground truth.

## System

- ZrS2 1T monolayer, hexagonal, a = 3.669 A, ~30 A vacuum.
- Structure: [zrs2_monolayer.cif](zrs2_monolayer.cif)
- Pseudopotentials: ONCV SG15 PBE v1.2 (`Zr_ONCV_PBE-1.2.upf`, `S_ONCV_PBE-1.2.upf`).
- Validated build: Quantum ESPRESSO 7.4.1 / EPW 5.8.1.

## Goal

1. Cross-validate the EPW Wannier interpolation of $|g(k, q, \nu)|$ against a
   direct DFPT reference at a finite q (the primary correctness check).
2. Produce the SERTA phonon-limited electron mobility $\mu(T)$ with the 2D
   Frohlich kernel enabled.

## Steps

Follow the numbered instructions in [`../../SKILL.md`](../../SKILL.md), using the
decks in [`../../resources/inputs/`](../../resources/inputs/). The transport path
is 1 -> 2 -> 4 -> 6 -> 7 -> 9; the $|g|$ validation adds steps 5 and 8.

## Expected values per step

| Step | Quantity | Expected (ZrS2, QE 7.4.1) |
|---|---|---|
| 1 SCF | total energy / VBM | -136.1241 Ry / -6.4435 eV |
| 2 NSCF | gap (CBM) | 1.1724 eV (CBM -5.2711 eV) |
| 3 DFPT-Gamma | $\varepsilon_\infty^{xx}$ / $Z^*_{xx}(\mathrm{Zr})$ | 2.973 / +7.003 |
| 4 DFPT uniform-q | irreducible q / phonon range | 7 / 16.9-337.6 cm^-1 (no imaginary) |
| 7 Wannierize | `num_wann` / $\Omega_I$ / $\Omega_{total}/\Omega_I$ | 9 / 18.5 A^2 / ~1.05 |
| 9 SERTA | $\mu_{xx}$ (100x100 mesh, 300 K, 1e13 cm^-2) | ~195 cm^2/V/s |

## Primary validation: EPW-interpolated |g| vs DFPT reference

At k = M, q = (0.005, 0, 0) crystal, comparing the DFPT reference (step 5,
`parse_prt.py`) to the EPW Wannier interpolation (step 8, `parse_epw_prtgkk.py`):

| Metric | Step 5 (DFPT) | Step 8 (EPW) | Difference |
|---|---|---|---|
| Rank-1 $\|g\|$ | 1909.7 meV | 1941.6 meV | 1.67 % |
| $\sum \|g\|^2$ over top-4 $\omega$ modes | 8744.0 meV^2 | 8748.7 meV^2 | 0.054 % |

The 1.67 % rank-1 offset is a gauge rotation within the near-degenerate LO/TO
subspace (the two modes lie within ~0.1 meV). The gauge-invariant sum
$\sum |g|^2$ agrees to **0.054 %**, confirming that the Wannier interpolation
reproduces the DFPT electron-phonon vertex. Validation passes on the
gauge-invariant sum, not on the raw per-mode value.

## Literature context for the mobility

Published deformation-potential-theory (DPT) studies report an electron mobility
of order $\sim 1.2 \times 10^3$ cm^2/V/s for ZrS2 monolayer at 300 K (roughly 4x
that of monolayer MoS2). The SERTA value here (~195 cm^2/V/s) is deliberately
lower and **not directly comparable**: DPT accounts only for acoustic
deformation-potential scattering, whereas this pipeline includes the polar-optical
(2D Frohlich) scattering that dominates in polar monolayers and strongly
suppresses $\mu$ (see Verdi & Giustino 2015 and Sohier, Calandra & Mauri 2016 in
[`../../SKILL.md`](../../SKILL.md#references)). A like-for-like literature
comparison must match the scattering model, the transport equation (SERTA vs
iterative BTE), and the carrier density. The rigorous correctness check for this
skill is therefore the DFPT-vs-EPW $|g|$ benchmark above, which isolates the
electron-phonon machinery from those modelling choices.

> [!NOTE]
> Artifacts (`.save` trees, `dvscf`, checkpoints) are not committed here; a full
> production run of step 9 can write >100 GB of interpolated `|g|` to disk.

## 3D Structures

- [zrs2_monolayer.cif](zrs2_monolayer.cif) — ZrS2 1T monolayer primitive cell.
