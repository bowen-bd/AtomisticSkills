# NValchemi neighbor-cache handling for 2.0.0

The locked toolkit 0.2.0 can silently omit periodic neighbors when a live
batch changes cell, periodicity or atom partition without changing its total
atom and graph counts. This affects enabled MACE inflight relaxation and can
also affect gradual fixed-batch variable-cell relaxation.

## Release decision

AtomisticSkills constructs every dynamics neighbor hook through
`make_neighbor_list_hook()`. For toolkit 0.2.x only, a small explicit guard
compares cell, PBC and `batch_ptr` values with the previous staging values.
A change requests upstream's complete allocation path, resetting periodic
image ranges, cell-list scratch space, adaptive neighbor capacity and Verlet
references. Fixed-cell steps retain their allocated buffers.

Value comparison is necessary: Warp can update cells without incrementing
PyTorch's tensor version counter. The guard does not edit installed packages
or globally replace the upstream class. Other toolkit versions receive the
native hook; revalidate the lock before retiring this compatibility code.

Disabling inflight alone would leave variable-cell relaxation exposed.
Disabling all affected dynamics would also remove working batch functionality.
Reallocating every step is correct but pays allocation costs even at fixed cell.
The guard keeps the enabled MACE path and pays that cost only when metadata
changes. MatGL inflight remains disabled because the separate inactive-output
repair still needs integration validation. No new relaxation speedup is claimed.

## Exposed paths

| Path | Cache use and 2.0.0 handling |
| --- | --- |
| MACE inflight relaxation | Refills can keep N/B constant; guarded and enabled. |
| Fixed-batch variable-cell relaxation, MACE/MatGL | In-place cell changes are guarded. |
| Fixed-cell relaxation | Guard compares metadata; unchanged allocations are reused. |
| Batch NVE, Nose–Hoover NVT and Langevin NVT | Fixed cell, no refills; use the same guarded constructor. |
| Batch NPT aliases | Cell-changing integrator is selected and its hook is guarded. The existing integration omits initial stress and currently falls back to ASE before advancing the cell; a native NPT trajectory is not certified here. |
| FairChem | `neighbor_config=None`; builds its own graphs and bypasses this hook. |
| Static batched inference | Calls fresh `compute_neighbors`; does not reuse this hook cache. |

## Reproduction through AtomisticSkills

The permanent MACE regression loads `MACE-OMAT-0-small`, supplies four two-atom
Si structures (hcp, displaced hcp, diamond primitive and bcc conventional),
and calls `relax_structure(..., relax_cell=False, max_batch_atoms=4, steps=3)`.
It audits the actual graph presented to MACE against a fresh hook, including
periodic shifts. On the unguarded branch, the live IDs change from `(0, 1)` to
`(1, 2)` and the comparison fails. With the guard, all refill comparisons pass.
The test asserts the inflight backend so a sequential fallback cannot hide it.

See [the regression tests](../../tests/utils/test_nvalchemi_neighbor_cache.py).

## Gradual geometry search

The broader model-free search used 96 starting structures (304 atoms total):
Cu primitive/conventional fcc, Si diamond primitive/bcc conventional/hcp, and
NaCl conventional. Each was scaled by 0.6, 0.75, 0.9, 1, 1.1, 1.25, 1.5 or 2,
with initial shear 0 or 0.35. Cells were converted to standard form. Cutoffs
were 5 and 6 Å, with automatic and `batch_naive` methods. Each trajectory
applied 1% increments of uniform compression, xy/xz/yz shear, or combined shear.
Directed neighbor pairs and image shifts were compared as unordered sets.

A 4,096-neighbor floor isolated geometry allocation defects from capacity
exhaustion. An initial search with default capacity also encountered an
explicit overflow (354 neighbors versus 352 allocated); this guard resets
adaptive capacity as well as geometric scratch space.

The unguarded 0–40% search had **464/820** mismatched batch comparisons. In the matched 0–12% subset, it failed **44/260** comparisons; the guarded hook passed **260/260**. These correspond to 24,960 individual system geometries in the guarded subset. Peak allocations were 59.2 MB unguarded and 73.7 MB guarded. The guard was not exhaustively rerun through 40%.

Small isolated regressions use two identical cells, initial shear 0.35,
and 1% compression increments:

| Cells | Cutoff | Method | First mismatch | Reused / fresh edges |
| --- | ---: | --- | ---: | ---: |
| Cu conventional | 5 Å | automatic | 4% | 396 / 400 |
| Si conventional | 6 Å | automatic | 10% | 984 / 992 |
| NaCl conventional | 5 Å | `batch_naive` | 12% | 504 / 512 |

The broader search also finds shear-only thresholds: with automatic selection,
the first xy/combined-shear mismatches occur at 7% for the 5 Å cutoff, and at
11%/10% respectively for 6 Å. These tests do not certify every possible cell
or trajectory; they establish that gradual changes are reachable failures.

## Fixed-cell cost

Measurements use an NVIDIA A100 80 GB with toolkit 0.2.0, ops 0.4.1,
PyTorch 2.14.1+cu126 and MACE 0.3.16. They are not GB10 measurements. Thirty
warmup calls precede nine rounds; variant order is shuffled each round.
Each hook-only round has 150 calls, and each hook-plus-MACE round has 50.
The reported median wall time includes CPU/GPU synchronization at round
boundaries. Other workloads share the host, so small timing differences may
be noise. The model timings include a forward evaluation, not a complete MD
integrator step.

| Fixed-cell workload | Native hook (µs) | Guard (µs) | Added (µs) | Relative | Reallocate every call (µs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2xSi2 | 5304.0 | 7007.3 | +1703.2 | +32.1% | 8180.4 |
| 32xSi8 | 3844.7 | 4808.0 | +963.2 | +25.1% | 6825.3 |
| 8xSi64 | 3120.9 | 4020.5 | +899.6 | +28.8% | 6113.9 |
| MACE_2xSi2 | 29929.5 | 31654.5 | +1725.0 | +5.8% | 33838.7 |
| MACE_8xSi64 | 31876.4 | 34524.0 | +2647.6 | +8.3% | 35504.5 |

Peak benchmark allocation: 858.6 MB. Disabling inflight adds no hook cost to unchanged fixed-cell paths, but does not fix the demonstrated variable-cell failures.

## Validation and upstream status

- Required NValchemi suite: **77 passed, 7 skipped**; skips require FairChem.
- Required CPU launcher/runtime/project/manifest/CLI suites: **951 passed, 1 skipped**.
- Real MACE inflight regression fails before the change and passes afterward.
- Real NVE and both NVT-family probes keep fresh-consistent graphs. The NPT
  selection test verifies the guarded hook and the existing missing-stress
  fallback; it does not count that fallback as NValchemi NPT execution.
- The separate inactive-output defect in toolkit 0.2.0 remains outside this
  change. MatGL inflight stays disabled, and previous variable-cell speedup
  claims remain withdrawn.

The upstream issue and PR have deliberately **not** been posted. There is no
issue URL to link yet. The affected
[upstream 0.2.0 hook](https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.2.0/nvalchemi/hooks/neighbor_list.py)
and this reproduction document are the references until the release owner
submits the report. Replace that reference with the issue URL when one exists.
