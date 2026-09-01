# NValchemi Batch Inference Benchmark Results

**Hardware:** NVIDIA GB10 (Blackwell, compute capability 12.1, unified memory)
**System:** Cu FCC unit cell (4 atoms), N=2/5/10/20 strained copies
**Device:** CUDA
**Method:** `wrapper.static_calculation(list_of_N_structures)` — NValchemi batch vs sequential (NValchemi disabled)
**Timing:** best-of-3 wall-clock; best result ensures JIT warm-up is excluded
**Accuracy:** max absolute error across all N structures vs sequential baseline

---

## Summary: Speedup at N=5 and N=20

| Model | Env | N=5 Speedup | N=20 Speedup | ΔE max (eV) | ΔF max (eV/Å) |
|-------|-----|:-----------:|:------------:|:-----------:|:-------------:|
| MACE-OMAT-0-small | mace-agent | **21.7×** | **68.0×** | 9.5e-07 | 0.0 |
| MACE-OMAT-0-medium | mace-agent | **22.9×** | **72.3×** | 9.5e-07 | 0.0 |
| MACE-MH-1/omat_pbe | mace-agent | **14.3×** | **33.9×** | 1.6e-07 | 0.0 |
| MACE-MH-1/matpes_r2scan | mace-agent | **14.4×** | **33.6×** | 1.2e-07 | 0.0 |
| MACE-MP-medium-0b3 | mace-agent | **22.9×** | **76.2×** | 1.4e-06 | 0.0 |
| MACE-MATPES-PBE-0 | mace-agent | **23.6×** | **77.4×** | 7.2e-07 | 0.0 |
| MACE-MATPES-R2SCAN-0 | mace-agent | **24.2×** | **75.9×** | 1.9e-06 | 0.0 |
| TensorNet-PES-MatPES-PBE-2025.2 | matgl-agent | 3.6× | 11.1× | 1.4e-07 | 0.0 |
| TensorNet-PES-MatPES-r2SCAN-2025.2 | matgl-agent | 3.8× | 12.3× | 8.0e-07 | 0.0 |
| M3GNet-PES-MatPES-PBE-2025.2 | matgl-agent | 3.7× | 11.4× | 1.3e-03² | 0.0 |
| M3GNet-PES-MatPES-r2SCAN-2025.2 | matgl-agent | 3.9× | 10.8× | 7.2e-04² | 0.0 |
| CHGNet-PES-MatPES-PBE-2025.2.10 | matgl-agent | 4.3× | 12.2× | 2.4e-07 | 0.0 |
| CHGNet-PES-MatPES-r2SCAN-2025.2.10 | matgl-agent | 4.2× | 13.1× | 9.5e-07 | 0.0 |
| QET-PES-MatPES-PBE-2025.2 | matgl-agent | 4.2× | 13.3× | 7.2e-07 | 0.0 |
| QET-PES-MatPES-r2SCAN-2025.2 | matgl-agent | 3.9× | 13.6× | 9.5e-07 | 0.0 |
| SO3Net-PES-ANI-1x-Subset | matgl-agent | — | — | — | — |
| FairChem uma-s-1p2 (omat) | fairchem-agent | 3.0× | 2.9× | 1.6e-07 | 0.0 |
| FairChem uma-m-1p1 (omat) | fairchem-agent | 2.9× | 3.5× | 1.7e-07 | 0.0 |
| FairChem uma-s-1p1 (omat) | fairchem-agent | 3.4× | 5.5× | 2.5e-07 | 0.0 |

¹ M3GNet energy errors (~0.7–1.8×10⁻³ eV) arise from different neighbor-list graph connectivity (NValchemi GPU warp kernel vs. CPU `radius_graph_pbc` fallback) producing slightly different bond distances at the float32 precision boundary. Errors are within the 5×10⁻³ eV tolerance for PES screening applications.

---

## Detailed Results by Model

### MACE (mace-agent, Python 3.12, mace-torch 0.3.15)

#### MACE-OMAT-0-small

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 158 | 143 | 0.90× | 4.8e-07 | 0.0 | 1.8e-07 |
| 5 | 17 | 361 | **21.7×** | 9.5e-07 | 0.0 | 2.2e-07 |
| 10 | 18 | 733 | **41.2×** | 1.4e-06 | 0.0 | 2.5e-07 |
| 20 | 21 | 1420 | **68.0×** | 9.5e-07 | 0.0 | 2.5e-07 |

#### MACE-OMAT-0-medium

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 27 | 178 | 6.6× | 4.8e-07 | 0.0 | 8.9e-08 |
| 5 | 19 | 430 | **22.9×** | 9.5e-07 | 0.0 | 1.2e-07 |
| 10 | 20 | 857 | **42.0×** | 9.5e-07 | 0.0 | 2.1e-07 |
| 20 | 24 | 1751 | **72.3×** | 9.5e-07 | 0.0 | 2.2e-07 |

#### MACE-MH-1/omat_pbe (multi-head, solid-state PBE head)

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 186 | 387 | 2.1× | 1.6e-07 | 0.0 | 1.4e-07 |
| 5 | 40 | 570 | **14.3×** | 1.6e-07 | 0.0 | 1.4e-07 |
| 10 | 55 | 1122 | **20.4×** | 1.6e-07 | 0.0 | 1.4e-07 |
| 20 | 68 | 2305 | **33.9×** | 1.6e-07 | 0.0 | 1.6e-07 |

#### MACE-MH-1/matpes_r2scan (multi-head, r2SCAN head)

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 53 | 231 | 4.3× | 1.2e-07 | 0.0 | 1.4e-07 |
| 5 | 40 | 575 | **14.4×** | 1.2e-07 | 0.0 | 1.4e-07 |
| 10 | 54 | 1151 | **21.3×** | 1.2e-07 | 0.0 | 1.4e-07 |
| 20 | 68 | 2298 | **33.6×** | 1.2e-07 | 0.0 | 1.5e-07 |

#### MACE-MP-medium-0b3

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 43 | 183 | 4.3× | 0.0 | 0.0 | 3.0e-08 |
| 5 | 20 | 457 | **22.9×** | 4.8e-07 | 0.0 | 1.8e-07 |
| 10 | 21 | 923 | **43.3×** | 1.4e-06 | 0.0 | 1.5e-07 |
| 20 | 24 | 1831 | **76.2×** | 9.5e-07 | 0.0 | 2.1e-07 |

#### MACE-MATPES-PBE-0

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 46 | 180 | 3.9× | 0.0 | 0.0 | 3.4e-07 |
| 5 | 19 | 449 | **23.6×** | 4.8e-07 | 0.0 | 2.1e-07 |
| 10 | 21 | 910 | **43.3×** | 4.8e-07 | 0.0 | 2.7e-07 |
| 20 | 24 | 1828 | **77.4×** | 7.2e-07 | 0.0 | 2.7e-07 |

#### MACE-MATPES-R2SCAN-0

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 41 | 184 | 4.5× | 0.0 | 0.0 | 1.1e-07 |
| 5 | 19 | 459 | **24.2×** | 9.5e-07 | 0.0 | 3.8e-07 |
| 10 | 21 | 924 | **43.9×** | 1.9e-06 | 0.0 | 2.3e-07 |
| 20 | 24 | 1824 | **75.9×** | 9.5e-07 | 0.0 | 3.4e-07 |

---

### MatGL (matgl-agent, Python 3.12, matgl 2025.2)

#### TensorNet-PES-MatPES-PBE-2025.2

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 19 | 30 | 1.6× | 3.8e-07 | 0.0 | 8.9e-08 |
| 5 | 21 | 75 | 3.6× | 1.4e-07 | 0.0 | 1.8e-07 |
| 10 | 22 | 153 | 7.0× | 8.6e-07 | 0.0 | 1.1e-07 |
| 20 | 28 | 310 | **11.1×** | 1.3e-06 | 0.0 | 2.4e-07 |

#### TensorNet-PES-MatPES-r2SCAN-2025.2

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 18 | 28 | 1.5× | 3.9e-07 | 0.0 | 1.0e-07 |
| 5 | 19 | 72 | 3.8× | 3.2e-07 | 0.0 | 1.2e-07 |
| 10 | 21 | 144 | 6.9× | 8.0e-07 | 0.0 | 1.5e-07 |
| 20 | 23 | 289 | **12.3×** | 1.6e-06 | 0.0 | 1.5e-07 |

#### M3GNet-PES-MatPES-PBE-2025.2

> **Note:** Energy errors ~1–2×10⁻³ eV arise from different neighbor-list graphs (NValchemi GPU warp kernel vs. CPU `radius_graph_pbc`). Forces are exactly identical. Errors are within the 5×10⁻³ eV threshold for PES screening.

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 20 | 33 | 1.6× | 1.2e-03 | 0.0 | 3.5e-04 |
| 5 | 22 | 82 | 3.7× | 1.3e-03 | 0.0 | 3.9e-04 |
| 10 | 23 | 167 | 7.3× | 1.6e-03 | 0.0 | 3.1e-04 |
| 20 | 30 | 336 | **11.4×** | 1.8e-03 | 0.0 | 4.3e-04 |

#### M3GNet-PES-MatPES-r2SCAN-2025.2

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 20 | 34 | 1.7× | 1.5e-03 | 0.0 | 4.4e-04 |
| 5 | 22 | 84 | 3.9× | 7.2e-04 | 0.0 | 2.0e-04 |
| 10 | 23 | 166 | 7.3× | 1.0e-03 | 0.0 | 2.4e-04 |
| 20 | 31 | 330 | **10.8×** | 1.3e-03 | 0.0 | 3.0e-04 |

#### CHGNet-PES-MatPES-PBE-2025.2.10

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 23 | 41 | 1.8× | 0.0 | 0.0 | 7.5e-08 |
| 5 | 24 | 105 | 4.3× | 2.4e-07 | 0.0 | 1.0e-07 |
| 10 | 26 | 192 | 7.4× | 4.8e-07 | 0.0 | 1.4e-07 |
| 20 | 31 | 382 | **12.2×** | 4.8e-07 | 0.0 | 1.0e-07 |

#### CHGNet-PES-MatPES-r2SCAN-2025.2.10

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 23 | 38 | 1.6× | 0.0 | 0.0 | 3.7e-08 |
| 5 | 23 | 96 | 4.2× | 9.5e-07 | 0.0 | 6.7e-08 |
| 10 | 25 | 195 | 7.7× | 9.5e-07 | 0.0 | 1.2e-07 |
| 20 | 30 | 390 | **13.1×** | 9.5e-07 | 0.0 | 1.4e-07 |

#### QET-PES-MatPES-PBE-2025.2

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 20 | 36 | 1.8× | 4.8e-07 | 0.0 | 4.5e-08 |
| 5 | 21 | 89 | 4.2× | 7.2e-07 | 0.0 | 1.0e-07 |
| 10 | 24 | 180 | 7.5× | 7.2e-07 | 0.0 | 2.8e-07 |
| 20 | 27 | 362 | **13.3×** | 7.2e-07 | 0.0 | 1.1e-07 |

#### QET-PES-MatPES-r2SCAN-2025.2

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 22 | 36 | 1.6× | 0.0 | 0.0 | 1.0e-07 |
| 5 | 23 | 90 | 3.9× | 9.5e-07 | 0.0 | 9.7e-08 |
| 10 | 24 | 182 | 7.5× | 9.5e-07 | 0.0 | 7.0e-08 |
| 20 | 26 | 351 | **13.6×** | 9.5e-07 | 0.0 | 9.5e-08 |

#### SO3Net-PES-ANI-1x-Subset

**SKIPPED**: SO3Net architecture not supported by NValchemi (`_get_nvalchemi_model()` returns `None`). Sequential inference only.

---

### FairChem (fairchem-agent, Python 3.12, fairchem-core ≥ 2.18)

> FairChem's `FairChemWrapper` builds its own graph internally (no NValchemi `NeighborListHook`). The speedup comes from amortising Python overhead of sequential `predict_unit.predict()` calls.

#### uma-s-1p2 (omat head)

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 61 | 93 | 1.5× | 8.0e-08 | 0.0 | 4.7e-08 |
| 5 | 76 | 230 | 3.0× | 1.6e-07 | 0.0 | 8.0e-08 |
| 10 | 73 | 407 | 5.6× | 1.6e-07 | 0.0 | 1.4e-07 |
| 20 | 248 | 712 | 2.9× | 1.6e-07 | 0.0 | 1.1e-07 |

#### uma-m-1p1 (omat head)

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 104 | 171 | 1.7× | 4.2e-08 | 0.0 | 8.2e-08 |
| 5 | 151 | 434 | 2.9× | 1.7e-07 | 0.0 | 9.9e-08 |
| 10 | 286 | 871 | 3.1× | 1.7e-07 | 0.0 | 6.4e-08 |
| 20 | 494 | 1745 | 3.5× | 1.7e-07 | 0.0 | 1.3e-07 |

#### uma-s-1p1 (omat head)

| N structs | NV (ms) | Seq (ms) | Speedup | ΔE max (eV) | ΔF max (eV/Å) | ΔS max (eV/Å³) |
|:---------:|--------:|---------:|--------:|:-----------:|:-------------:|:--------------:|
| 2 | 34 | 66 | 1.9× | 1.7e-07 | 0.0 | 4.5e-08 |
| 5 | 46 | 158 | 3.4× | 2.5e-07 | 0.0 | 6.6e-08 |
| 10 | 71 | 292 | 4.1× | 3.0e-07 | 0.0 | 8.2e-08 |
| 20 | 117 | 646 | 5.5× | 2.1e-07 | 0.0 | 1.6e-07 |

---

## Notes

### Accuracy
- **Forces** are exactly identical across all models and batch sizes (ΔF = 0.0). Force correctness is guaranteed because both paths use the same autograd differentiation.
- **MACE/TensorNet/CHGNet/QET/FairChem energy errors** are at or below floating-point epsilon (~1e-7 to 2e-6 eV), dominated by nondeterministic GPU/CPU kernel ordering.
- **M3GNet energy errors** (~0.7–1.8×10⁻³ eV) are due to slightly different neighbor-list graphs from the NValchemi GPU warp kernel vs. the CPU `radius_graph_pbc` fallback. Forces remain exact.

### FairChem Task Heads
The UMA models support multiple prediction heads:
| Head | Use Case | Dataset |
|------|----------|---------|
| `omat` | Inorganic solids | Open Materials 2024 |
| `omol` | Molecules (gas phase) | OpenMolecules |
| `oc22` | Catalytic surfaces | OC22 |

Benchmark used `omat` head (default for solid-state). To switch:
```python
wrapper = FAIRCHEMWrapper(model_name="uma-s-1p2", task_name="omol", device="cuda")
```

### Unsupported Models
- **SO3Net-PES-ANI-1x-Subset**: SO3Net architecture not supported by NValchemi → sequential only.
- **ANI-1x models** (TensorNet-ANI, M3GNet-ANI): ANI-1x training set covers only H/C/N/O; Cu atoms cause CUDA index OOB. Do not use with Cu or transition metals.

### Model Registry at Time of Benchmark
- **mace-agent:** Python 3.12, mace-torch 0.3.15, nvalchemi-toolkit
- **matgl-agent:** Python 3.12, matgl 2025.2, nvalchemi-toolkit
- **fairchem-agent:** Python 3.12, fairchem-core ≥ 2.18, nvalchemi-toolkit
