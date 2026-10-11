# MLIP r2SCAN Foundation Model Benchmarking

This directory contains a complete example workflow demonstrating how to extract baseline DFT data from the Materials Project API and rapidly evaluate the fidelity of state-of-the-art Machine Learning Interatomic Potentials (MLIPs).

## Workflow Overview

1. **`fetch_r2scan.py`**: A customized Materials Project API ingestion script that combines the `thermo` and `summary` endpoints to seamlessly pair structural geometries with their uncorrected `r2SCAN` energy computations.
2. **`run_benchmark.py`**: The main execution engine under `../scripts/` orchestrates `load_wrapper` instantiations to perform inference testing on identical structures.
3. **`plot_benchmark.py`**: Renders numerical MAE/RMSE results mapped to visual `.png` parity distributions out of the resulting prediction data arrays.

## MatPES-r2SCAN Foundation Benchmarks

We ran this workflow on 100 r2SCAN structures from Materials Project (fetched 2026-10-10, f-block
excluded) against three foundation potentials trained on MatPES-r2SCAN. All three use the same
dataset, on an NVIDIA GB10 (mlip env: matgl 4.1.0, mace-torch 0.3.16):

| Framework | Pre-Trained Foundation Potential Model | Energy MAE (meV/atom) | Energy RMSE (meV/atom) |
| --- | --- | ---: | ---: |
| **MACE**  | `MACE-MATPES-R2SCAN-0` | **73.7** | 179.0 |
| **MatGL** | `CHGNet-PES-MatPES-r2SCAN-1M-2026.9` | **77.5** | 176.7 |
| **MatGL** | `TensorNet-MatPES-r2SCAN-v2025.1-PES` (loads `TensorNet-PES-MatPES-r2SCAN-2025.2`) | **88.3** | 169.7 |

MP's r2SCAN calculations use different settings from MatPES, so these MAEs include that offset.
The RMSE is dominated by a few outliers (see the parity plots).

*(Note: Lanthanide and Actinide elements are programmatically excluded in the fetch script to avoid global potential correlation anomalies.)*

## Running the Benchmarks

You can fully reproduce these findings by performing the following execution steps from the project root.

1. **Fetch Target Dataset**
```bash
venv/run cpu python skills/ml-mlip-benchmark/examples/fetch_r2scan.py
```

2. **Evaluate Foundation Models**
Execute inference testing across each model in their respective environments:

```bash
# MACE Potential Benchmark (Env: mlip)
venv/run mlip python skills/ml-mlip-benchmark/scripts/run_benchmark.py \
    --model MACE-MATPES-R2SCAN-0 --backend mace \
    --data_path r2scan_data.json \
    --output research/2026-03-03_r2SCAN_benchmark/mace_results.json

# CHGNet Potential Benchmark (Env: mlip)
venv/run mlip python skills/ml-mlip-benchmark/scripts/run_benchmark.py \
    --model CHGNet-PES-MatPES-r2SCAN-1M-2026.9 --backend matgl \
    --data_path r2scan_data.json \
    --output research/2026-03-03_r2SCAN_benchmark/chgnet_results.json

# TensorNet Potential Benchmark (Env: mlip)
venv/run mlip python skills/ml-mlip-benchmark/scripts/run_benchmark.py \
    --model TensorNet-MatPES-r2SCAN-v2025.1-PES --backend matgl \
    --data_path r2scan_data.json \
    --output research/2026-03-03_r2SCAN_benchmark/tensornet_results.json
```

3. **Generate Parity Plots**
```bash
venv/run cpu python skills/ml-mlip-benchmark/scripts/plot_benchmark.py \
    --results research/2026-03-03_r2SCAN_benchmark/mace_results.json \
    --output_dir skills/ml-mlip-benchmark/examples/
# (Repeat for the matgl json outputs)
```
