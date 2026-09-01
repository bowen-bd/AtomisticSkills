# PES Sampling of LiMnO₂ with MatGL (CHGNet)

This example demonstrates off-equilibrium PES sampling of layered LiMnO₂ (mp-18767) using a high-temperature MD trajectory followed by K-Means clustering of latent crystal features. The sampled structures are used to augment MLIP training datasets with diverse, physically meaningful configurations.

## Method

1. **Structure**: Fetch LiMnO₂ from the Materials Project (`mp-18767`). The primitive cell is automatically expanded to ~50 atoms before simulation.
2. **Potential**: `CHGNet-PES-MatPES-PBE-2025.2.10` via MatGL (CHGNet foundation model trained on MatPES-PBE).
3. **MD sampling**: NVT ensemble at 2000 K, 2000 steps (5 fs/step → 10 ps total), with Maxwell–Boltzmann velocity initialization.
4. **Clustering**: Per-atom latent features are extracted from the final CHGNet graph-convolution layer at each saved frame. K-Means (k = 10) selects maximally diverse configurations.
5. **Output**: Sampled CIF structures saved to `LiMnO2_matgl_results/`, named by their MD timestamp (e.g., `LiMnO2_MatGL_8.05ps.cif`).

## Usage

### Step 1: Fetch the structure via MCP

```python
mcp_base_search_materials_project_by_formula(
    formula="LiMnO2",
    output_dir="LiMnO2_matgl_results"
)
# Save mp-18767 as LiMnO2_initial.cif
```

### Step 2: Sample with the unified CLI

```bash
# Env: matgl-agent
conda run -n matgl-agent python skills/mat-sample-pes-by-md/scripts/run_sampling.py \
    LiMnO2_initial.cif \
    --model_type matgl --model_name CHGNet-PES-MatPES-PBE-2025.2.10 \
    --total_steps 2000 --temperature 2000 --ensemble nvt \
    --n_clusters 10 --target_atoms 50 \
    --output_dir LiMnO2_matgl_results
```

## Output Files

| File | Description |
|------|-------------|
| `LiMnO2_initial.cif` | Ground-state structure from Materials Project (fetched at runtime) |
| `LiMnO2_matgl_results/input_configs.yaml` | Full simulation parameters for reproducibility (generated at runtime) |
| `LiMnO2_matgl_results/LiMnO2_MatGL_*.cif` | 3 representative structures included here (full run produces 10) |

## Reference

This workflow is based on the dimensionality reduction and stratified sampling strategy described in:

> Qi, J. et al. "Robust training of machine learning interatomic potentials with dimensionality reduction and stratified sampling." *npj Computational Materials* **10**, 43 (2024). https://www.nature.com/articles/s41524-024-01227-4
