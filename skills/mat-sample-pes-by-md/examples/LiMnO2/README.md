# PES Sampling of LiMnO₂ with MatGL (CHGNet)

This example demonstrates off-equilibrium PES sampling of layered LiMnO₂ (mp-18767) using a high-temperature MD trajectory followed by K-Means clustering of latent crystal features. The sampled structures are used to augment MLIP training datasets with diverse, physically meaningful configurations.

## Method

1. **Structure**: Fetch LiMnO₂ from the Materials Project (`mp-18767`). The 8-atom cell is automatically expanded toward `--target_atoms 50`; the run used a 64-atom supercell.
2. **Potential**: `CHGNet-PES-MatPES-PBE-1M-2026.9` via MatGL (CHGNet foundation model trained on MatPES-PBE).
3. **MD sampling**: NVT ensemble at 2000 K, 2000 steps (5 fs/step → 10 ps total), with Maxwell–Boltzmann velocity initialization.
4. **Clustering**: Per-atom latent features are extracted from the final CHGNet graph-convolution layer at each saved frame. K-Means (k = 10) selects maximally diverse configurations.
5. **Output**: Sampled CIF structures saved to `LiMnO2_matgl_results/` as `sampled_0.cif` … `sampled_9.cif`. The three kept here are renamed by their MD time (e.g. `LiMnO2_MatGL_4.85ps.cif`).

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
venv/run mlip python skills/mat-sample-pes-by-md/scripts/run_sampling.py \
    LiMnO2_initial.cif \
    --model_type matgl --model_name CHGNet-PES-MatPES-PBE-1M-2026.9 \
    --total_steps 2000 --temperature 2000 --ensemble nvt \
    --n_clusters 10 --target_atoms 50 \
    --output_dir LiMnO2_matgl_results
```

## Results (re-run 2026-10, NVIDIA GB10, about 70 s, under 1 GB GPU)

- MD starts at 50 K and the thermostat reaches the 2000 K target by about 2 ps. The mean T over
  the last 5 ps is 2032 K, and E<sub>pot</sub>/atom rises from -22.247 to about -21.97 eV.
- K-Means over the 201 saved frames (128-dimensional CHGNet features) picked frames at
  0.05, 0.35, 2.05, 2.25, 3.60, 4.40, 4.85, 5.20, 6.90 and 7.65 ps. These span the heating ramp
  and the 2000 K plateau.

## Output Files

| File | Description |
|------|-------------|
| `LiMnO2_initial.cif` | Ground-state structure from Materials Project (fetched at runtime) |
| `LiMnO2_matgl_results/input_configs.yaml` | Full simulation parameters for reproducibility (generated at runtime) |
| `LiMnO2_MatGL_{0.35,4.85,7.65}ps.cif` | 3 of the 10 sampled structures (early ramp, plateau, late plateau), kept here |

## Reference

This workflow is based on the dimensionality reduction and stratified sampling strategy described in:

> Qi, J. et al. "Robust training of machine learning interatomic potentials with dimensionality reduction and stratified sampling." *npj Computational Materials* **10**, 43 (2024). https://www.nature.com/articles/s41524-024-01227-4

## 3D Structures

- [LiMnO2_MatGL_0.35ps.cif](LiMnO2_MatGL_0.35ps.cif)
- [LiMnO2_MatGL_4.85ps.cif](LiMnO2_MatGL_4.85ps.cif)
- [LiMnO2_MatGL_7.65ps.cif](LiMnO2_MatGL_7.65ps.cif)
