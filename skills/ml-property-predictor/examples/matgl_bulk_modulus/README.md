# MatGL Bulk Modulus Property Predictor Tutorial

This directory demonstrates `skills/ml-property-predictor/scripts/train_matgl_property.py`. It
fine-tunes the pretrained MatGL CHGNet potential `CHGNet-PES-MatPES-PBE-1M-2026.9` to predict the
**bulk modulus** K_VRH, an intensive property, from crystal structures.

## Contents
- `fetch_bulk_modulus.py`: builds the dataset `.agents/test/mp_bulk_modulus.json`. It holds 500
  non-deprecated Materials Project elasticity entries with at most 12 sites and
  0 < K_VRH < 600 GPa, taken in MP-ID order.
- `run_matgl.py`: trains for 25 epochs (batch 16, lr 1e-3, full model) in the `mlip` environment.
  It then reloads the saved model with `MatGLWrapper` and predicts the first five structures. Finally
  it deletes intermediate Lightning checkpoints and logs.

## Running the Example
From the repository root:

```bash
venv/run cpu python skills/ml-property-predictor/examples/matgl_bulk_modulus/fetch_bulk_modulus.py \
    --output .agents/test/mp_bulk_modulus.json          # needs MP_API_KEY
venv/run cpu python skills/ml-property-predictor/examples/matgl_bulk_modulus/run_matgl.py
```

`run_matgl.py` restarts itself in the `mlip` environment and writes the best model to
`matgl_results/matgl_model/` (`model.json`, `model.pt`, `state.pt`). Model checkpoints are not
kept in this example folder. The CHGNet head is extensive, so for an intensive target
the script divides the output by the number of atoms, both in training and in the reload check.

## Results (NVIDIA GB10, about 3 min, about 1 GB GPU)

The data are split 400/50/50 (train/validation/test) with seed 42.

| Model | Best validation MAE (GPa) |
| :--- | ---: |
| CHGNet-PES-MatPES-PBE-1M-2026.9, full fine-tune, lr 1e-3 (`run_matgl.py`, epoch 24) | 69.7 |
| same, lr 1e-4 | 70.2 |
| same, `--freeze_backbone` | 74.3 |
| **Baseline: predict the training-set mean (118.4 GPa)** | **56.9** |

Reload check on the first five structures (target → prediction, GPa): 1.97 → 1.86, 166.98 → 124.48,
1.87 → 2.01, 365.05 → 183.98, 39.59 → 34.09.

**This short run does not beat the mean-value baseline.** With 400 training structures and
25 epochs, the model captures the trend (soft vs stiff solids) but compresses the range: 365 GPa is
predicted as 184 GPa. Treat this example as a check that the training, saving and reload path works,
not as a usable bulk-modulus model. For a usable model, train longer on more data and compare
against the same baseline. Examples of more data are the full MP elasticity set and Matbench
`matbench_log_kvrh`, which uses log K.
