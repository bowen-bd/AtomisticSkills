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

## Results (NVIDIA GB10, about 3 min per run, about 1 GB GPU)

The data are split 400/50/50 (train/validation/test) with seed 42. Errors are averaged over
structures, and the best-validation epoch is saved.

| Model | Best validation MAE (GPa) |
| :--- | ---: |
| CHGNet-PES-MatPES-PBE-1M-2026.9, full fine-tune, lr 1e-3 (`run_matgl.py`, epoch 25) | **20.8** |
| same, lr 1e-4 (epoch 18) | 21.0 |
| same, `--freeze_backbone` (epoch 25) | 24.7 |
| Baseline: predict the training-set mean (118.4 GPa) | 56.9 |

The saved lr 1e-3 model was also checked after reloading with `MatGLWrapper`:
- its validation MAE reproduces at 20.81 GPa;
- on the 50 held-out test structures, which training never sees, it scores **19.0 GPa MAE and
  R² = 0.88**, against 62.1 GPa for the mean-value baseline.

Reload check on the first five structures (target → prediction, GPa): 1.97 → 0.98,
166.98 → 167.04, 1.87 → 13.48, 365.05 → 290.33, 39.59 → 37.78.

The pretrained CHGNet representation transfers well to bulk modulus, even with 400 training
structures and 25 epochs. Full fine-tuning beats the frozen backbone by about 4 GPa. Very soft
(< 5 GPa) and very stiff (> 300 GPa) solids carry the largest relative errors.

**Correction:** an earlier version of this example reported about 70 GPa, worse than the
baseline. That came from a batching bug in `train_matgl_property.py`. Dividing the per-structure
CHGNet output (shape [B]) by a [B, 1] atom-count tensor broadcast to [B, B], so the loss mixed
predictions across structures. `tests/matgl/test_matgl_property_batching.py` now compares batched
predictions and errors with one-structure-at-a-time evaluation.
