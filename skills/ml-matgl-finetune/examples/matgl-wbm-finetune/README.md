# MatGL Fine-Tuning Example: Softening Dataset

This example demonstrates how to fine-tune the `CHGNet-PES-MatPES-PBE-1M-2026.9` foundation model on a [softening dataset](https://figshare.com/articles/dataset/WBM_high_energy_states/27307776?file=50005317) containing high-energy crystal structures.

## Goal
To document a functional 10-epoch fine-tuning workflow on challenging high-energy VASP data, proving the stability of the decoupled `train_matgl.py` infrastructure directly without wrapping the internal MatGL native components.

## Instructions

### 1. Prepare Training Data
The dataset is processed using the `prepare_matgl_data.py` script. The training data must contain `structure` (pymatgen dict or ASE atoms), `energy` (or `vasp_e`), `forces` (or `vasp_f`), and `stress` (or `vasp_s`).

Because the raw `vasp_s` labels in this dataset are recorded in `kB`, we MUST pass the `--vasp-stress-conversion` flag to multiply them by `-1/1602.1766208` during extraction, converting them to `eV/Å³` and matching standard convention.

```bash
venv/run mlip python skills/ml-matgl-finetune/scripts/prepare_matgl_data.py \
    --data private_data/WBM_high_energy_states.json \
    --model CHGNet-PES-MatPES-PBE-1M-2026.9 \
    --output-dir .agents/test/wbm \
    --val-split 0.1 \
    --vasp-stress-conversion
```

### 2. Run MatGL Trainer
The script structures data internally with `MGLDataset` and natively connects directly to PyTorch Lightning via the `train_matgl.py` protocol. You can boot the loop directly onto GPU without supplementary extraction passes:

```bash
venv/run mlip python skills/ml-matgl-finetune/scripts/train_matgl.py \
    --train-data .agents/test/wbm/train_data.json \
    --val-data .agents/test/wbm/val_data.json \
    --model CHGNet-PES-MatPES-PBE-1M-2026.9 \
    --output-dir .agents/test/wbm/ft \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 8 \
    --freeze-backbone \
    --patience 3
```

The split JSONs (25 MB) and `fine_tuned_model.pth` (4.4 MB) stay in the scratch directory.

### 3. Check the checkpoint round trip

```bash
venv/run mlip python skills/ml-matgl-finetune/examples/matgl-wbm-finetune/verify_checkpoint.py \
    --run-dir .agents/test/wbm/ft --val-data .agents/test/wbm/val_data.json \
    > skills/ml-matgl-finetune/examples/matgl-wbm-finetune/checkpoint_roundtrip.json
```

`train_matgl.py` records the fine-tuned potential's energy for the first validation structure in
`finetune_record.json`. `verify_checkpoint.py` reloads `fine_tuned_model.pth` in a fresh process
with `MatGLWrapper(model_name=...).load_checkpoint()`, predicts the same structure on CPU and
compares the two energies.

## Results (re-run 2026-10, NVIDIA GB10, about 6 min, 1.1 GB GPU)

8,377 training and 931 validation structures (at most 30 atoms each). "Epoch 0" is the zero-shot
model before any update. Only the readout layers are trained (`--freeze-backbone`). Energy errors
are per atom: matgl's loss function already divides by each structure's atom count.

| Validation metric | Epoch 0 (zero-shot) | Epoch 10 |
| :--- | ---: | ---: |
| Total loss | 3.194 | 1.572 |
| Energy MAE (meV/atom) | 241.3 | 250.6 |
| Energy RMSE (meV/atom) | 382.2 | 328.8 |
| Force MAE (meV/Å) | 154.6 | 146.3 |
| Stress MAE (meV/Å³) | 4.97 | 3.66 |

- **The zero-shot energy MAE is confirmed independently.** Evaluating the pretrained model with
  `MatGLWrapper` on all 931 validation structures gives 241.3 meV/atom (median 106.3) against the
  DFT labels. These high-energy states are far from the MatPES training distribution. An earlier
  version of this README reported 31.4 meV/atom because the history callback divided the
  already-per-atom MAE by the mean atom count a second time.
- **Readout-only fine-tuning (10 epochs) lowers the loss, the force MAE (−5 %), the stress MAE
  (−26 %) and the energy RMSE (−14 %).** The energy MAE rises slightly (+9 meV/atom): the loss
  weights forces 10× over energy, and the fit reduces large outliers at the cost of typical errors.
  Unfreezing the backbone, or raising `--energy-weight`, changes that balance.
- **Checkpoint round trip** ([`checkpoint_roundtrip.json`](checkpoint_roundtrip.json)):
  - `fine_tuned_model.pth` loaded with `MatGLWrapper.load_checkpoint()` predicts val[0] (8 atoms)
    at -23.609739 eV, identical (0.0 meV) to the energy recorded at the end of training.
  - The pretrained model gives -24.596069 eV for the same structure, so the reloaded weights are
    the fine-tuned ones (+986 meV shift).
  - The checkpoint's 95 state-dict keys match `MatGLWrapper.model` after `load()`.

## Files
- `finetune_record.json`: training configuration, result and the reference prediction.
- `training_history.json`, `training_history.png`: per-epoch losses and MAEs (energy in meV/atom,
  forces in meV/Å, stress in meV/Å³; epoch 0 is the zero-shot validation).
- `verify_checkpoint.py`, `checkpoint_roundtrip.json`: the round-trip check above.
- `fine_tuned_model.pth` (written to `--output-dir`, not committed): `{"model_state_dict",
  "model_name", "is_fine_tuned", "training_history"}`, the format `MatGLWrapper.save_checkpoint()`
  writes.

---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
