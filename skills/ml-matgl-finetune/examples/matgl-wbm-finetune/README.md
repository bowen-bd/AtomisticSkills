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

The split JSONs (25 MB) and `fine_tuned_model.pth` (4.4 MB) stay in the scratch directory. Only
`finetune_record.json`, `training_history.json` and `training_history.png` are kept here.

## Results (re-run 2026-10, NVIDIA GB10, about 6 min, 1.4 GB GPU)

8,377 training and 931 validation structures (at most 30 atoms each). "Epoch 0" is the zero-shot
model before any update; only the readout layers are trained (`--freeze-backbone`).

| Validation metric | Epoch 0 (zero-shot) | Epoch 10 |
| :--- | ---: | ---: |
| Total loss | 3.194 | 1.550 |
| Energy MAE (meV/atom) | 31.4 | 32.7 |
| Force MAE (meV/Å) | 154.6 | 145.9 |
| Stress MAE (meV/Å³) | 4.97 | 3.66 |

The loss weights forces 10× over energy, so the readout-only fine-tune trades a small amount of
energy accuracy (+1.3 meV/atom) for lower force (−6 %) and stress (−26 %) errors on these
high-energy states. Unfreezing the backbone, or raising the energy weight, changes that balance.

## Expected Outputs
Artifacts of the run above (9,308 structures, 90/10 random split):

- `fine_tuned_model.pth` (written to `--output-dir`, not committed): The natively wrapped checkpoint perfectly mapped back to standalone `MatGLWrapper.load_checkpoint()` logic.
- `training_history.json`: Parsed loss and MAE metrics dynamically tracked per-epoch independent of DGL dependencies.
- `training_history.png`: Standardized loss degradation plot identical to the original wrapper output format verifying stability.
- `finetune_record.json`: Summary configuration defining model paths and basic convergence variables.

---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
