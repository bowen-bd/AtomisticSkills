# CT-SCD_matbench Tutorial

This directory contains a small example showing how to fine-tune the pretrained `ct-scd-amp` checkpoint on the Matbench `MBgap` task using the native `SelfConditionedDenoisingAtoms/train.py` entrypoint and the upstream `configs/finetune_matbench.yaml` recipe.

This example mirrors the intended upstream flow:

```bash
python train.py --conf configs/finetune_matbench.yaml --load-hf ct-scd-amp --job-id CT-SCD_matbench_fold0
```

## Important Availability Note

The public `SelfConditionedDenoisingAtoms/README.md` explicitly notes that `configs/finetune_matbench.yaml` depends on the unreleased `StructureCloud` utilities. Treat this example as the correct pattern for checkouts where the Matbench dataset wrapper is available, not as a guaranteed public quickstart.

## Contents

1. `run_ct_scd_matbench.py`: a wrapper script that locates the `SelfConditionedDenoisingAtoms` checkout, runs the upstream `train.py` there with its own interpreter, and runs a safe smoke test by default.

## Running the Example

Run it in the `scd` environment from the AtomisticSkills checkout; `venv/run`
creates the environment on first use (x86_64 Linux):

```bash
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --dry-run
```

Without `--dry-run` it launches a short smoke-test training run:

```bash
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py
```

Use `--dry-run` first to confirm the resolved config, fold, and dataset wrapper without launching training:

```bash
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --dry-run --fold 2
```

Before launching, check GPU availability and current usage. As with the QM9 example, preferring live stdout over buffered wrappers is general guidance for SCD runs, not just this script:

```bash
nvidia-smi
```

On shared machines, prefer a GPU with no active compute job and low memory usage. Do not assume that a low-utilization GPU is free if another process already has memory allocated.

To run on one selected GPU:

```bash
venv/run scd env WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python -u skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --num-steps 2 --val-interval 1
```

To run on all selected visible GPUs:

```bash
venv/run scd env WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3 python -u skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --num-steps 2 --val-interval 1 --use-all-visible-gpus
```

The wrapper defaults to a single visible GPU unless `--use-all-visible-gpus` is requested.

For smoke tests without a live W&B session, pass:

```bash
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --wandb-mode offline
```

For W&B online mode, first log in from the `scd` environment:

```bash
venv/run scd wandb login
```

Then launch without `WANDB_MODE=offline`, or set `WANDB_MODE=online` explicitly.

By default this example performs a short smoke test with `--num-steps 100`. To launch the full upstream schedule instead:

```bash
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --full-run
```

For the quickest smoke tests, consider copying `configs/finetune_matbench.yaml` into a task-specific smoke config and disabling expensive reporting such as `parity_plot` if you add it. Short `max_steps` runs can still spend significant time in evaluation or plotting callbacks.

The example uses Matbench fold `0` by default. To change folds:

```bash
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --fold 1
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --fold 2
```

## Other Matbench Properties

In the current upstream recipe, `finetune_matbench.yaml` is specifically wired for `dataset: MBgap`, and `dataset_arg` selects the fold rather than the property.

To train on another Matbench property:

1. create a sibling dataset wrapper in `SelfConditionedDenoisingAtoms/data/datasets/matbench.py` similar to `mbench_gap`, but pointing to another task from `MBDataset_base.avail_tasks`
2. export that new wrapper from `SelfConditionedDenoisingAtoms/data/datasets/__init__.py`
3. copy `configs/finetune_matbench.yaml` into a new task-specific config YAML
4. run this example script with `--dataset-class <NewWrapper>` and `--config <new_config.yaml>`

For example, once a new wrapper is exported:

```bash
venv/run scd python skills/ml-property-predict-scd/examples/CT-SCD_matbench/run_ct_scd_matbench.py --dataset-class MBdielectric --config configs/finetune_matbench_dielectric.yaml
```

## Notes

- Actual training requires a CUDA-visible GPU. On CPU-only hosts the wrapper now exits early with a clear message instead of letting `train.py` fail later inside PyTorch Lightning.
- If the default Matplotlib config directory is not writable, the wrapper automatically uses a temporary `MPLCONFIGDIR`.
- This example follows the upstream material finetuning settings: `noise_in_loader=True`, `allow_periodic=True`, and `set_head_agg: mean`.
- Runs in the `scd` environment. The upstream checkout is found at `--repo-root`, `$SCD_REPO_DIR`, or `~/.cache/atomisticskills/SelfConditionedDenoisingAtoms` (see the skill's First Checks).
- Training outputs are written under `SelfConditionedDenoisingAtoms/experiments/<job_id>`.
- The W&B project is derived by `train.py` from the dataset class name, so the default `MBgap` run appears under `SCD_bench_MBgap`.
- On this public checkout, the Matbench dataset path still depends on `StructureCloud`, so W&B login alone is not enough to make the example runnable.
- As with QM9, live stdout is the best way to distinguish real startup work from a genuine hang.
