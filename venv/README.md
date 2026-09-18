# Python environments

Three uv projects replace the per-capability conda environments for everything
except the generative stack. Each is a complete uv project with its own
`pyproject.toml`, `uv.lock` and resolver boundary.

| uv project | Purpose | Accelerator |
| :--- | :--- | :--- |
| `venv/cpu` | Materials, chemistry, drug discovery and analysis | none |
| `venv/mlip` | MACE and MatGL, on top of the CPU stack | torch 2.14, CUDA |
| `venv/fairchem` | FairChem | torch 2.10, CUDA |

Run a skill script by naming its project; no activation step is involved, and
uv builds the environment from the lock on first use:

```bash
uv run --project venv/cpu   python skills/<skill>/scripts/<script>.py ...
uv run --project venv/mlip  python skills/<skill>/scripts/<script>.py ...
```

Use the project the skill declares in its frontmatter. Do not assume support
because two projects happen to contain the same package.

## Why three, and not one

The boundaries are forced by the dependency graph, not chosen:

- **`mlip` and `fairchem` cannot merge.** `mace-torch` pins `e3nn==0.4.4`;
  `fairchem-core` requires `e3nn>=0.5`. There is no resolution, on any
  architecture.
- **`cpu` exists so the common case carries no torch.** It covers the great
  majority of skill scripts and stays small.

## Both architectures

Every project resolves for `linux/x86_64` and `linux/aarch64`, and torch ships
CUDA wheels for both, so GPU work runs on ordinary clusters and on GB10-class
hardware from the same lock. Two packages genuinely have no aarch64 wheel and
carry markers rather than being dropped for everyone:

- `pymol-open-source` — x86_64 only, and only as a pre-release for 3.12
- `scine-utilities` / `scine-readuct` — x86_64 only, so the ORCA skills are
  x86_64 only (they need a user-supplied ORCA binary regardless)

## The fairchem override

`fairchem-core` declares a torch range the validated environment deliberately
violates, because torch 2.8 has no sm_121 build. That is recorded as an
`override-dependencies` entry with the reason beside it, rather than being
hidden inside a frozen package list. Do not "fix" it by relaxing the pin.

## The generative stack is not here

`adit`, `diffcsp` and `mattergen` stay on their existing container path, for
reasons that are not stylistic:

- `mattergen` hard-pins `torch==2.2.1+cu118` and `torchvision==0.17.1+cu118` —
  local-version wheels that never existed for aarch64
- its PyTorch Geometric extensions have no wheel for that combination, so they
  are compiled from source
- ADiT and DiffCSP++ are not published packages at all

See `docker/README.md` for that path.
