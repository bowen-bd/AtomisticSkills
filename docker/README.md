# Container images for the MCP servers

These images exist so that installing the Claude Code plugin yields **working**
MCP servers. Without them the plugin registers ten servers whose `command`
points at conda environments totalling roughly 58 GB that the user has almost
certainly not built, Claude Code silently skips every server that fails to
start, and the skills appear to install fine while quietly doing nothing.

`docker/images.json` is the single source of truth. Adding or moving a server
means editing that file and re-running the renderers; the Dockerfiles, the CI
matrix, the in-image server table and the plugin's `mcpServers` block are all
projections of it.

## The four images

| Image | Servers | Python | torch | Platforms |
| :--- | :--- | :--- | :--- | :--- |
| `atomisticskills-lightweight` | base, drugdisc, smol, atomate2 | 3.11 | none | amd64 + arm64 |
| `atomisticskills-mace` | mace, matgl | 3.12 | 2.12.0 | arm64 |
| `atomisticskills-fairchem` | fairchem | 3.12 | 2.10.0 | arm64 |
| `atomisticskills-generative` | adit, diffcsp, mattergen | 3.10 | 2.9.1+cu130 | arm64 |

Ten servers in four images, and the boundaries are forced rather than chosen:

- **`mace` and `fairchem` can never share an image.** `mace-torch 0.3.15` pins
  `e3nn==0.4.4`; `fairchem-core` requires `e3nn>=0.5`. There is no resolution.
- **`lightweight` is one merged environment.** Its four source environments are
  the only group whose dependency union resolves cleanly — verified with
  `uv pip compile --python-version 3.11`, 175 packages. Merging them keeps the
  CPU-only image small and genuinely multi-arch.
- **The GPU images keep their environments separate**, installed side by side
  under their original names. This is deliberate: see *Why lockfiles* below.

## Why lockfiles, not specs

The MLIP and generative environments **cannot be reproduced from a declarative
spec**, because several of them deliberately violate their own declared
constraints:

```
$ conda run -n fairchem-agent pip check
fairchem-core 2.19.0 has requirement torch~=2.8.0, but you have torch 2.10.0.
```

torch 2.8 ships no sm_121 / aarch64 / CUDA 13 build, so a newer torch was
force-installed. Any resolver asked to honour `torch~=2.8.0` on this platform
fails — which is exactly what happens if you try to solve these environments,
and why the yaml files carry "manual installation required" comments.

So the GPU images install from exact locks with `--no-deps`:

- `conda-envs/<env>/lock/conda-<subdir>.txt` — `conda list --explicit --md5`,
  exact URLs with checksums
- `conda-envs/<env>/lock/pip-<subdir>.txt` — pinned pip layer only

Regenerate them from known-good environments with:

```bash
# Env: base-agent
python docker/export_locks.py                    # all environments
python docker/export_locks.py --env mace-agent   # just one
```

`--no-deps` is not a shortcut. The lock is already closed, and re-resolving it
would either fail or silently install versions other than the ones the skills
were validated against — which is the same silent breakage in a new costume.

## Building

```bash
# Lightweight, multi-arch
docker buildx build -f docker/Dockerfile.lightweight \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/learningmatter-mit/atomisticskills-lightweight:dev .

# A GPU image: IMAGE_NAME picks the server set, CONDA_ENVS the locks to install
docker buildx build -f docker/Dockerfile.cuda \
  --platform linux/arm64 \
  --build-arg IMAGE_NAME=mace \
  --build-arg CONDA_ENVS="mace-agent matgl-agent" \
  -t ghcr.io/learningmatter-mit/atomisticskills-mace:dev .
```

CI does this for every image on push to `main` and on `v*` tags; the matrix
comes from `python docker/render.py matrix`.

## Verifying an image

Building green proves nothing about whether a server actually runs. The smoke
test drives a real MCP handshake — `initialize`, then `tools/list` — and fails
unless the server returns its identity and a non-empty tool list:

```bash
bash docker/smoke_test.sh atomisticskills-lightweight:dev base drugdisc smol atomate2
```

## Running a server directly

```bash
docker run -i --rm -v "$PWD:/work" -w /work \
  ghcr.io/learningmatter-mit/atomisticskills-lightweight:1.3.4 base
```

The first argument is the server name; run with none to list what an image
serves. `docker/entrypoint.sh` resolves the name against the baked-in
`server-map.txt`, activates the right environment and execs the module.

The entrypoint matches the owner of `/work` before starting, so results land in
your project owned by you rather than by root.

## Known limitations

- **GPU images are arm64-only.** Their locks were frozen on `linux-aarch64` and
  the generative image compiles PyG extensions for `TORCH_CUDA_ARCH_LIST=12.1`
  (GB10, sm_121). amd64 variants need locks generated on an amd64 host and a
  different arch list.
- **`fpocket` is missing from the arm64 `lightweight` image.** conda-forge
  publishes it for linux-64 only, so `drug-pocket-detection` must use its
  P2Rank path on arm64.
- **The generative image is expensive to build.** `torch-scatter`,
  `torch-cluster` and `torch-sparse` have no aarch64 + CUDA 13 wheels and are
  compiled from source with `--no-build-isolation`.
- **Model weights are not baked in.** Checkpoints download on first use into
  the mounted cache volume, which the plugin points at
  `${CLAUDE_PLUGIN_DATA}/model-cache` so it survives plugin updates.
- **Gated checkpoints still need credentials.** Anything requiring a Hugging
  Face token (for example fairchem UMA) needs `HF_TOKEN` passed into the
  container.
