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
| `atomisticskills-mace` | mace, matgl | 3.12 | 2.12 | amd64 + arm64 |
| `atomisticskills-fairchem` | fairchem | 3.12 | 2.8 (amd64) / 2.10 (arm64) | amd64 + arm64 |
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

That paragraph describes **aarch64**, and the distinction matters: the conflict
is a property of the platform, not of the packages. torch 2.8 has no sm_121
build, so the aarch64 environment had to override the pin. On x86_64 it is
satisfiable, and `fairchem-agent` resolves to the declared torch 2.8.0 with no
override at all.

That is why the two architectures are locked by different means:

- **aarch64** — frozen from the validated environments on the workstation that
  has them, with `docker/export_locks.py`. These cannot be re-resolved.
- **amd64** — resolved from each environment's declared spec with
  `docker/resolve_locks.py`, which runs `uv pip compile` for
  `x86_64-unknown-linux-gnu`. No amd64 machine is needed to produce them.

Versions are therefore resolved independently per architecture and will not
match exactly across them. That is deliberate: forcing amd64 to adopt aarch64's
overrides would import constraint violations that only ever existed because of
sm_121.

`resolve_locks.py` covers `mace-agent`, `matgl-agent` and `fairchem-agent` only.
The generative environments were assembled by hand -- `adit-agent`'s spec
declares nothing but python, pip and uv -- so there is no spec to resolve and
their amd64 locks must come from a built environment.

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
  ghcr.io/learningmatter-mit/atomisticskills-lightweight:1.3.5 base
```

The first argument is the server name; run with none to list what an image
serves. `docker/entrypoint.sh` resolves the name against the baked-in
`server-map.txt`, activates the right environment and execs the module.

The entrypoint matches the owner of `/work` before starting, so results land in
your project owned by you rather than by root.

## Testing images published from a fork

CI pushes to `ghcr.io/<repository_owner>`, so images built on a fork land in the
fork's namespace, not the upstream one. The plugin reads the registry from a
`userConfig` value rather than hardcoding it, so point it at your fork when
prompted:

```
Image registry: ghcr.io/<your-github-user>
```

The GHCR package is private on first publish. Make it public under
*Packages → atomisticskills-\<image\> → Package settings → Change visibility*,
otherwise anyone pulling it needs a token.

## Running on HPC (Apptainer)

Clusters do not give users root and do not run a Docker daemon; Apptainer
(formerly Singularity) is the standard runtime there. It is supported directly
-- set the runtime option and nothing else changes:

```bash
claude plugin install atomistic-skills@atomistic-skills \
  --config container_runtime=apptainer \
  --config work_dir=/home/<you>/atomistic-work \
  --config image_registry=ghcr.io/learningmatter-mit
```

**Prepare the images first.** Claude Code probes every server in parallel and
allows each 30 seconds to connect, while converting a 3.5 GB OCI image to a SIF
takes minutes. Without a pre-build, every server times out on first run and the
parallel probe starts several conversions at once -- an HPC test burned 29 GB of
quota exactly that way. Run this once, before starting Claude Code:

```bash
bash docker/prepare_images.sh --runtime apptainer \
  --registry ghcr.io/learningmatter-mit
```

It builds only the images matching your architecture, sequentially, and reports
the ones it skips. The launcher will still build on demand if you skip this
step, but the first connections will time out until the build finishes.

**Where the SIF lands, and why the launcher searches for it.** `plugin.json`
points `ATOMISTIC_MODEL_CACHE` at `${CLAUDE_PLUGIN_DATA}/model-cache`, but
Claude Code does not create that directory until the first session loads the
plugin -- which is *after* the pre-build above. So `prepare_images.sh` normally
writes to `~/.cache/atomisticskills` and says so. That is correct, not a
degraded mode: `run_server.sh` checks the model cache first and then that
shared location, so the pre-build is used either way.

`prepare_images.sh` searches the same list before building, so a second run --
once Claude Code has created the data directory and the target therefore moves
-- reports `have` instead of rebuilding. Without that, an HPC node spent 15m12s
regenerating a 1.2 GB image it already had. Keep the two lists in step.

Earlier releases had the pre-build try to predict the plugin's data directory
instead. It guessed wrong on every HPC install, and the node sat through four
30-second connect timeouts with a valid 1.2 GB SIF already on disk. Do not
reintroduce that by guessing harder at `CLAUDE_PLUGIN_DATA`; the search in the
launcher is the part that makes the two agree. `ATOMISTIC_SIF_DIR` overrides the
search if you keep SIFs somewhere else entirely.

`docker/run_server.sh` translates between the two, because their arguments are
not interchangeable:

| | Docker / Podman | Apptainer / Singularity |
| :--- | :--- | :--- |
| invoke | `run --rm -i IMAGE srv` | `exec docker://IMAGE /entrypoint srv` |
| bind | `--volume h:/work` | `--bind h:/work` |
| workdir | `--workdir /work` | `--pwd /work` |
| GPU | `--gpus all` | `--nv` |

Cached SIFs live in `<model-cache>/sif/` and are reused; the launcher takes a
`flock` around the build so ten servers starting together cannot each convert
the same image.

Three further Apptainer specifics, all learned from a cluster:

- **`mksquashfs` thread exhaustion.** It defaults to one thread per core. On a
  448-core node with `ulimit -u` of 768 it dies with `FATAL ERROR: Failed to
  create thread`. The launcher bounds `-processors` from the actual limit;
  override with `ATOMISTIC_SQUASHFS_PROCS` if needed.
- **`/tmp` mounted `nodev`.** Apptainer warns this can corrupt a build, so
  `APPTAINER_TMPDIR` is pointed beside the cache instead.
- **Architecture.** The launcher refuses an image built for another
  architecture before downloading anything, so on an x86_64 cluster the three
  arm64-only generative servers fail instantly with an explanation rather than
  pulling gigabytes and then failing.

Apptainer already runs as the invoking user, so the entrypoint's privilege drop
is a no-op there -- files in `/work` are yours either way.

## Installing non-interactively

`claude plugin install` does not prompt in a non-interactive shell; it installs
and reports the options as unset. Pass them explicitly:

```bash
claude plugin install atomistic-skills@atomistic-skills \
  --config container_runtime=docker \
  --config work_dir=/path/to/workdir \
  --config image_registry=ghcr.io/learningmatter-mit
```

Existing installs can be reconfigured with
`/plugin configure atomistic-skills@atomistic-skills` inside Claude Code.

## Troubleshooting

**`Failed to connect — ENOENT: Executable not found in $PATH: "stdio"`**

The runtime binary is missing. Claude Code's error sanitiser replaces the
command name with `"stdio"`, which makes this look like a protocol fault when
it is simply "docker is not installed". Check with `command -v docker podman
apptainer`, then set `container_runtime` to whatever the host actually has.
The launcher checks for the runtime before invoking it and reports the missing
binary by name, along with the runtimes it did find on the host.

**`claude plugin details` reports `MCP servers (0)`**

Cosmetic. The inventory counts servers declared through an external file, and
this plugin declares them inline in `plugin.json`. `claude mcp list` is the
accurate view -- the servers are registered and will show there.

**A server keeps failing instantly, even after you fixed the cause**

Claude Code caches a failed MCP connection for about 15 minutes:
`Skipping connection (recent failure cached, retries automatically in 15 min,
or edit the plugin config to retry now)`. After pre-building an image or
changing the runtime, a retry inside that window will not even invoke the
launcher. Either wait it out or touch the plugin configuration
(`/plugin configure atomistic-skills@atomistic-skills`) to clear it.

**Materials Project or literature tools return authentication errors**

Those need credentials, which the launcher forwards from the host environment
when they are set: `MP_API_KEY`, `HF_TOKEN`, `OPENALEX_EMAIL`,
`ELSEVIER_API_KEY`, `ELSEVIER_INST_TOKEN`, `SPRINGER_API_KEY`,
`UNPAYWALL_EMAIL`. Export them before starting Claude Code.

## Known limitations

- **The generative image is arm64-only.** `adit`, `diffcsp` and `mattergen`
  therefore refuse on x86_64, with the architecture gate explaining why and
  downloading nothing. The other seven servers run on both. Producing an amd64
  generative image means deriving its locks from a built environment, since
  those three have no resolvable spec.
- **GPU target lists differ by architecture.** arm64 targets GB10 (sm_121)
  alone; amd64 targets `8.0;8.6;8.9;9.0;12.0`, covering A100, A40/A6000,
  L40S/RTX 40xx, H100/H200 and RTX 50xx. A GPU outside that list falls back to
  JIT compilation on first use, or fails if the architecture is too new.
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
