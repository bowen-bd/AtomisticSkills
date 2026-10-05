# Container Images for AtomisticSkills

These container images serve as the automatic fallback runtime for the `venv/run` launcher and the Claude Code plugin. If a host cannot run the `uv` environments natively (due to older glibc, missing build compilers, macOS, or cluster policies), `venv/run` automatically executes commands and MCP servers inside the corresponding container image.

`docker/images.json` is the single source of truth for container images, their build strategies, platforms, and the MCP servers they carry. All derived manifests and configuration files are generated via `docker/render.py`.

## Image Inventory

| Image | Build Strategy | Servers Carried | Platforms | GPU |
| :--- | :--- | :--- | :--- | :--- |
| `atomisticskills-cpu` | `uv` (from `venv/cpu/uv.lock`) | `atomate2`, `base`, `drugdisc`, `smol` | linux/amd64, linux/arm64 | No |
| `atomisticskills-mlip` | `uv` (from `venv/mlip/uv.lock`) | `mace`, `matgl` | linux/amd64, linux/arm64 | Yes |
| `atomisticskills-fairchem` | `uv` (from `venv/fairchem/uv.lock`) | `fairchem` | linux/amd64, linux/arm64 | Yes |
| `atomisticskills-generative` | `conda-lock` (from conda locks) | `adit`, `diffcsp`, `mattergen` | linux/arm64 | Yes |

### Build Strategies

1. **`uv` projects (`cpu`, `mlip`, `fairchem`)**:
   Built directly from their committed `venv/<name>/uv.lock` by `docker/Dockerfile`. This ensures that the container runtime runs the exact same pinned environment as a native host `uv` install, with the build tools, system libraries (OpenMM dependencies, Boost, fpocket), and Ubuntu 24.04 runtime pre-packaged.

2. **`conda-lock` (`generative`)**:
   Built by `docker/Dockerfile.cuda` for linux/arm64. The generative models (ADiT, DiffCSP++, MatterGen) rely on non-PyPI packages and specific PyTorch/PyG combinations compiled from source.

## Building Images Locally

### 1. Build uv-based Images (`cpu`, `mlip`, `fairchem`)

Use `docker/Dockerfile` with build arguments `VENV` and `IMAGE_NAME`:

```bash
# CPU image
docker build -f docker/Dockerfile \
  --build-arg VENV=cpu --build-arg IMAGE_NAME=cpu \
  -t ghcr.io/learningmatter-mit/atomisticskills-cpu:dev .

# MLIP image
docker build -f docker/Dockerfile \
  --build-arg VENV=mlip --build-arg IMAGE_NAME=mlip \
  -t ghcr.io/learningmatter-mit/atomisticskills-mlip:dev .

# FairChem image
docker build -f docker/Dockerfile \
  --build-arg VENV=fairchem --build-arg IMAGE_NAME=fairchem \
  -t ghcr.io/learningmatter-mit/atomisticskills-fairchem:dev .
```

To build multi-arch images with Buildx:
```bash
docker buildx build -f docker/Dockerfile \
  --platform linux/amd64,linux/arm64 \
  --build-arg VENV=cpu --build-arg IMAGE_NAME=cpu \
  -t ghcr.io/learningmatter-mit/atomisticskills-cpu:dev .
```

### 2. Build the Generative Image

```bash
docker buildx build -f docker/Dockerfile.cuda \
  --platform linux/arm64 \
  --build-arg IMAGE_NAME=generative \
  -t ghcr.io/learningmatter-mit/atomisticskills-generative:dev .
```

## Manifest Rendering (`docker/render.py`)

Never manually edit derived configuration files. Re-render them from `docker/images.json`:

```bash
# Render venv/servers.tsv (read by venv/run)
python docker/render.py servers

# Render mcpServers block in .claude-plugin/plugin.json
python docker/render.py plugin-mcp

# Render GitHub Actions build matrix for build-images.yml
python docker/render.py matrix

# Check that rendered files are up to date (used by CI)
python docker/render.py servers --check
python docker/render.py plugin-mcp --check
```

To refresh conda locks for the generative image:
```bash
python docker/export_locks.py
```

## Running Servers and Commands Through Containers

The unified launcher `venv/run` automatically mounts the active workspace to `/work` inside the container and manages user identity and GPU passthrough:

```bash
# Run via Docker backend
ATOMISTIC_RUNTIME=docker venv/run cpu python skills/mat-xrd-calculator/scripts/calculate_xrd.py ...

# Start an MCP server via Docker
ATOMISTIC_RUNTIME=docker venv/run --server base
```

### Apptainer / HPC Clusters

On HPC clusters without a Docker daemon, set `ATOMISTIC_RUNTIME: apptainer` in `~/.config/atomistic_skills.yaml` (or via environment variable).

`venv/run` converts the OCI images into SIF format under `<model-cache>/sif/` with proper locking. To pre-build SIF containers ahead of time so MCP servers do not time out on first launch:
```bash
venv/run --setup
```

## Known Limitations

- **Generative stack is arm64 only**: The `generative` image (`adit`, `diffcsp`, `mattergen`) is built for linux/arm64. On x86_64, these servers report that containers are arm64-only and exit cleanly.
- **GPU Driver Requirements**: On aarch64, CUDA 13 wheels require NVIDIA driver ≥ 580.
- **Model Checkpoints**: Model weights are downloaded on first use into the mounted cache directory (`ATOMISTIC_MODEL_CACHE` or `~/.cache/atomisticskills`), so they persist across container restarts and updates.
