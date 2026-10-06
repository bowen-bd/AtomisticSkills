# AtomisticSkills Setup Guide

Guide the user step-by-step through setting up AtomisticSkills. Ask before running
anything that installs software, and wait for each step to finish.

## How AtomisticSkills runs

Every skill command and every MCP server starts through one launcher, `venv/run`.
It runs the code in one of three Python environments -- uv projects under `venv/`:

| Environment | Contents |
| :--- | :--- |
| `cpu` | materials, chemistry, drug discovery and analysis (no torch) |
| `mlip` | MACE and MatGL on top of the CPU stack (GPU) |
| `fairchem` | FairChem (UMA, eSEN) on top of the CPU stack (GPU) |

The launcher uses **uv** on this machine when it can, and otherwise falls back to
a **container image** built from the same lock (Docker, Podman or Apptainer), with
the same paths, so nothing else changes. Environments are created on first use;
`venv/run --setup` creates them ahead of time.

What a native (uv) install needs:

- Linux on x86_64 or aarch64, glibc 2.28 or newer (aarch64 GPU stacks: 2.34)
- [uv](https://docs.astral.sh/uv/) and a C compiler (`gcc`), for the few packages
  that build from source. Python itself comes from uv (a managed CPython with its
  headers), so no system Python or `python3-devel` package is needed
- For GPU work, an NVIDIA driver 525 or newer. The mlip and fairchem environments
  carry two torch builds and `venv/run` picks by driver: CUDA 13 (driver ≥ 580,
  required for GB10/Blackwell) or CUDA 12.6 (drivers 525–579, common on clusters).
  `venv/run --doctor` shows the choice; `ATOMISTIC_TORCH_CUDA=cu126|cu130`
  overrides it. Container images carry the CUDA 13 build only

Hosts that do not meet these -- older clusters, macOS, no compiler -- use the
container fallback automatically once a container runtime is installed.

## Option A: Claude Code plugin

Ask the user to run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # skip if uv is installed
claude plugin marketplace add learningmatter-mit/AtomisticSkills
claude plugin install atomistic-skills@atomistic-skills
```

To install from a local checkout instead, add a fresh clone as the marketplace:
a local path is copied whole into the plugin cache, including any
environments already built under `venv/*/.venv` (tens of GB).

The plugin brings every skill and all MCP servers. Its options are optional:
`runtime` (default `auto`; set `apptainer` on an HPC cluster that should use
containers), and `image_registry` / `image_tag` for the container fallback.

The first use of the GPU environments downloads several GB. To do it up front, have
Claude run the `general-atomisticskills-setup` skill (it runs `venv/run --setup` and
`venv/run --doctor` from the plugin's install directory). An MCP server whose
environment is still being created reports that it is preparing it; reconnect it
with `/mcp` once the setup finishes.

## Option B: Clone the repository (any agent)

For Claude Code, Codex, Cursor, Gemini or Windsurf, and for developing skills:

```bash
git clone git@github.com:learningmatter-mit/AtomisticSkills.git
cd AtomisticSkills
curl -LsSf https://astral.sh/uv/install.sh | sh        # skip if uv is installed
venv/run --setup                                         # cpu, mlip, fairchem (several GB)
python3 configure_mcp.py                                 # register MCP servers and skills
```

`configure_mcp.py` auto-detects installed agents; pass `--agent claude` (or codex,
gemini, cursor, windsurf) to choose, and `--scope global` to make the tools
available outside this repository:

| Client | Project scope | Global scope (all projects) |
|--------|--------------|----------------------------|
| **Claude Code** | `.mcp.json` (project root) | `~/.claude.json` |
| **Codex CLI** | `.codex/config.toml` | `~/.codex/config.toml` |
| **Cursor** | `.cursor/mcp.json` | `~/.cursor/mcp.json` |
| **Windsurf** | — | `~/.codeium/windsurf/mcp_config.json` |
| **Gemini CLI** | `.gemini/settings.json` | `~/.gemini/settings.json` |

For Claude Code it also links every skill into `.claude/skills/`, so the skills are
registered natively. With `--scope global`, Codex and Gemini also get the skills and
a pointer to the rules and workflows in their global configuration. Restart the
assistant after changing its configuration.

## API keys and settings

Settings live in `~/.config/atomistic_skills.yaml` (environment variables of the
same name take precedence). Offer this template and fill in what the user has:

```yaml
# Materials Project (mat-db-mp, phase diagrams, many workflows)
MP_API_KEY: "your_mp_api_key_here"

# Hugging Face token: required for FairChem UMA, a gated model (request access at
# https://huggingface.co/facebook/UMA first)
HF_TOKEN: "your_hf_token_here"

# Atomate2 remote project for DFT jobs (jobflow-remote)
ATOMATE2_REMOTE_PROJECT: "your_project_name"

# ORCA binary for the chem-dft-orca-* skills (x86_64 only)
ORCA_BINARY_PATH: /path/to/orca_directory/orca

# How skills and MCP servers run: auto (default), uv, docker, podman, apptainer
ATOMISTIC_RUNTIME: auto
```

## HPC clusters

- Run `venv/run --setup` once on a login node with network access; the
  environments live in the checkout (or the plugin directory) and are shared with
  compute nodes over the shared filesystem.
- If the cluster's glibc is too old or there is no compiler, set
  `ATOMISTIC_RUNTIME: apptainer`. `venv/run --setup` then converts the images to
  SIF files once (minutes per image), so MCP servers start without timing out.
- Even with `auto`, extras that need a newer glibc than the host's (`cpu+openmm`
  and `cpu+pymol` on RHEL 8) run from the Apptainer image. The first such command
  converts the image to a SIF, which took 15–30 minutes on an NFS home directory.
  `--setup cpu` does not do this when `cpu` itself runs natively, so trigger it
  once ahead of time with `venv/run cpu+openmm python -c 1`.
- After an upgrade, delete superseded `atomisticskills-*.sif` files (about 1.4 GB
  each) from `~/.cache/atomisticskills/sif/` or `$ATOMISTIC_MODEL_CACHE/sif/`.
- GPU nodes can differ from login nodes. A node with glibc older than 2.28 (for
  example CentOS 7) needs a container runtime installed there, and the images
  carry the CUDA 13 build (driver ≥ 580). On a native uv node, a driver of 525–579
  gets the CUDA 12.6 build automatically.
- Keep large caches off a small home quota with `UV_CACHE_DIR` (uv's download cache)
  and `ATOMISTIC_MODEL_CACHE` (container checkpoints and SIF files).

## Research stacks

MatterGen, ADiT and DiffCSP++ (`ml-generative-*`), ICEBERG (`chem-msms-predict`),
React-OT (`chem-react-ot`) and SelfConditionedDenoisingAtoms
(`ml-property-predict-scd`) each have their own pinned uv project under `venv/`,
created on first use like the shared ones. Their compiled dependencies set what
a host needs, and `venv/run --doctor` shows what this one can run:

- MatterGen, ADiT, DiffCSP++: natively on x86_64 with glibc ≥ 2.32 (PyG's
  wheels need it); elsewhere (aarch64, or EL8-era clusters with glibc 2.28) from
  the `generative` container image (`venv/run --setup generative` pulls it).
- SelfConditionedDenoisingAtoms (`scd`): x86_64 with glibc ≥ 2.32; no image.
- ICEBERG (`msms`): x86_64 with GCC 9's libstdc++ (DGL needs `GLIBCXX_3.4.26`:
  glibc ≥ 2.31 distributions ship it). On EL8, put a newer GCC runtime first
  (e.g. `module load gcc` or `LD_LIBRARY_PATH=<gcc>/lib64:$LD_LIBRARY_PATH`) and
  run it natively with `ATOMISTIC_RUNTIME=uv`.
- React-OT (`reactot`): x86_64 and aarch64.

These stacks download several GB of CUDA libraries on first use; on a cluster,
create them ahead of time (`venv/run --setup mattergen adit ...`) rather than on
an MCP server's first start.

LAMMPS with ML plugins (`mat-lammps-md`) is built against the `mlip`
environment by the skill's build scripts; FairChem's `lmp_fc` comes with the
`fairchem+lammps` extra. Nothing uses conda.

## Check the installation

```bash
venv/run --doctor
```

reports the host, which backend each environment uses, and whether it is ready.
Then run a live test with the user:

- **Materials Project**: "Search the Materials Project for the stable structure of
  LiFePO4." (`base.search_materials_project_by_formula`)
- **MLIP**: "Relax this LiFePO4 structure with MACE and report the energy."
  (`mace.load_model`, then `mace.relax_structure`)

## Best Practices for Users

- **Leverage local GPUs**: MLIP tasks are far faster on a machine with a GPU.
- **Customize**: Add your own skills, MCP tools and workflows to the project.
- **Contribute back**: If you develop a robust, general skill, please open a PR.

## Common Issues

| Issue | Fix |
|-------|-----|
| An MCP server is not connected | Run `venv/run --doctor`. A first start creates the environment in the background; reconnect with `/mcp` when it finishes, or run `venv/run --setup` first. |
| `needs glibc >= …` or `needs a C compiler` | Install a container runtime (Apptainer on HPC, Docker elsewhere); `auto` then uses it. |
| `No module named ...` in a skill script | Run the command exactly as the skill writes it: `venv/run <env> ...` picks the environment the script needs. |
| GPU not used | Run `venv/run --doctor`: drivers 525–579 get the CUDA 12.6 build, 580+ CUDA 13, older ones the CPU. In a container (images are CUDA 13) a driver older than 580 means CPU; use the uv backend there. |
| FairChem `load_model` fails with `401` / gated repo | UMA checkpoints are gated: request access at https://huggingface.co/facebook/UMA, then set `HF_TOKEN` (in `~/.config/atomistic_skills.yaml` or the environment). |
| Model download fails with `CERTIFICATE_VERIFY_FAILED` | The launcher points Python at the system CA bundle; behind a proxy or with a custom bundle, set `SSL_CERT_FILE` to it. |
| `SyntaxError` running a `tools/` script | `python3` is too old (Python 3.6 on RHEL 8); run it as `venv/run cpu python tools/<script>.py`. |
| Atomate2 remote worker issues | See [docs/atomate2_remote_workers.md](atomate2_remote_workers.md) |
