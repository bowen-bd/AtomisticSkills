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
- For GPU work, NVIDIA driver 580 or newer: the mlip and fairchem environments use
  CUDA 13 builds of torch on both architectures. On an older driver the GPU is not
  used; `venv/run --doctor` warns about it

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
- Keep large caches off a small home quota with `UV_CACHE_DIR` (uv's download cache)
  and `ATOMISTIC_MODEL_CACHE` (container checkpoints and SIF files).

## Generative models and other conda stacks

MatterGen, ADiT and DiffCSP++ (`ml-generative-*`), React-OT (`chem-react-ot`),
ICEBERG (`chem-msms-predict`) and the LAMMPS builds (`mat-lammps-md`) cannot live
in a uv project. On aarch64 the generative MCP servers run from the `generative`
container image (`venv/run --setup generative`). Otherwise build the conda
environment the skill names in its `metadata.conda_env`, using
`conda-envs/<env>/install.sh`; `configure_mcp.py` uses those environments for the
generative MCP servers when it finds them.

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
| GPU not used | Check `nvidia-smi`: the driver must be 580 or newer (CUDA 13 builds). On a cluster, pick a GPU partition with a current driver. |
| FairChem `load_model` fails with `401` / gated repo | UMA checkpoints are gated: request access at https://huggingface.co/facebook/UMA, then set `HF_TOKEN` (in `~/.config/atomistic_skills.yaml` or the environment). |
| Model download fails with `CERTIFICATE_VERIFY_FAILED` | The launcher points Python at the system CA bundle; behind a proxy or with a custom bundle, set `SSL_CERT_FILE` to it. |
| `SyntaxError` running a `tools/` script | `python3` is too old (Python 3.6 on RHEL 8); run it as `venv/run cpu python tools/<script>.py`. |
| Atomate2 remote worker issues | See `conda-envs/atomate2-agent/atomate2_remote_worker_setup.md` |
