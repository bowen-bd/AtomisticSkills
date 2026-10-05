---
name: general-atomisticskills-setup
description: Set up, check or troubleshoot how AtomisticSkills runs on this machine -- creating its Python environments, connecting its MCP servers, choosing uv or a container runtime, and configuring API keys. Use it when installing AtomisticSkills, when a skill command or MCP tool fails to start, or before a first research task.
metadata:
  category: [general]
  venv: [cpu, mlip]
---

# AtomisticSkills Setup and Runtime

## Goal

Make every AtomisticSkills skill and MCP tool runnable on the current machine, and
know how they run: which environment a command uses, where results go, and what to
do when something fails to start.

## How skills run

- **Commands.** Every skill command starts through the launcher
  `${CLAUDE_SKILL_DIR}/../../venv/run <env> ...`, where `<env>` is `cpu`, `mlip`
  (MACE, MatGL) or `fairchem`, sometimes with an extra such as `cpu+openmm`. Run
  commands exactly as a skill writes them; the launcher picks the environment, and
  creates it on first use.
- **Backends.** The launcher uses uv on this machine when it can, and otherwise a
  container image built from the same lock (Docker, Podman, Apptainer), with the
  same paths. `ATOMISTIC_RUNTIME` (or the plugin's `runtime` option) forces one.
- **MCP tools.** Skills write MCP tools as `server.tool`, e.g. `mace.relax_structure`.
  If a server is not connected, the same tool runs from the shell; tools named in
  one command share a process, so a loaded model stays loaded:
  `${CLAUDE_SKILL_DIR}/../../venv/run mlip python -m src.mcp_server.cli mace load_model relax_structure structure_data=POSCAR`.
  `--list` shows a server's tools and arguments.
- **Results.** Scripts and MCP servers write into the current project: a research
  task gets its own `research/<date>_<topic>/` directory
  (`base.create_research_dir`), and later outputs default to it. Never write
  results inside the skill or plugin directory.

## Instructions

### 1. Check the machine

```bash
${CLAUDE_SKILL_DIR}/../../venv/run --doctor
```

It reports the host (architecture, glibc, uv, container runtimes, GPU and driver)
and, per environment, whether it runs with uv or a container and whether it is ready.

### 2. Create the environments

```bash
${CLAUDE_SKILL_DIR}/../../venv/run --setup
```

This creates `cpu`, `mlip` and `fairchem` -- several GB for the GPU environments, so
warn the user and use a long timeout. Name environments to create only some
(`--setup cpu mlip`). On aarch64 with a container runtime, `--setup generative`
also fetches the image behind the `adit`, `diffcsp` and `mattergen` servers.

If `--doctor` reports a missing prerequisite, fix it before retrying:

| Report | Fix |
| :--- | :--- |
| `uv is not installed` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (ask the user first) |
| `needs glibc >= …` or `needs a C compiler` | install a container runtime (Apptainer on HPC, Docker elsewhere); `auto` then uses it |
| MCP server "being created in the background" | wait for `--setup` (or the background log it names) to finish, then reconnect the server with `/mcp` |

### 3. Configure API keys

Settings live in `~/.config/atomistic_skills.yaml` (environment variables of the
same name take precedence). Ask the user for the keys they need -- never invent
them:

```yaml
MP_API_KEY: "..."        # Materials Project
HF_TOKEN: "..."          # gated Hugging Face models, e.g. FairChem UMA
ATOMISTIC_RUNTIME: auto  # or uv, docker, podman, apptainer
```

### 4. Verify with a real calculation

```bash
${CLAUDE_SKILL_DIR}/../../venv/run cpu python -m src.mcp_server.cli base --list
${CLAUDE_SKILL_DIR}/../../venv/run mlip python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Then, with the MCP servers connected, relax a small structure: `mace.load_model`
followed by `mace.relax_structure` on a two-atom silicon cell should finish in
seconds on a GPU.

## Constraints

- **Platforms**: native environments need Linux on x86_64 or aarch64; other hosts
  use the container fallback. PyMOL, SCINE/ORCA and fpocket exist for x86_64 only.
- **Conda stacks**: skills that declare `metadata.conda_env` (generative models,
  React-OT, ICEBERG, LAMMPS builds) need that conda environment from
  `conda-envs/<env>/install.sh`, or for the generative MCP servers on aarch64, the
  `generative` container image.
- **Installing software**: ask the user before installing uv, a container runtime or
  system packages, and before downloading multi-GB environments.

---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
