---
description:
alwaysApply: true
---

# AtomisticSkills Agent Instructions

You are an atomistic research agent with access to literature, Skills, and MCP tools.

## Project Rules

**Read these rules files at the start of every conversation** (imported below via @):
- `.agents/rules/research-standards.md` — research protocol, intent classification, plan workflow
- `.agents/rules/coding-standards.md` — coding rules, environment management, MCP stability
- `.agents/rules/mcp-environments.md` — uv environment and MCP server runtime mapping

@.agents/rules/coding-standards.md
@.agents/rules/mcp-environments.md
@.agents/rules/research-standards.md

**Read these on demand when the task requires it:**
- `.agents/rules/skill-standards.md` — for creating or editing a skill
- `.agents/rules/workflow-standards.md` — for creating or editing a workflow
- `.agents/rules/plot-standards.md` — for creating or editing a plotting script
- `.agents/rules/release-standards.md` — for preparing a release tag

## Framework Overview

This project decomposes complex research tasks into three levels:

- **Tools** (`src/mcp_server/`): Low-level operations exposed via MCP (relax structure, run MD, query databases). Strict typed I/O.
- **Skills** (`skills/`): Mid-level tutorials combining tools and scripts to solve focused tasks. Each has a `SKILL.md` with step-by-step instructions.
- **Workflows** (`.agents/workflows/`): High-level research campaigns that chain multiple skills.

When a user asks a research question, check workflows first for end-to-end protocols, then find the relevant skill(s).

## Skill Discovery

Skills are at `skills/`. In Claude Code they are registered as native
project skills, so each one is listed by name and description and can be invoked
directly with the Skill tool — no searching needed.

**If the skills are not listed, they have not been configured yet.** Run:
```bash
python configure_mcp.py --agent claude
```
This symlinks every `skills/<name>` into `.claude/skills/<name>`, which
Claude Code discovers automatically. `.claude/` is gitignored, so this is a
per-checkout setup step; re-run it after cloning or after a skill is added or
removed. `skills/` stays the single source of truth — the symlinks are
never copies.

Fallback for any agent without native skill registration — scan the frontmatter
descriptions directly:
```bash
grep -r "^description:" skills/*/SKILL.md
```

Then read the full `SKILL.md` for any matching skill and follow its numbered instructions.

Use the `/skill-search` command for interactive discovery: `/skill-search [search term]`

## Executing Skills

Skill commands run through the launcher `venv/run`:
```bash
${CLAUDE_SKILL_DIR}/../../venv/run <venv>[+<extra>] python ${CLAUDE_SKILL_DIR}/scripts/<script>.py ...
```
Or from the repository root:
```bash
venv/run <venv>[+<extra>] python skills/<skill-name>/scripts/<script>.py ...
```
Where `<venv>` is a shared project (`cpu`, `mlip`, `fairchem`) or a research stack
(`adit`, `diffcsp`, `mattergen`, `msms`, `reactot`, `scd`), matching `metadata.venv`
in the skill's `SKILL.md`, with optional extras if needed (e.g. `cpu+openmm`, `mlip+lammps`).

### MCP tool calls

MCP steps in skills are written `server.tool` (e.g. `matgl.relax_structure`).
- When connected directly, the tool is named `mcp__<server>__<tool>`.
- When installed as a plugin, it is named `mcp__plugin_atomistic-skills_<server>__<tool>`.
- Without a connected server, the same tool runs from the shell via the CLI fallback:
  ```bash
  venv/run <venv> python -m src.mcp_server.cli <server> <tool> key=value ...
  ```
  Run with `--list` to see available tools. Several tools named in one command share a process, so state from `load_model` persists.

## MCP Server Setup

Run `configure_mcp.py` to write configs for your agent:
```bash
python configure_mcp.py                    # auto-detect installed agents
python configure_mcp.py --agent claude    # Claude Code only
python configure_mcp.py --scope global    # write to global user config
```

See `README.md` for full installation instructions.

### Containerised servers and images

Every container image is built from committed `venv/<name>/uv.lock` files, so a container
runs the same environment as a native install: `cpu`, `mlip` and `fairchem` (linux/amd64 and
linux/arm64) via `docker/Dockerfile`, and `generative` (linux/arm64) via `docker/Dockerfile.cuda`,
which installs `adit`, `diffcsp` and `mattergen` side by side and compiles their PyG extensions
with CUDA. On x86_64 the generative servers run from their uv projects on the host.

`docker/images.json` is the single source of truth mapping each server to its runtime image.
The server table `venv/servers.tsv`, the `mcpServers` block of `.claude-plugin/plugin.json`,
the CI build matrix, and the in-image server maps are all rendered from it.

**When you change anything about a server's packaging or images**, re-render rather than
hand-editing derived files:
```bash
python docker/render.py servers --check        # or without --check to rewrite venv/servers.tsv
python docker/render.py plugin-mcp --check     # plugin wiring in .claude-plugin/plugin.json
python docker/render.py matrix                 # CI build matrix for build-images.yml
venv/run cpu python tools/sync_version.py --check           # manifest versions match VERSION
```
CI validates that rendered files and version manifests are current.
