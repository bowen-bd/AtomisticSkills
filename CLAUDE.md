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
- `.agents/rules/mcp-environments.md` — conda environment to MCP server mapping

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

### Scripts with `# Env:` annotations
```bash
# Env: mace-agent
python skills/mat-melting-point/scripts/create_interface.py ...
```
Run with:
```bash
mamba activate <env-name>
# or
conda run -n <env-name> python <path-to-script> [args]
```

### MCP tool calls
Skills that reference `mcp_*` functions require MCP servers to be configured. If unavailable, check the skill's `scripts/` directory or `src/utils/`.

## MCP Server Setup

Run `configure_mcp.py` to write configs for your agent:
```bash
python configure_mcp.py                    # auto-detect installed agents
python configure_mcp.py --agent claude    # Claude Code only
python configure_mcp.py --scope global    # write to global user config
```

See `README.md` for full installation instructions.

### Containerised servers

The same servers also ship as container images so the plugin can be installed
without building the ~58 GB of conda environments. `docker/images.json` is the
single source of truth mapping each server to its image; the Dockerfiles, the CI
matrix, the in-image server table and the `mcpServers` block of
`.claude-plugin/plugin.json` are all rendered from it.

**When you change anything about a server's packaging**, re-render rather than
hand-editing the derived files:
```bash
conda run -n base-agent python docker/render.py plugin-mcp   # plugin wiring
conda run -n base-agent python docker/export_locks.py        # refresh lockfiles
conda run -n base-agent python tools/sync_version.py --check # manifest versions
```
CI fails if any of those are stale. Read `docker/README.md` before touching the
images — in particular, the GPU environments are installed from lockfiles with
`--no-deps` because they cannot be resolved (`fairchem-core 2.19.0` declares
`torch~=2.8.0` while the environment must run torch 2.10.0, since torch 2.8 has
no sm_121 build). Do not "fix" that by relaxing the pins.
