# AtomisticSkills Agent Instructions

You are an atomistic research agent with access to literature, Skills, and MCP tools.

**Read these rules files at the start of every conversation:**
- `.agents/rules/research-standards.md` — research protocol, intent classification, plan workflow
- `.agents/rules/coding-standards.md` — coding rules, environment management, MCP stability
- `.agents/rules/mcp-environments.md` — conda environment to MCP server mapping

**Read these on demand when the task requires it:**
- `.agents/rules/skill-standards.md` — for creating or editing a skill
- `.agents/rules/workflow-standards.md` — for creating or editing a workflow
- `.agents/rules/plot-standards.md` — for creating or editing a plotting script

## Framework Overview

This project decomposes complex research tasks into three levels:

- **Tools** (`src/mcp_server/`): Low-level operations exposed via MCP (relax structure, run MD, query databases). Strict typed I/O.
- **Skills** (`skills/`): Mid-level tutorials combining tools and scripts to solve focused tasks. Each has a `SKILL.md` with step-by-step instructions.
- **Workflows** (`.agents/workflows/`): High-level research campaigns that chain multiple skills.

When a user asks a research question, check workflows first for end-to-end protocols, then find the relevant skill(s).

## Skill Discovery

Skills are at `skills/`. Scan frontmatter descriptions to find relevant ones:
```bash
grep -r "^description:" skills/*/SKILL.md
```

Then read the full `SKILL.md` for any matching skill and follow its numbered instructions.

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
python configure_mcp.py --agent gemini     # Gemini CLI only
python configure_mcp.py --scope global    # write to global user config
```

See `README.md` for full installation instructions.
