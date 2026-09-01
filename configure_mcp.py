#!/usr/bin/env python3
"""Configure AtomisticSkills MCP servers for any supported AI agent.

Writes MCP server configs to the correct location for each agent, adapting
paths to the local conda installation.

Supported agents:
  claude   - Claude Code (.mcp.json or ~/.claude/settings.json)
  codex    - OpenAI Codex CLI (.codex/config.toml)
  gemini   - Google Gemini CLI and IDE (.gemini/settings.json, config/mcp_config.json, and custom plugin)
  cursor   - Cursor (.cursor/mcp.json)
  windsurf - Windsurf (~/.codeium/windsurf/mcp_config.json, global only)

Skills (skills/) and workflows (.agents/workflows/) are the universal
cross-platform paths — no changes needed for different agents.
Instruction files (CLAUDE.md / AGENTS.md / GEMINI.md) are auto-generated
for agents that don't already have one.

Usage:
    python configure_mcp.py                        # auto-detect installed agents
    python configure_mcp.py --agent claude         # specific agent only
    python configure_mcp.py --agent claude codex   # multiple agents
    python configure_mcp.py --conda /path/to/miniforge3
    python configure_mcp.py --scope global         # write to global config only
    python configure_mcp.py --scope project        # write to project config only
    python configure_mcp.py --list-agents          # show detected agents
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV_PATTERN = re.compile(r".*/envs/([^/]+)/bin/python$")
PROJECT_ROOT = Path(__file__).resolve().parent
MCP_SOURCE = PROJECT_ROOT / "mcp_config.json"

# Instruction file stub used when a target agent has no instruction file yet.
INSTRUCTION_STUB = """\
# AtomisticSkills Agent Instructions

This project uses AtomisticSkills — a framework for atomistic simulation
workflows combining literature, MLIP tools, and MCP servers.

See CLAUDE.md for the full instructions (Claude Code format, also applicable
to other agents). Skills are in skills/, workflows in .agents/workflows/.
"""

ATOMISTICSKILLS_GLOBAL_MARKER = "# AtomisticSkills Global Reference"


def _global_reference_block() -> str:
    """Return a compact global instruction block for agents outside this repo."""
    return f"""\
{ATOMISTICSKILLS_GLOBAL_MARKER}

If your current workspace is NOT {PROJECT_ROOT} (or any of its subdirectories), and the task involves atomistic research, materials simulation, drug discovery, spectroscopy, ML interatomic potentials, or related scientific workflows:
- The AtomisticSkills repository is installed at {PROJECT_ROOT}.
- Rules live at {PROJECT_ROOT}/.agents/rules/.
- Skills live at {PROJECT_ROOT}/skills/.
- Workflows live at {PROJECT_ROOT}/.agents/workflows/.
- First read these rules:
  - {PROJECT_ROOT}/.agents/rules/research-standards.md
  - {PROJECT_ROOT}/.agents/rules/coding-standards.md
  - {PROJECT_ROOT}/.agents/rules/mcp-environments.md
- For skill discovery, scan descriptions with:
  grep -r "^description:" {PROJECT_ROOT}/skills/*/SKILL.md
- For end-to-end protocols, inspect:
  find {PROJECT_ROOT}/.agents/workflows -maxdepth 2 -type f
- Read the full SKILL.md or workflow file before following it.

If your current workspace IS {PROJECT_ROOT} or one of its subdirectories, ignore this global reference because the project-local AGENTS.md and project skills are already available.
"""


def _upsert_marked_block(path: Path, marker: str, block_content: str) -> None:
    """Append or replace a marked instruction block in a markdown file."""
    content = ""
    if path.exists():
        content = path.read_text()

    if marker in content:
        before = content.split(marker, 1)[0].rstrip()
        content = before + "\n\n" + block_content
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n" + block_content

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.lstrip())


def _symlink_target(path: Path) -> Path:
    """Return the absolute target path for a symlink."""
    target = path.readlink()
    if not target.is_absolute():
        target = path.parent / target
    return target


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is inside parent without requiring it to exist."""
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _remove_stale_project_skill_symlinks(
    skills_dir: Path,
    project_skills_dir: Path,
) -> int:
    """Remove global skill symlinks that point to removed project skills."""
    removed = 0
    if not skills_dir.exists():
        return removed

    for global_skill in skills_dir.iterdir():
        if not global_skill.is_symlink():
            continue

        target = _symlink_target(global_skill)
        if _is_relative_to(target, project_skills_dir) and not target.exists():
            global_skill.unlink()
            removed += 1

    return removed


def _reset_directory_symlink(link_path: Path, target_path: Path) -> str:
    """Replace a path with a directory symlink and report the action."""
    action = "Created"
    if link_path.exists() or link_path.is_symlink():
        action = "Refreshed" if link_path.is_symlink() else "Replaced"
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()

    link_path.symlink_to(target_path, target_is_directory=True)
    return action


def _link_claude_project_skills() -> None:
    """Expose project skills to Claude Code as native project-scope skills.

    Claude Code discovers skills as `.claude/skills/<name>/SKILL.md`. Every
    skill in `skills/` is symlinked into `.claude/skills/` so the
    repository stays the single source of truth — no copies to keep in sync.
    Only the `name` and `description` frontmatter of each SKILL.md is loaded
    into the system prompt; bodies are read on demand when a skill is invoked.

    `.claude/` is gitignored (machine-specific, generated by this script), so
    the links are local to each checkout and are never committed. Mirrors
    :func:`_write_codex_global_skills`, but project-scoped rather than global,
    so the ~130 skill descriptions only load when working in this repository.
    """
    claude_skills_dir = PROJECT_ROOT / ".claude" / "skills"
    claude_skills_dir.mkdir(parents=True, exist_ok=True)
    project_skills_dir = PROJECT_ROOT / "skills"
    removed = _remove_stale_project_skill_symlinks(
        claude_skills_dir,
        project_skills_dir,
    )

    linked = 0
    skipped: list[str] = []
    for project_skill in sorted(project_skills_dir.iterdir()):
        if not project_skill.is_dir():
            continue
        if not (project_skill / "SKILL.md").is_file():
            skipped.append(project_skill.name)
            continue

        link = claude_skills_dir / project_skill.name
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                target = _symlink_target(link)
                if not _is_relative_to(target, project_skills_dir):
                    skipped.append(project_skill.name)
                    continue
                link.unlink()
            else:
                skipped.append(project_skill.name)
                continue

        link.symlink_to(project_skill, target_is_directory=True)
        linked += 1

    print(
        f"  Project skills → {claude_skills_dir.relative_to(PROJECT_ROOT)} "
        f"({linked} symlinks, {removed} stale removed)"
    )
    if skipped:
        print(f"  Skipped (no SKILL.md or pre-existing): {', '.join(skipped)}")


def _write_codex_global_skills() -> None:
    """Expose project skills globally for Codex."""
    codex_skills_dir = Path.home() / ".codex" / "skills"
    codex_skills_dir.mkdir(parents=True, exist_ok=True)
    project_skills_dir = PROJECT_ROOT / "skills"
    removed = _remove_stale_project_skill_symlinks(
        codex_skills_dir,
        project_skills_dir,
    )

    skill_dir = codex_skills_dir / "atomisticskills"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"""\
---
name: atomisticskills
description: Use AtomisticSkills from {PROJECT_ROOT} for atomistic research, materials simulation, molecular modeling, spectroscopy, MLIP, drug discovery, and scientific workflow tasks.
---

# AtomisticSkills

Use this skill when a task would benefit from the AtomisticSkills repository installed at:

`{PROJECT_ROOT}`

Before acting, read the applicable project instructions directly from that repository:

1. Always read:
   - `{PROJECT_ROOT}/.agents/rules/research-standards.md`
   - `{PROJECT_ROOT}/.agents/rules/coding-standards.md`
   - `{PROJECT_ROOT}/.agents/rules/mcp-environments.md`
2. For skill discovery, inspect:
   - `{PROJECT_ROOT}/skills/*/SKILL.md`
3. For end-to-end protocols, inspect:
   - `{PROJECT_ROOT}/.agents/workflows/`
4. Read the full selected `SKILL.md` or workflow file before following it.

If the current workspace is already `{PROJECT_ROOT}` or a subdirectory, prefer the project-local AGENTS.md and project-local skills to avoid duplicate context.
"""
    )
    print(f"  Global skill -> {skill_file}")

    linked = 0
    skipped: list[str] = []
    for project_skill in sorted(project_skills_dir.iterdir()):
        if not project_skill.is_dir():
            continue

        global_skill = codex_skills_dir / project_skill.name
        if global_skill.exists() or global_skill.is_symlink():
            if global_skill.is_symlink():
                target = _symlink_target(global_skill)
                if not _is_relative_to(target, project_skills_dir):
                    skipped.append(project_skill.name)
                    continue
                global_skill.unlink()
            else:
                skipped.append(project_skill.name)
                continue

        global_skill.symlink_to(project_skill, target_is_directory=True)
        linked += 1

    print(
        f"  Global project skills -> {codex_skills_dir} "
        f"({linked} symlinks, {removed} stale removed)"
    )
    if skipped:
        separator = ", "
        print(f"  Skipped existing non-symlink skills: {separator.join(skipped)}")


# ---------------------------------------------------------------------------
# Conda detection
# ---------------------------------------------------------------------------


def detect_conda_base() -> str | None:
    for cmd in ("conda", "mamba", "micromamba"):
        try:
            result = subprocess.run(
                [cmd, "info", "--base"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                base = result.stdout.strip()
                if base and Path(base).is_dir():
                    return base
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    for name in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        candidate = Path.home() / name
        if candidate.is_dir():
            return str(candidate)

    return None


# ---------------------------------------------------------------------------
# MCP config loading and path patching
# ---------------------------------------------------------------------------


def load_mcp_servers(conda_base: str) -> dict[str, Any]:
    """Load mcp_config.json and rewrite conda env paths for this machine."""
    with open(MCP_SOURCE) as fh:
        config = json.load(fh)

    project_root = str(PROJECT_ROOT)

    for server in config.get("mcpServers", {}).values():
        match = ENV_PATTERN.match(server.get("command", ""))
        if match:
            env_name = match.group(1)
            server["command"] = f"{conda_base}/envs/{env_name}/bin/python"
        env = server.get("env", {})
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = project_root
        # Rewrite CONDA_PREFIX so Triton's ptxas-blackwell fallback resolves
        # correctly on Blackwell+ GPUs even when the MCP server is launched
        # without full conda activation (no PATH / CONDA_PREFIX from conda init).
        if "CONDA_PREFIX" in env and match:
            env["CONDA_PREFIX"] = f"{conda_base}/envs/{env_name}"
        # Explicit Triton ptxas-blackwell path: more direct than CONDA_PREFIX
        # fallback. Required on Blackwell GPUs (sm_100+, compute capability ≥ 12.0)
        # where torch.compile triggers Triton JIT compilation via nvalchemi hooks.
        if "TRITON_PTXAS_BLACKWELL_PATH" in env and match:
            env["TRITON_PTXAS_BLACKWELL_PATH"] = (
                f"{conda_base}/envs/{env_name}/bin/ptxas"
            )
        # Rewrite PATH: replace the placeholder conda env bin dir so that
        # shutil.which('ptxas-blackwell') resolves correctly in MCP server
        # processes that do not have full conda activation.
        if "PATH" in env and match:
            env_bin = f"{conda_base}/envs/{env_name}/bin"
            # Replace any existing envs/<name>/bin prefix in PATH
            import re as _re

            env["PATH"] = _re.sub(
                r"[^ ]*?/envs/[^/]+/bin",
                env_bin,
                env["PATH"],
                count=1,
            )

    return config.get("mcpServers", {})


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------

KNOWN_AGENTS = ["claude", "codex", "gemini", "cursor", "windsurf"]


def detect_agents() -> list[str]:
    """Return list of agent names that appear to be installed."""
    found = []

    # Claude Code
    if shutil.which("claude"):
        found.append("claude")

    # OpenAI Codex CLI
    if shutil.which("codex"):
        found.append("codex")

    # Gemini CLI
    if shutil.which("gemini"):
        found.append("gemini")

    # Cursor — check for global config dir or project dir
    cursor_global = Path.home() / ".cursor"
    if cursor_global.is_dir() or (PROJECT_ROOT / ".cursor").is_dir():
        found.append("cursor")

    # Windsurf
    windsurf_global = Path.home() / ".codeium" / "windsurf"
    if windsurf_global.is_dir():
        found.append("windsurf")

    return found


# ---------------------------------------------------------------------------
# Per-agent writers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict, *, merge_key: str | None = None) -> None:
    """Write JSON, optionally merging into an existing file under merge_key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    if merge_key:
        existing.setdefault(merge_key, {})
        existing[merge_key].update(data)
        data = existing
    else:
        existing.update(data)
        data = existing

    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _write_instruction_file(path: Path, source: Path) -> None:
    """Write agent instruction file if it doesn't already exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # If the source (e.g. CLAUDE.md) exists, symlink; otherwise write stub.
    if source.exists():
        # Write a stub that points to CLAUDE.md rather than symlinking,
        # so the file is portable across checkouts.
        path.write_text(f"# See CLAUDE.md for full instructions\n\n{INSTRUCTION_STUB}")
    else:
        path.write_text(INSTRUCTION_STUB)
    print(f"  Created {path.relative_to(PROJECT_ROOT)}")


def configure_claude(servers: dict, scope: str) -> None:
    """Claude Code: .mcp.json (project) or ~/.claude.json (global user scope)."""
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".mcp.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        # Global user scope: ~/.claude.json (same file as `claude mcp add --scope user`)
        path = Path.home() / ".claude.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Global MCP  → {path}")

    # Register skills as native Claude Code project skills.
    _link_claude_project_skills()

    # CLAUDE.md already exists — nothing to do for instruction file.


def configure_codex(servers: dict, scope: str) -> None:
    """Codex CLI: .codex/config.toml (TOML format)."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # backport
        except ImportError:
            tomllib = None

    try:
        import tomli_w
    except ImportError:
        tomli_w = None

    def _toml_key(key: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+", key):
            return key
        return json.dumps(key)

    def _toml_path(prefix: str, key: str) -> str:
        key_part = _toml_key(key)
        return f"{prefix}.{key_part}" if prefix else key_part

    def _toml_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(_toml_value(item) for item in value) + "]"
        return json.dumps(str(value))

    def _emit_toml_table(lines: list[str], prefix: str, table: dict) -> None:
        scalar_items = [
            (key, value) for key, value in table.items() if not isinstance(value, dict)
        ]
        child_items = [
            (key, value) for key, value in table.items() if isinstance(value, dict)
        ]

        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalar_items:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        if scalar_items and child_items:
            lines.append("")

        for index, (key, value) in enumerate(child_items):
            if lines and lines[-1] != "":
                lines.append("")
            child_prefix = _toml_path(prefix, key)
            _emit_toml_table(lines, child_prefix, value)
            if index != len(child_items) - 1 and lines[-1] != "":
                lines.append("")

    def _write_toml(path: Path, servers: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if path.exists() and tomllib:
            try:
                with open(path, "rb") as fh:
                    existing = tomllib.load(fh)
            except Exception:
                pass

        existing.setdefault("mcp_servers", {})
        for name, cfg in servers.items():
            existing["mcp_servers"][name] = {
                "command": cfg["command"],
                "args": cfg.get("args", []),
                **({"env": cfg["env"]} if cfg.get("env") else {}),
            }

        if tomli_w:
            with open(path, "wb") as fh:
                tomli_w.dump(existing, fh)
        else:
            lines: list[str] = []
            _emit_toml_table(lines, "", existing)
            path.write_text("\n".join(lines).rstrip() + "\n")

    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".codex" / "config.toml"
        _write_toml(path, servers)
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".codex" / "config.toml"
        _write_toml(path, servers)
        print(f"  Global MCP  → {path}")

        global_agents = Path.home() / ".codex" / "AGENTS.md"
        _upsert_marked_block(
            global_agents,
            ATOMISTICSKILLS_GLOBAL_MARKER,
            _global_reference_block(),
        )
        print(f"  Global instructions → {global_agents}")
        _write_codex_global_skills()

    _write_instruction_file(
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "CLAUDE.md",
    )


def configure_gemini(servers: dict, scope: str) -> None:
    """Gemini CLI and IDE: settings.json, mcp_config.json, and custom plugin."""
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".gemini" / "settings.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        # Gemini CLI settings
        cli_path = Path.home() / ".gemini" / "settings.json"
        _write_json(cli_path, servers, merge_key="mcpServers")
        print(f"  Global MCP (CLI) → {cli_path}")

        # Gemini IDE settings
        ide_path = Path.home() / ".gemini" / "config" / "mcp_config.json"
        _write_json(ide_path, servers, merge_key="mcpServers")
        print(f"  Global MCP (IDE) → {ide_path}")

        # Setup IDE Plugin for global skills
        plugin_dir = (
            Path.home()
            / ".gemini"
            / "config"
            / "plugins"
            / "Google.atomisticskills.atomisticskills"
        )
        plugin_dir.mkdir(parents=True, exist_ok=True)

        plugin_json = plugin_dir / "plugin.json"
        plugin_data = {
            "name": "atomisticskills",
            "description": "AtomisticSkills workspace plugin",
            "disabled": False,
        }
        with open(plugin_json, "w") as fh:
            json.dump(plugin_data, fh)
            fh.write("\n")
        print(f"  Created plugin config → {plugin_json}")

        skills_symlink = plugin_dir / "skills"
        target_skills = PROJECT_ROOT / "skills"
        symlink_action = _reset_directory_symlink(skills_symlink, target_skills)
        print(
            f"  {symlink_action} skills symlink → {skills_symlink} to {target_skills}"
        )

        # Global Instructions in ~/.gemini/GEMINI.md
        global_md = Path.home() / ".gemini" / "GEMINI.md"
        marker = "# AtomisticSkills Global Reference"
        block_content = f"""\
{marker}

If your current workspace is NOT {PROJECT_ROOT} (or any of its subdirectories), and you need to perform atomistic research, materials discovery, molecular simulation, or related tasks:
- The AtomisticSkills repository is installed at {PROJECT_ROOT}.
- You can access its Skills at {PROJECT_ROOT}/skills/ and workflows at {PROJECT_ROOT}/.agents/workflows/.
- Discover skills by running: grep -r "^description:" {PROJECT_ROOT}/skills/*/SKILL.md
- Read and follow these rules from the AtomisticSkills repo:
  - [research-standards.md](file://{PROJECT_ROOT}/.agents/rules/research-standards.md)
  - [coding-standards.md](file://{PROJECT_ROOT}/.agents/rules/coding-standards.md)
  - [mcp-environments.md](file://{PROJECT_ROOT}/.agents/rules/mcp-environments.md)
  - [skill-standards.md](file://{PROJECT_ROOT}/.agents/rules/skill-standards.md)
  - [workflow-standards.md](file://{PROJECT_ROOT}/.agents/rules/workflow-standards.md)
  - [plot-standards.md](file://{PROJECT_ROOT}/.agents/rules/plot-standards.md)
- NOTE: If you are already inside the {PROJECT_ROOT} directory, ignore this section to avoid loading duplicate rules or context.
"""
        content = ""
        if global_md.exists():
            content = global_md.read_text()

        if marker in content:
            parts = content.split(marker)
            before = parts[0].rstrip()
            content = before + "\n\n" + block_content
        else:
            if content and not content.endswith("\n"):
                content += "\n"
            content += "\n" + block_content

        global_md.parent.mkdir(parents=True, exist_ok=True)
        global_md.write_text(content)
        print(f"  Updated global instructions → {global_md}")

    _write_instruction_file(
        PROJECT_ROOT / "GEMINI.md",
        PROJECT_ROOT / "CLAUDE.md",
    )


def configure_cursor(servers: dict, scope: str) -> None:
    """Cursor: .cursor/mcp.json (project) or ~/.cursor/mcp.json (global)."""
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".cursor" / "mcp.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".cursor" / "mcp.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Global MCP  → {path}")

    # Cursor uses .cursor/rules/*.mdc — write a minimal rule file if missing.
    rules_dir = PROJECT_ROOT / ".cursor" / "rules"
    rules_file = rules_dir / "atomisticskills.mdc"
    if not rules_file.exists():
        rules_dir.mkdir(parents=True, exist_ok=True)
        rules_file.write_text(
            "---\ndescription: AtomisticSkills research agent rules\nalwaysApply: true\n---\n\n"
            + INSTRUCTION_STUB
        )
        print(f"  Created {rules_file.relative_to(PROJECT_ROOT)}")


def configure_windsurf(servers: dict, scope: str) -> None:
    """Windsurf: global only (~/.codeium/windsurf/mcp_config.json)."""
    if scope == "project":
        print(
            "  Windsurf does not support project-level MCP config. "
            "Use --scope global or --scope both."
        )
        return

    path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    _write_json(path, servers, merge_key="mcpServers")
    print(f"  Global MCP  → {path}")

    # Windsurf reads .windsurfrules at project root.
    _write_instruction_file(
        PROJECT_ROOT / ".windsurfrules",
        PROJECT_ROOT / "CLAUDE.md",
    )


AGENT_WRITERS = {
    "claude": configure_claude,
    "codex": configure_codex,
    "gemini": configure_gemini,
    "cursor": configure_cursor,
    "windsurf": configure_windsurf,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure AtomisticSkills MCP servers for AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--agent",
        "-a",
        nargs="+",
        choices=KNOWN_AGENTS + ["all"],
        default=None,
        metavar="AGENT",
        help=f"Agent(s) to configure: {', '.join(KNOWN_AGENTS)}, all "
        "(default: auto-detect installed agents)",
    )
    parser.add_argument(
        "--conda",
        default=None,
        metavar="PATH",
        help="Path to conda/mamba base directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--scope",
        choices=["project", "global", "both"],
        default="project",
        help="Where to write config: project dir, global user dir, or both "
        "(default: project).",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="Detect and print installed agents, then exit.",
    )
    args = parser.parse_args()

    if args.list_agents:
        detected = detect_agents()
        if detected:
            print("Detected agents:", ", ".join(detected))
        else:
            print("No supported agents detected.")
        return

    # Resolve conda base
    conda_base: str | None = args.conda
    if conda_base is not None:
        if not Path(conda_base).is_dir():
            print(f"Error: {conda_base} is not a valid directory.", file=sys.stderr)
            sys.exit(1)
    else:
        conda_base = detect_conda_base()
        if conda_base is None:
            print(
                "Error: Could not auto-detect a conda/mamba installation.\n"
                "Provide the base path explicitly: --conda /path/to/miniforge3",
                file=sys.stderr,
            )
            sys.exit(1)

    if not MCP_SOURCE.exists():
        print(f"Error: {MCP_SOURCE} not found.", file=sys.stderr)
        sys.exit(1)

    servers = load_mcp_servers(conda_base)

    # Resolve agent list
    if args.agent is None:
        agents = detect_agents()
        if not agents:
            print(
                "No supported agents auto-detected. "
                "Specify one with --agent (claude, codex, gemini, cursor, windsurf).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected agents: {', '.join(agents)}")
    elif "all" in args.agent:
        agents = KNOWN_AGENTS
    else:
        agents = args.agent

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Conda base   : {conda_base}")
    print(f"Scope        : {args.scope}")
    print()

    for agent in agents:
        print(f"[{agent}]")
        AGENT_WRITERS[agent](servers, args.scope)
        print()

    print("Done. Skills (skills/) and workflows (.agents/workflows/)")
    print("are the cross-platform standard path — no changes needed there.")


if __name__ == "__main__":
    main()
