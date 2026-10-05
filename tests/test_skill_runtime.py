"""Every SKILL.md must give commands that run wherever the skill is installed.

A skill is read by an agent that may have it from a plugin cache, a project's
.claude/skills symlink or a plain checkout, and that may or may not have the
MCP servers connected. These tests hold each SKILL.md to the conventions that
make all of those work:

* commands run through ``${CLAUDE_SKILL_DIR}/../../venv/run <venv>[+extra]``,
  which picks uv or a container runtime, with paths relative to the skill;
* the venvs and extras named exist, and ``metadata.venv`` lists exactly the
  venvs a skill uses;
* scripts a command names exist;
* nothing points at the repository root or at a conda environment, except
  for the few stacks that have no uv project (``metadata.conda_env``);
* every MCP tool written ``server.tool`` exists on that server.

``tools/migrate_skill_commands.py`` rewrites skills into this form, and the last
test checks that running it again would change nothing.

Requirements:
    - Environment: cpu (run with: venv/run cpu python -m pytest tests/test_skill_runtime.py)
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS = sorted(p.parent for p in (PROJECT_ROOT / "skills").glob("*/SKILL.md"))
VENV_DIR = PROJECT_ROOT / "venv"
LAUNCHER = "${CLAUDE_SKILL_DIR}/../../venv/run"
LAUNCH = re.compile(re.escape(LAUNCHER) + r" (?P<spec>\w[\w+-]*)")
ALLOWED_TOP_LEVEL = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}
FENCE = re.compile(r"^\s*(```|~~~)")


def frontmatter(skill: Path) -> dict:
    text = (skill / "SKILL.md").read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, f"{skill.name}: no frontmatter"
    return yaml.safe_load(m.group(1))


def code_lines(skill: Path) -> list[str]:
    """Lines inside fenced code blocks."""
    out, inside = [], False
    for line in (skill / "SKILL.md").read_text().split("\n"):
        if FENCE.match(line):
            inside = not inside
            continue
        if inside:
            out.append(line)
    return out


def venvs() -> dict[str, set[str]]:
    """{venv: extras it declares}."""
    out = {}
    for pyproject in VENV_DIR.glob("*/pyproject.toml"):
        data = tomllib.loads(pyproject.read_text())
        out[pyproject.parent.name] = set(
            data["project"].get("optional-dependencies", {})
        )
    return out


def server_tools() -> dict[str, set[str]]:
    """{server: tool names}, read statically from src/mcp_server/*_server.py."""
    out = {}
    for path in (PROJECT_ROOT / "src" / "mcp_server").glob("*_server.py"):
        tree = ast.parse(path.read_text())
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool"
                for d in node.decorator_list
            )
        }
        out[path.stem.removesuffix("_server")] = names
    return out


@pytest.fixture(scope="module")
def known_venvs():
    return venvs()


ids = [s.name for s in SKILLS]


@pytest.mark.parametrize("skill", SKILLS, ids=ids)
def test_frontmatter_keys_are_spec_compliant(skill):
    """Extra top-level keys break importing the repository into claude.ai."""
    keys = set(frontmatter(skill))
    assert keys <= ALLOWED_TOP_LEVEL, f"non-spec keys: {keys - ALLOWED_TOP_LEVEL}"


@pytest.mark.parametrize("skill", SKILLS, ids=ids)
def test_commands_name_real_venvs_and_extras(skill, known_venvs):
    # The whole file: commands also appear in inline code and in the MCP note.
    for line in (skill / "SKILL.md").read_text().split("\n"):
        for m in LAUNCH.finditer(line):
            venv, *extras = m.group("spec").split("+")
            assert venv in known_venvs, f"unknown venv {venv!r}: {line.strip()}"
            unknown = set(extras) - known_venvs[venv]
            assert not unknown, f"unknown extras {unknown} for {venv}: {line.strip()}"


@pytest.mark.parametrize("skill", SKILLS, ids=ids)
def test_metadata_venv_matches_the_commands(skill):
    meta = frontmatter(skill).get("metadata", {})
    # Counted over the whole file: the MCP note's shell fallbacks are needs too.
    used = sorted(
        {
            m.group("spec").split("+")[0]
            for m in LAUNCH.finditer((skill / "SKILL.md").read_text())
        }
    )
    assert (
        sorted(meta.get("venv", [])) == used
    ), f"metadata.venv {meta.get('venv')} != venvs used {used}; run tools/migrate_skill_commands.py"


@pytest.mark.parametrize("skill", SKILLS, ids=ids)
def test_named_scripts_exist(skill):
    text = (skill / "SKILL.md").read_text()
    for m in re.finditer(r"\$\{CLAUDE_SKILL_DIR\}(?P<rel>/[\w./-]+\.(?:py|sh))", text):
        path = (skill / m.group("rel").lstrip("/")).resolve()
        assert path.is_file(), f"{m.group(0)} does not exist ({path})"


@pytest.mark.parametrize("skill", SKILLS, ids=ids)
def test_no_repository_relative_or_conda_commands(skill):
    conda_env = frontmatter(skill).get("metadata", {}).get("conda_env")
    problems = []
    for line in code_lines(skill):
        stripped = line.strip()
        if re.search(
            r"(^|\s)(?:python3?\s+)?(?:\./)?(?:\.agents/)?skills/[a-z0-9-]+/", stripped
        ):
            problems.append(f"repository-relative path: {stripped[:90]}")
        if "uv run --project" in stripped:
            problems.append(f"bare uv command (use venv/run): {stripped[:90]}")
        if re.search(r"\b(conda|mamba) (run|activate)\b", stripped) and not conda_env:
            problems.append(f"conda command in a uv skill: {stripped[:90]}")
        if re.match(r"#\s*(Env|Venv):", stripped):
            problems.append(f"obsolete annotation: {stripped[:90]}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("skill", SKILLS, ids=ids)
def test_mcp_tools_exist(skill):
    """`server.tool` names must be real; stale names send the agent nowhere."""
    tools = server_tools()
    text = (skill / "SKILL.md").read_text()
    if "mcp-tools-note" not in text:
        return
    missing = set()
    for m in re.finditer(
        r"(?<![\w.])(?P<server>" + "|".join(tools) + r")\.(?P<tool>[a-z_][a-z0-9_]*)\(",
        text,
    ):
        if m.group("tool") not in tools[m.group("server")]:
            missing.add(f"{m.group('server')}.{m.group('tool')}")
    for m in re.finditer(
        r"`(?P<server>" + "|".join(tools) + r")\.(?P<tool>[a-z_][a-z0-9_]*)`", text
    ):
        if m.group("tool") not in tools[m.group("server")]:
            missing.add(f"{m.group('server')}.{m.group('tool')}")
    assert not missing, f"unknown MCP tools: {sorted(missing)}"


def test_conda_skills_are_declared():
    """A conda environment is the exception, and must be named where it is used."""
    for skill in SKILLS:
        meta = frontmatter(skill).get("metadata", {})
        uses_conda = any(re.search(r"\bconda run\b", ln) for ln in code_lines(skill))
        if uses_conda:
            assert meta.get(
                "conda_env"
            ), f"{skill.name} runs conda but declares no metadata.conda_env"


def test_migration_tool_converges():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "migrate_skill_commands.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
