"""Rewrite skill invocations from conda environments to uv projects.

Skills annotated a code block with ``# Env: <conda-env>`` and then ran a bare
``python script.py``. That only works if the reader has already activated the
environment, which a plugin user has not and cannot: the containers expose MCP
servers, not conda environments, so 112 of 131 skills failed at their first
script step with ``EnvironmentNameNotFound``.

The replacement names the environment in the command itself::

    # Env: mace-agent                  # Venv: venv/mlip
    python skills/x/scripts/y.py  ->   uv run --project venv/mlip python skills/x/scripts/y.py

``uv run`` needs no activation and builds the environment from the lock on
first use, so the same line works for a developer, a plugin user and CI.

Most blocks are a bare ``python`` call and convert mechanically. A minority
start with ``cd``, ``bash``, an inline ``import``, or a comment, and those are
reported rather than guessed at -- a wrong rewrite of a working command is
worse than a listed one.

Usage:
    # Env: base-agent
    python tools/migrate_skill_envs.py --check    # report only, change nothing
    python tools/migrate_skill_envs.py            # rewrite in place

Requirements:
    - Conda environment: base-agent (standard library only)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS = PROJECT_ROOT / "skills"

# Where each former conda environment now lives. Derived from venv/README.md;
# keep the two in step.
ENV_TO_PROJECT = {
    "base-agent": "cpu",
    "drugdisc-agent": "cpu",
    "smol-agent": "cpu",
    "atomate2-agent": "cpu",
    "nmr-agent": "cpu",
    "phasefield-agent": "cpu",
    "calphad-agent": "cpu",
    "xrd-agent": "cpu",
    "drugmd-agent": "cpu",
    "orca-agent": "cpu",
    "atomistic-agent": "cpu",
    "mace-agent": "mlip",
    "matgl-agent": "mlip",
    "scd-agent": "mlip",
    "react-ot-agent": "mlip",
    "ms-gen": "mlip",
    "fairchem-agent": "fairchem",
    # "any" means the block does not care; the CPU stack is the cheapest.
    "any": "cpu",
}

# The generative stack has no uv project: mattergen hard-pins +cu118 wheels
# that never existed for aarch64, and ADiT/DiffCSP++ are not published
# packages. Those blocks keep their annotation and are reported.
CONTAINER_ONLY = {"adit-agent", "diffcsp-agent", "mattergen-agent"}

# The trailing group matters: several annotations carry prose after the name,
# e.g. "# Env: xrd-agent. Quote the path because of (PO3)." Requiring the line
# to end at the env name silently skipped eight of them.
ENV_LINE = re.compile(r"^(?P<indent>\s*)#\s*Env:\s*(?P<env>[\w-]+)(?P<rest>.*)$")
# Already-migrated annotations are matched too, so that a block whose
# annotation was created by the conda pass still gets its command prefixed.
# This is what makes repeated runs converge instead of leaving bare `python`.
VENV_LINE = re.compile(
    r"^(?P<indent>\s*)#\s*Venv:\s*venv/(?P<project>[\w-]+)(?P<rest>.*)$"
)


def retarget_prose(rest: str) -> str:
    """Rewrite any other conda env names mentioned in the trailing prose.

    Left alone, a note like "(or matgl-agent, fairchem-agent)" would keep
    pointing at environments that no longer exist.
    """
    for env, proj in sorted(ENV_TO_PROJECT.items(), key=lambda kv: -len(kv[0])):
        rest = re.sub(rf"\b{re.escape(env)}\b", f"venv/{proj}", rest)
    return rest


FENCE = re.compile(r"^\s*```")
# Console scripts provided by the projects themselves; these need the
# environment on PATH exactly as a python call does.
ENV_BINARIES = ("obabel", "lobsterpy", "pymol", "vina", "packmol")
CONDA_RUN = re.compile(r"^(?P<i>\s*)conda run -n [\w-]+ (?P<rest>.*)$")


def migrate(path: Path, apply: bool) -> tuple[list[str], list[str]]:
    """Rewrite one SKILL.md.

    Works on the fenced block that the annotation introduces, not just the
    following line: blocks legitimately begin with ``cd``, ``export``, an
    inline ``MP_API_KEY=...`` or a ``for`` loop, and the command that actually
    needs the environment appears further down.
    """
    lines = path.read_text().split("\n")
    changes: list[str] = []
    residue: list[str] = []

    # Index the fenced blocks once so an annotation can be tied to its block.
    fences = [i for i, ln in enumerate(lines) if FENCE.match(ln)]
    blocks = list(zip(fences[::2], fences[1::2]))

    def block_of(idx: int) -> tuple[int, int] | None:
        for start, end in blocks:
            if start < idx < end:
                return start, end
        return None

    for i, line in enumerate(lines):
        m = ENV_LINE.match(line)
        already = VENV_LINE.match(line) if not m else None
        if already:
            env, indent, rest = None, already.group("indent"), already.group("rest")
            project = already.group("project")
        elif m:
            env, indent, rest = m.group("env"), m.group("indent"), m.group("rest")
            project = None
        else:
            continue

        if env is not None and env in CONTAINER_ONLY:
            residue.append(f"{path.parent.name}:{i + 1} {env} (container-only stack)")
            continue
        if project is None:
            project = ENV_TO_PROJECT.get(env)
        if project is None:
            residue.append(f"{path.parent.name}:{i + 1} {env} (no project mapped)")
            continue

        prefix = f"uv run --project venv/{project} "
        span = block_of(i)
        scope = range(i + 1, span[1]) if span else range(i + 1, min(i + 6, len(lines)))

        rewrote = 0
        for j in scope:
            s = lines[j]
            st = s.lstrip()
            ind = s[: len(s) - len(st)]
            if st.startswith(prefix) or st.startswith("uv run "):
                rewrote += 1  # already migrated; keep idempotent
                continue
            cr = CONDA_RUN.match(s)
            if cr:
                lines[j] = f"{cr.group('i')}{prefix}{cr.group('rest')}"
                rewrote += 1
                continue
            if st.startswith("python "):
                lines[j] = f"{ind}{prefix}{st}"
                rewrote += 1
                continue
            # `KEY=value python ...` keeps the assignment ahead of uv run.
            inline = re.match(
                r"^(?P<env>(?:[A-Z_][A-Z0-9_]*=\S*\s+)+)(?P<cmd>python\s.*)$", st
            )
            if inline:
                lines[j] = f"{ind}{inline.group('env')}{prefix}{inline.group('cmd')}"
                rewrote += 1
                continue
            if st.split(" ")[0] in ENV_BINARIES:
                lines[j] = f"{ind}{prefix}{st}"
                rewrote += 1
                continue

        lines[i] = f"{indent}# Venv: venv/{project}{retarget_prose(rest)}"
        if already:
            # Nothing to report: the block was already migrated. Counting it
            # would make a second run look like it changed 347 things.
            continue
        if rewrote:
            changes.append(
                f"{path.parent.name}:{i + 1} {env} -> {project} ({rewrote} cmd)"
            )
        else:
            # An inline python snippet, or an MCP tool call: the annotation
            # tells the reader which project to use and there is no shell
            # command to prefix. Correct, but listed so it can be eyeballed.
            first = "?"
            for j in scope:
                if lines[j].strip() and not lines[j].lstrip().startswith("#"):
                    first = lines[j].strip().split(" ")[0]
                    break
            residue.append(
                f"{path.parent.name}:{i + 1} {env} -> {project}, no shell command (starts {first!r})"
            )

    if apply:
        path.write_text("\n".join(lines))
    return changes, residue


ACTIVATE = re.compile(r"^(?P<i>\s*)(?:conda|mamba) activate (?P<env>[\w-]+)\s*$")
CONDA_RUN_ANY = re.compile(r"(?P<pre>\s*)conda run -n (?P<env>[\w-]+) (?P<rest>.*)$")


def migrate_conda_commands(path: Path, apply: bool) -> tuple[list[str], list[str]]:
    """Convert `conda run -n` and `conda activate` anywhere in the file.

    These appear in blocks that never carried an ``# Env:`` annotation, so the
    annotation-driven pass does not see them. They are just as broken for a
    plugin user: the environments they name no longer exist.
    """
    lines = path.read_text().split("\n")
    changes: list[str] = []
    residue: list[str] = []
    out: list[str] = []

    for i, line in enumerate(lines):
        m = CONDA_RUN_ANY.match(line)
        if m:
            env = m.group("env")
            proj = ENV_TO_PROJECT.get(env)
            if proj is None:
                residue.append(
                    f"{path.parent.name}:{i + 1} conda run -n {env} (unmapped)"
                )
                out.append(line)
                continue
            out.append(
                f"{m.group('pre')}uv run --project venv/{proj} {m.group('rest')}"
            )
            changes.append(
                f"{path.parent.name}:{i + 1} conda run -n {env} -> venv/{proj}"
            )
            continue

        a = ACTIVATE.match(line)
        if a:
            env = a.group("env")
            if env in CONTAINER_ONLY:
                residue.append(
                    f"{path.parent.name}:{i + 1} activate {env} (container-only)"
                )
                out.append(line)
                continue
            proj = ENV_TO_PROJECT.get(env)
            if proj is None:
                residue.append(f"{path.parent.name}:{i + 1} activate {env} (unmapped)")
                out.append(line)
                continue
            # uv needs no activation step; the project moves onto the command
            # itself, which the annotation pass has already done or will do.
            out.append(f"{a.group('i')}# Venv: venv/{proj}")
            changes.append(f"{path.parent.name}:{i + 1} activate {env} -> venv/{proj}")
            continue

        out.append(line)

    if apply and changes:
        path.write_text("\n".join(out))
    return changes, residue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="report without writing anything"
    )
    args = parser.parse_args()

    all_changes: list[str] = []
    all_residue: list[str] = []
    touched = 0
    for skill in sorted(SKILLS.glob("*/SKILL.md")):
        changes, residue = migrate(skill, apply=not args.check)
        c2, r2 = migrate_conda_commands(skill, apply=not args.check)
        changes += c2
        residue += r2
        if changes or residue:
            touched += 1
        all_changes += changes
        all_residue += residue

    verb = "would rewrite" if args.check else "rewrote"
    print(f"{verb} {len(all_changes)} invocations across {touched} skills")
    if all_residue:
        print(f"\n{len(all_residue)} blocks need review:")
        for r in all_residue:
            print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
