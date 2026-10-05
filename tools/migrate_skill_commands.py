#!/usr/bin/env python3
"""Make every SKILL.md command runnable wherever the skill is installed.

Skills used to give commands relative to the repository root::

    uv run --project venv/mlip python skills/mat-phonon/scripts/run.py

That works in a checkout opened at its root and nowhere else. Installed as a
Claude Code plugin, the skill lives in the plugin cache while the agent works in
the user's project, so neither ``venv/mlip`` nor ``skills/...`` resolves. Claude
Code substitutes ``${CLAUDE_SKILL_DIR}`` -- the directory holding the SKILL.md --
when it loads a skill (project, personal and plugin skills alike), so commands
are rewritten relative to that::

    ${CLAUDE_SKILL_DIR}/../../venv/run mlip python ${CLAUDE_SKILL_DIR}/scripts/run.py

``venv/run`` picks uv or a container runtime, so the same line also works on a
host that cannot install the environment natively.

The same pass:

* adds ``+openmm``, ``+pymol`` or ``+docking`` when the invoked script imports a
  package that lives in that optional extra;
* renames MCP tools from ``mcp_<server>_<tool>`` to ``<server>.<tool>`` and adds
  a note saying how to call them with or without an MCP connection;
* records the environments a skill uses as ``metadata.venv`` in its frontmatter.

Skills whose stack has no uv project (React-OT, ICEBERG, the generative models)
keep a conda environment, named in ``metadata.conda_env``.

Usage:
    python tools/migrate_skill_commands.py --check   # report, change nothing
    python tools/migrate_skill_commands.py           # rewrite in place

Re-running is a no-op once a file is migrated.

Requirements:
    - Python 3.10+, standard library only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS = PROJECT_ROOT / "skills"
SERVERS_TABLE = PROJECT_ROOT / "venv" / "servers.tsv"

LAUNCHER = "${CLAUDE_SKILL_DIR}/../../venv/run"
SKILL_DIR = "${CLAUDE_SKILL_DIR}"

# Skills whose dependencies cannot live in a uv project; their commands run in
# the named conda environment from conda-envs/.
CONDA_SKILLS = {
    "chem-msms-predict": "ms-gen",
    "chem-react-ot": "react-ot-agent",
    # LAMMPS is compiled against each MLIP's conda environment.
    "mat-lammps-md": "[mace-agent, matgl-agent, fairchem-agent]",
    "ml-generative-adit": "adit-agent",
    "ml-generative-diffcsp": "diffcsp-agent",
    "ml-generative-mattergen": "mattergen-agent",
}

# Optional extras of the uv projects, keyed by the modules that need them.
EXTRA_FOR_MODULE = {
    "openmm": "openmm",
    "pdbfixer": "openmm",
    "parmed": "openmm",
    "pymol": "pymol",
    "vina": "docking",
    "VOID": "void",
}

FENCE = re.compile(r"^\s*(```|~~~)")
VENV_ANNOTATION = re.compile(
    r"^(?P<indent>\s*)#\s*Venv:\s*venv/[\w-]+\.?\s*(?P<rest>.*)$"
)
UV_RUN = re.compile(
    r"uv run --project venv/(?P<venv>[\w-]+)(?P<extras>(?: --extra [\w-]+)*) "
)
# A skill path not already rooted somewhere else (a URL, ${CLAUDE_SKILL_DIR}, ...).
SKILL_PATH = re.compile(
    r"(?<![\w/.$}{-])(?:\./)?(?:\.agents/)?skills/(?P<name>[a-z0-9][a-z0-9-]*)/"
)
REPO_PATH = re.compile(
    r"(?<![\w/.$}{-])(?P<root>src|conda-envs|docker|tools)/(?=[\w.-])"
)
SCRIPT = re.compile(r"python\s+(?P<path>(?:\./)?(?:\.agents/)?skills/[\w./-]+\.py)")
MCP_NAME = re.compile(
    r"\bmcp_(?P<server>mace|matgl|fairchem|base|drugdisc|smol|atomate2|adit|diffcsp|mattergen)_(?P<tool>[a-z0-9_]+)"
)
NOTE_MARKER = "<!-- mcp-tools-note -->"


def load_servers() -> dict[str, str]:
    """Return {server: venv or '-'} from venv/servers.tsv."""
    out = {}
    for line in SERVERS_TABLE.read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split("\t")
            out[fields[0]] = fields[1]
    return out


def extras_for(script: str) -> list[str]:
    """Return the extras the script imports unconditionally."""
    rel = script.removeprefix("./").removeprefix(".agents/")
    path = PROJECT_ROOT / rel
    if not path.is_file():
        return []
    found = set()
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^(?P<indent>\s*)(?:import|from)\s+(?P<mod>[a-zA-Z_]\w*)", line)
        if not m or m.group("mod") not in EXTRA_FOR_MODULE:
            continue
        # An import inside try/except is an optional feature (the script runs
        # without it), so it does not justify a heavier environment.
        if m.group("indent"):
            continue
        found.add(EXTRA_FOR_MODULE[m.group("mod")])
    return sorted(found)


LAUNCHED = re.compile(
    re.escape("${CLAUDE_SKILL_DIR}/../../venv/run") + r" (?P<spec>[\w+-]+) python3? "
    r"(?P<script>\$\{CLAUDE_SKILL_DIR\}[\w./-]*\.py)"
)
BARE_PYTHON = re.compile(
    r"^(?P<indent>\s*)python3?\s+(?P<script>\$\{CLAUDE_SKILL_DIR\}[\w./-]*\.py)"
)


def venv_for_script(script: str, skill: str) -> str:
    """Infer the uv project a script needs from what it imports."""
    rel = script.replace("${CLAUDE_SKILL_DIR}/", f"skills/{skill}/")
    path = (PROJECT_ROOT / rel).resolve()
    text = path.read_text(errors="replace") if path.is_file() else ""
    if re.search(r"^\s*(?:import|from)\s+fairchem\b", text, re.M):
        return "fairchem"
    if re.search(
        r"^\s*(?:import|from)\s+(?:mace|matgl|torch|nvalchemi)\b", text, re.M
    ) or ("src.utils.mlips" in text or "load_wrapper" in text):
        return "mlip"
    return "cpu"


def rewrite_paths(text: str, skill: str, in_code: bool) -> str:
    """Rewrite repository-relative paths to ${CLAUDE_SKILL_DIR}-relative ones."""

    def skill_path(m: re.Match) -> str:
        name = m.group("name")
        if not (SKILLS / name / "SKILL.md").is_file():
            return m.group(0)
        return f"{SKILL_DIR}/" if name == skill else f"{SKILL_DIR}/../{name}/"

    text = SKILL_PATH.sub(skill_path, text)
    if in_code:
        text = REPO_PATH.sub(lambda m: f"{SKILL_DIR}/../../{m.group('root')}/", text)
    return text


def rewrite_command(
    line: str, skill: str, in_code: bool = True, known: dict[str, str] | None = None
) -> str:
    """Rewrite one line of a code block, or one inline code span of prose.

    ``known`` maps a script to the venv spec its other commands in the same
    SKILL.md already use, so a bare ``python script.py`` line agrees with them.
    """
    launched = LAUNCHED.search(line) if in_code else None
    if launched and skill not in CONDA_SKILLS:
        # Already a launcher command: keep its venv, refresh the extras its
        # script needs (they can change when a script's imports do).
        venv, *extras = launched.group("spec").split("+")
        rel = launched.group("script").replace(
            "${CLAUDE_SKILL_DIR}/", f"skills/{skill}/"
        )
        wanted = sorted(
            set(extras)
            | set(extras_for(str(Path(rel).resolve().relative_to(PROJECT_ROOT))))
        )
        spec = "+".join([venv, *wanted])
        return line[: launched.start("spec")] + spec + line[launched.end("spec") :]
    bare = BARE_PYTHON.match(line) if in_code else None
    if bare and LAUNCHER not in line:
        script = bare.group("script")
        if skill in CONDA_SKILLS:
            prefix = f"conda run --no-capture-output -n {CONDA_SKILLS[skill]} "
        else:
            spec = (known or {}).get(script)
            if spec is None:
                rel = script.replace("${CLAUDE_SKILL_DIR}/", f"skills/{skill}/")
                extras = extras_for(str(Path(rel)))
                spec = "+".join([venv_for_script(script, skill), *extras])
            prefix = f"{LAUNCHER} {spec} "
        return bare.group("indent") + prefix + line[len(bare.group("indent")) :]
    m = UV_RUN.search(line)
    if m:
        venv = m.group("venv")
        extras = re.findall(r"--extra ([\w-]+)", m.group("extras"))
        script = SCRIPT.search(line[m.end() - 1 :])
        if script:
            extras += extras_for(script.group("path"))
        if skill in CONDA_SKILLS:
            prefix = f"conda run --no-capture-output -n {CONDA_SKILLS[skill]} "
        else:
            spec = "+".join([venv, *sorted(set(extras))])
            prefix = f"{LAUNCHER} {spec} "
        line = line[: m.start()] + prefix + line[m.end() :]
    return rewrite_paths(line, skill, in_code=in_code)


def mcp_note(calls: list[tuple[str, str]], servers: dict[str, str]) -> str:
    """Return the note explaining how this skill's MCP tools can be called.

    ``calls`` are the (server, tool) pairs the skill uses, in order of first use.
    """
    server, tool = calls[0]
    lines = [
        NOTE_MARKER,
        "> [!NOTE]",
        f"> Steps written `server.tool` are MCP tool calls: `{server}.{tool}` is the `{tool}`",
        f"> tool of the `{server}` server (`mcp__{server}__{tool}`, or",
        f"> `mcp__plugin_atomistic-skills_{server}__{tool}` when installed as a plugin).",
    ]
    used = list(dict.fromkeys(s for s, _ in calls))
    shell = [s for s in used if servers.get(s, "-") != "-"]
    if shell:
        lines += [
            "> Without a connected server, run the same tools from the shell. Tools named in",
            "> one command share a process, so a model loaded by `load_model` stays loaded:",
            ">",
            "> ```bash",
        ]
        for s in shell:
            tools = list(dict.fromkeys(t for srv, t in calls if srv == s))[:2]
            example = " ".join(f"{t} key=value" for t in tools)
            lines.append(
                f"> {LAUNCHER} {servers[s]} python -m src.mcp_server.cli {s} {example}"
            )
        lines.append("> ```")
    container_only = [s for s in used if servers.get(s, "-") == "-"]
    if container_only:
        lines.append(
            "> "
            + ", ".join(f"`{s}`" for s in container_only)
            + " run only as MCP servers, from the `generative` container image (arm64)"
            " or the matching conda environment in `conda-envs/`."
        )
    return "\n".join(lines)


def set_metadata(text: str, venvs: list[str], conda_env: str | None) -> str:
    """Write metadata.venv (and metadata.conda_env) into the frontmatter."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return text
    lines = [
        ln for ln in m.group(1).split("\n") if not re.match(r"^  (venv|conda_env):", ln)
    ]
    try:
        idx = next(i for i, ln in enumerate(lines) if re.match(r"^  category:", ln))
    except StopIteration:
        return text
    new = [f"  venv: [{', '.join(venvs)}]"]
    if conda_env:
        new.append(f"  conda_env: {conda_env}")
    lines[idx + 1 : idx + 1] = new
    return "---\n" + "\n".join(lines) + "\n---\n" + text[m.end() :]


def migrate(path: Path, servers: dict[str, str]) -> str:
    """Return the migrated text of one SKILL.md."""
    skill = path.parent.name
    original = path.read_text()
    known = {
        m.group("script"): m.group("spec")
        for m in re.finditer(
            re.escape(LAUNCHER)
            + r" (?P<spec>[\w+-]+) python3? (?P<script>\$\{CLAUDE_SKILL_DIR\}[\w./-]*\.py)",
            original,
        )
    }
    out, in_code = [], False
    for line in original.split("\n"):
        if FENCE.match(line):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            ann = VENV_ANNOTATION.match(line)
            if ann:
                # The command now names its environment; keep any prose.
                if ann.group("rest"):
                    out.append(f"{ann.group('indent')}# {ann.group('rest')}")
                continue
            line = rewrite_command(line, skill, known=known)
        else:
            # Inline code spans in prose hold commands and paths too.
            # Inline code holds commands and skill paths too; a bare reference
            # to src/... there is documentation and stays as written.
            line = re.sub(
                r"`[^`]+`",
                lambda m: rewrite_command(m.group(0), skill, in_code=False),
                line,
            )
        line = MCP_NAME.sub(lambda m: f"{m.group('server')}.{m.group('tool')}", line)
        out.append(line)
    text = "\n".join(out)

    # The note goes in once, from the original mcp_* names: after renaming,
    # `matgl.load_model(...)` in a Python snippet is indistinguishable from the
    # MCP tool of the same name, so later runs leave an existing note alone.
    calls = list(
        dict.fromkeys(
            (m.group("server"), m.group("tool")) for m in MCP_NAME.finditer(original)
        )
    )
    if calls and NOTE_MARKER not in text:
        title = re.search(r"^# .+\n", text, re.M)
        if title:
            text = (
                text[: title.end()]
                + "\n"
                + mcp_note(calls, servers)
                + "\n"
                + text[title.end() :]
            )

    venvs = sorted(set(re.findall(re.escape(LAUNCHER) + r" (\w+)", text)))
    return set_metadata(text, venvs, CONDA_SKILLS.get(skill))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report only; exit 1 if anything would change",
    )
    args = parser.parse_args()

    servers = load_servers()
    changed = []
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        new = migrate(path, servers)
        if new != path.read_text():
            changed.append(path.parent.name)
            if not args.check:
                path.write_text(new)
    verb = "would change" if args.check else "changed"
    print(f"{verb} {len(changed)} SKILL.md files")
    for name in changed:
        print(f"  {name}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
