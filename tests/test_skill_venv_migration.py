"""Tests that every skill names a runnable environment.

Skills used to annotate a block with ``# Env: <conda-env>`` and then run a bare
``python script.py``, which only works for someone who had already activated
that environment. A plugin user cannot: the containers expose MCP servers, not
conda environments, so 112 of 131 skills failed at their first script step.
These tests keep the replacement honest -- every annotation must name a uv
project that exists, and every invocation must carry the project with it.

Requirements:
    - Conda environment: base-agent
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS = PROJECT_ROOT / "skills"
VENV = PROJECT_ROOT / "venv"

# The generative stack has no uv project: mattergen hard-pins +cu118 wheels
# that never existed for aarch64, and ADiT/DiffCSP++ are not published
# packages. Those blocks stay on the container path.
CONTAINER_ONLY_ENVS = {"adit-agent", "diffcsp-agent", "mattergen-agent"}

VENV_LINE = re.compile(r"^\s*#\s*Venv:\s*venv/(?P<project>[\w-]+)")
ENV_LINE = re.compile(r"^\s*#\s*Env:\s*(?P<env>[\w-]+)")
UV_RUN = re.compile(r"uv run --project venv/(?P<project>[\w-]+)")


def projects() -> set[str]:
    return {p.name for p in VENV.iterdir() if (p / "pyproject.toml").is_file()}


def skill_files() -> list[Path]:
    return sorted(SKILLS.glob("*/SKILL.md"))


class TestNoStaleCondaAnnotations:
    def test_only_the_container_stack_still_names_a_conda_env(self):
        """A leftover `# Env:` sends the reader to an environment that is gone."""
        stale = []
        for sk in skill_files():
            for n, line in enumerate(sk.read_text().split("\n"), 1):
                m = ENV_LINE.match(line)
                if m and m.group("env") not in CONTAINER_ONLY_ENVS:
                    stale.append(f"{sk.parent.name}:{n} {m.group('env')}")
        assert not stale, "run python tools/migrate_skill_envs.py\n" + "\n".join(stale)

    def test_no_skill_tells_the_reader_to_activate_conda(self):
        """uv run needs no activation, and these environments are gone.

        The generative stack is exempt: it still runs from a container, so a
        conda environment is still the right instruction there.
        """
        banned = ("conda activate", "mamba activate", "conda run -n")
        offenders = []
        for sk in skill_files():
            for n, line in enumerate(sk.read_text().split("\n"), 1):
                if not any(b in line for b in banned):
                    continue
                if any(env in line for env in CONTAINER_ONLY_ENVS):
                    continue
                offenders.append(f"{sk.parent.name}:{n} {line.strip()[:70]}")
        assert not offenders, "run tools/migrate_skill_envs.py\n" + "\n".join(offenders)


class TestAnnotationsResolve:
    def test_every_venv_annotation_names_a_real_project(self):
        known = projects()
        bad = []
        for sk in skill_files():
            for n, line in enumerate(sk.read_text().split("\n"), 1):
                m = VENV_LINE.match(line)
                if m and m.group("project") not in known:
                    bad.append(f"{sk.parent.name}:{n} venv/{m.group('project')}")
        assert not bad, f"unknown projects (have {sorted(known)}):\n" + "\n".join(bad)

    def test_every_uv_run_names_a_real_project(self):
        known = projects()
        bad = []
        for sk in skill_files():
            for n, line in enumerate(sk.read_text().split("\n"), 1):
                for m in UV_RUN.finditer(line):
                    if m.group("project") not in known:
                        bad.append(f"{sk.parent.name}:{n} venv/{m.group('project')}")
        assert not bad, "\n".join(bad)

    def test_each_project_has_a_lock(self):
        """An unlocked project cannot be reproduced by `uv run`."""
        for p in sorted(projects()):
            assert (VENV / p / "uv.lock").is_file(), f"venv/{p} has no uv.lock"


class TestInvocationsCarryTheirEnvironment:
    def test_annotated_shell_blocks_do_not_call_bare_python(self):
        """`python script.py` under a `# Venv:` line silently uses system python.

        Only shell blocks are checked: an inline Python snippet legitimately
        contains `import`/`from` lines with no command to prefix.
        """
        offenders = []
        for sk in skill_files():
            lines = sk.read_text().split("\n")
            fences = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("```")]
            shell = {
                (s, e)
                for s, e in zip(fences[::2], fences[1::2])
                if lines[s].lstrip().startswith("```bash")
                or lines[s].lstrip().startswith("```sh")
            }
            for start, end in shell:
                if not any(VENV_LINE.match(lines[k]) for k in range(start + 1, end)):
                    continue
                for k in range(start + 1, end):
                    if re.match(r"^\s*python\s+skills/", lines[k]):
                        offenders.append(
                            f"{sk.parent.name}:{k + 1} {lines[k].strip()[:60]}"
                        )
        assert not offenders, "bare python under a Venv annotation:\n" + "\n".join(
            offenders
        )
