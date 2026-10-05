#!/usr/bin/env python3
"""Run every skill script a SKILL.md names, with --help, in the environment it names.

`--help` makes each script parse its arguments and exit, which is enough to
import everything it imports at module level. A failure therefore means the
skill's first command breaks on this machine: a dependency missing from the uv
project, an import error, or a script path that does not exist.

Usage:
    python tools/check_skill_commands.py                 # every skill
    python tools/check_skill_commands.py --venv mlip     # one environment
    python tools/check_skill_commands.py --skill mat-phonon --jobs 1

Exit status is the number of failing commands (capped at 100).

Requirements:
    - Python 3.10+, standard library only; the environments are created on demand
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "venv" / "run"
COMMAND = re.compile(
    r"\$\{CLAUDE_SKILL_DIR\}/\.\./\.\./venv/run (?P<spec>[\w+-]+) python3? "
    r"(?P<script>\$\{CLAUDE_SKILL_DIR\}[\w./-]*\.py)"
)


def collect(skills: list[str] | None, venv: str | None) -> list[tuple[str, str, Path]]:
    """Return unique (skill, spec, script path) triples."""
    seen, out = set(), []
    for skill_md in sorted((PROJECT_ROOT / "skills").glob("*/SKILL.md")):
        skill = skill_md.parent.name
        if skills and skill not in skills:
            continue
        for m in COMMAND.finditer(skill_md.read_text()):
            spec = m.group("spec")
            if venv and spec.split("+")[0] != venv:
                continue
            script = (
                skill_md.parent / m.group("script").removeprefix("${CLAUDE_SKILL_DIR}/")
            ).resolve()
            key = (spec, script)
            if key not in seen:
                seen.add(key)
                out.append((skill, spec, script))
    return out


def check(
    item: tuple[str, str, Path], timeout: float
) -> tuple[str, str, Path, bool, str, float]:
    skill, spec, script = item
    t0 = time.monotonic()
    if not script.is_file():
        return skill, spec, script, False, "script does not exist", 0.0
    try:
        proc = subprocess.run(
            [str(LAUNCHER), spec, "python", str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        ok = proc.returncode == 0
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        detail = "" if ok else (tail[-1] if tail else f"exit {proc.returncode}")
    except subprocess.TimeoutExpired:
        ok, detail = False, f"timed out after {timeout:.0f}s"
    return skill, spec, script, ok, detail, time.monotonic() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--venv", help="only commands in this environment")
    parser.add_argument(
        "--skill", action="append", help="only these skills (repeatable)"
    )
    parser.add_argument("--jobs", type=int, default=4, help="parallel commands")
    parser.add_argument(
        "--timeout", type=float, default=300, help="seconds per command"
    )
    args = parser.parse_args()

    items = collect(args.skill, args.venv)
    # Create each environment once up front, so parallel first uses do not race.
    for venv in sorted({spec for _, spec, _ in items}):
        subprocess.run(
            [str(LAUNCHER), venv, "python", "-c", "pass"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(f"checking {len(items)} script/environment pairs")
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for skill, spec, script, ok, detail, secs in pool.map(
            lambda it: check(it, args.timeout), items
        ):
            rel = (
                script.relative_to(PROJECT_ROOT)
                if script.is_relative_to(PROJECT_ROOT)
                else script
            )
            print(
                f"{'ok  ' if ok else 'FAIL'} {spec:16} {rel} ({secs:.1f}s){'' if ok else ' -- ' + detail}"
            )
            if not ok:
                failures.append((skill, spec, rel, detail))
    print(f"\n{len(items) - len(failures)} passed, {len(failures)} failed")
    for skill, spec, rel, detail in failures:
        print(f"  {skill}: venv/run {spec} python {rel} --help -> {detail}")
    return min(len(failures), 100)


if __name__ == "__main__":
    sys.exit(main())
