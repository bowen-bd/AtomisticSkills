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


def classify(output: str, arch: str) -> str | None:
    """Classify a command failure output against known platform limits.

    Returns a skip reason string if the failure is an expected platform limitation
    on the given architecture, or None otherwise.
    """
    normalized_arch = arch.lower()
    if normalized_arch not in ("aarch64", "arm64"):
        return None

    # PyMOL is x86_64 only
    if re.search(r"No module named ['\"]pymol['\"]", output):
        return "the pymol extra is x86_64 only"

    # SCINE utilities/readuct has no aarch64 wheels
    if re.search(r"No module named ['\"]scine_(?:utilities|readuct)['\"]", output):
        return "scine_utilities / scine_readuct has no aarch64 wheels"

    # AutoDock Vina on aarch64 requires the docking extra and Boost
    if re.search(r"No module named ['\"]vina['\"]", output):
        return "vina on aarch64 requires docking extra and Boost"
    if "vina" in output.lower() and (
        "boost" in output.lower() or "libboost" in output.lower()
    ):
        return "vina on aarch64 requires docking extra and Boost"

    return None


def format_summary(
    passed: int = 0,
    failed: int = 0,
    skipped: int = 0,
    *,
    ok: int | None = None,
    fail: int | None = None,
    skip: int | None = None,
) -> str:
    """Format the end-of-run summary line."""
    p = ok if ok is not None else passed
    f = fail if fail is not None else failed
    s = skip if skip is not None else skipped
    return f"{p} passed, {f} failed, {s} skipped"


def check(
    item: tuple[str, str, Path], timeout: float, arch: str | None = None
) -> tuple[str, str, Path, bool, str, float, str | None]:
    skill, spec, script = item
    t0 = time.monotonic()
    if arch is None:
        import platform

        arch = platform.machine()
    if not script.is_file():
        return skill, spec, script, False, "script does not exist", 0.0, None
    try:
        proc = subprocess.run(
            [str(LAUNCHER), spec, "python", str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        ok = proc.returncode == 0
        raw_output = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
        tail = raw_output.splitlines()
        detail = "" if ok else (tail[-1] if tail else f"exit {proc.returncode}")
        skip_reason = classify(raw_output, arch) if not ok else None
    except subprocess.TimeoutExpired:
        ok, detail, skip_reason = False, f"timed out after {timeout:.0f}s", None
    return skill, spec, script, ok, detail, time.monotonic() - t0, skip_reason


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

    import platform

    current_arch = platform.machine()
    print(f"checking {len(items)} script/environment pairs on {current_arch}")
    ok_count = 0
    failures = []
    skips = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for skill, spec, script, ok, detail, secs, skip_reason in pool.map(
            lambda it: check(it, args.timeout, current_arch), items
        ):
            rel = (
                script.relative_to(PROJECT_ROOT)
                if script.is_relative_to(PROJECT_ROOT)
                else script
            )
            if ok:
                ok_count += 1
                print(f"ok    {spec:16} {rel} ({secs:.1f}s)")
            elif skip_reason:
                skips.append((skill, spec, rel, skip_reason))
                print(f"skip  {spec:16} {rel} ({secs:.1f}s) -- {skip_reason}")
            else:
                failures.append((skill, spec, rel, detail))
                print(f"FAIL  {spec:16} {rel} ({secs:.1f}s) -- {detail}")

    print(f"\n{format_summary(ok=ok_count, fail=len(failures), skip=len(skips))}")
    if skips:
        print("\nSkipped (known platform limits):")
        for skill, spec, rel, reason in skips:
            print(f"  {skill}: {spec} {rel} -- {reason}")
    if failures:
        print("\nFailures:")
        for skill, spec, rel, detail in failures:
            print(f"  {skill}: venv/run {spec} python {rel} --help -> {detail}")
    return min(len(failures), 100)


if __name__ == "__main__":
    sys.exit(main())
