#!/usr/bin/env python3
"""Freeze the working conda environments into per-architecture lockfiles.

The MLIP and generative environments cannot be reproduced from a declarative
spec: several of them deliberately violate their own declared constraints
because the constrained version has no build for this platform. For example
``fairchem-core 2.19.0`` declares ``torch~=2.8.0`` while the environment runs
torch 2.10.0, since torch 2.8 ships no sm_121 / aarch64 / CUDA 13 wheel. Any
resolver asked to solve that spec fails, which is why the container images are
built from exact locks instead.

For every environment this writes two files under ``conda-envs/<env>/lock/``:

``conda-<subdir>.txt``
    ``conda list --explicit --md5`` output: exact package URLs with checksums,
    consumable by ``conda create --file`` or ``micromamba create --file``.
``pip-<subdir>.txt``
    Pinned versions of the pip-installed (``pypi`` channel) packages only, so
    the pip layer never re-installs what the conda layer already provided.
    Install these with ``--no-deps``; the lock is already closed.

Local editable installs are rewritten to their public git origin so the locks
are usable off this machine. Any remaining local path is reported and dropped
rather than silently baked into an image that could never build.

Usage:
    # Env: base-agent
    python docker/export_locks.py                  # every environment
    python docker/export_locks.py --env mace fairchem

Requirements:
    - conda/mamba on PATH; the target environments must already exist
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_SPEC = PROJECT_ROOT / "docker" / "images.json"

# Local editable checkouts rewritten to something installable off this machine.
# Keyed by the distribution name pip reports for the editable install.
#
# nvalchemi-toolkit is on PyPI, so pin it there rather than pointing at the git
# repository. A `nvalchemi @ git+...` requirement fails: pip clones, builds the
# metadata, finds the project is actually named nvalchemi-toolkit and refuses
# with "Generating metadata for package nvalchemi produced metadata for project
# name nvalchemi-toolkit". A plain PyPI pin avoids the clone and the mismatch.
EDITABLE_ORIGINS = {
    "nvalchemi": "nvalchemi-toolkit==0.1.0",
    "nvalchemi-toolkit": "nvalchemi-toolkit==0.1.0",
}

# Compiled PyTorch Geometric extensions are deliberately kept out of the pip
# locks. They have no aarch64 + CUDA 13 wheels, and their build backends import
# torch to generate metadata, so installing them in the same `pip install -r`
# pass as torch fails with "Failed to build 'torch-scatter' when getting
# requirements to build wheel". The Dockerfile builds them in a later, dedicated
# step with --no-build-isolation, once torch is importable.
SOURCE_BUILT = {"torch-scatter", "torch-cluster", "torch-sparse"}

LOCAL_PATH_RE = re.compile(r"@\s*file://|^-e\s|^\s*-e\s")


def run(cmd: list[str]) -> str:
    """Run a command and return stdout, raising with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr}")
    return proc.stdout


def conda_subdir(env: str) -> str:
    """Return the conda platform subdir the environment was solved for."""
    info = json.loads(run(["conda", "info", "--json"]))
    return info["platform"]


def pypi_packages(env: str) -> list[str]:
    """Return ``name==version`` for packages conda attributes to the pypi channel."""
    rows = json.loads(run(["conda", "list", "-n", env, "--json"]))
    out = []
    for row in rows:
        if row.get("channel") != "pypi":
            continue
        out.append(f"{row['name']}=={row['version']}")
    return sorted(out, key=str.lower)


def rewrite_editables(pins: list[str], env: str) -> tuple[list[str], list[str]]:
    """Map editable/local installs to public origins; report what was dropped."""
    kept, dropped = [], []
    for pin in pins:
        name = re.split(r"[=@\[]", pin, maxsplit=1)[0].strip().lower()
        if name in SOURCE_BUILT:
            # Compiled separately by the Dockerfile; see SOURCE_BUILT.
            continue
        if LOCAL_PATH_RE.search(pin):
            if name in EDITABLE_ORIGINS:
                kept.append(EDITABLE_ORIGINS[name])
            else:
                dropped.append(pin)
            continue
        if name in EDITABLE_ORIGINS and "git+" not in pin:
            # Installed from a local checkout but reported as a plain version
            # pin; point at the public origin so the lock builds anywhere.
            kept.append(EDITABLE_ORIGINS[name])
            continue
        kept.append(pin)
    return kept, dropped


def export(env: str, subdir: str) -> dict:
    """Write both lockfiles for one environment and return a summary."""
    lock_dir = PROJECT_ROOT / "conda-envs" / env / "lock"
    lock_dir.mkdir(parents=True, exist_ok=True)

    explicit = run(["conda", "list", "-n", env, "--explicit", "--md5"])
    conda_lock = lock_dir / f"conda-{subdir}.txt"
    conda_lock.write_text(explicit)

    pins, dropped = rewrite_editables(pypi_packages(env), env)
    header = (
        f"# {env} pip layer, locked on {subdir}.\n"
        "# Generated by docker/export_locks.py -- do not edit by hand.\n"
        "# Install with --no-deps: this lock is already closed, and several\n"
        "# pins intentionally violate upstream constraints that have no build\n"
        "# for this platform.\n"
    )
    pip_lock = lock_dir / f"pip-{subdir}.txt"
    pip_lock.write_text(header + "\n".join(pins) + "\n")

    return {
        "env": env,
        "conda": len([ln for ln in explicit.splitlines() if ln.startswith("http")]),
        "pip": len(pins),
        "dropped": dropped,
    }


def main() -> int:
    spec = json.loads(IMAGES_SPEC.read_text())
    all_envs = [e for image in spec["images"] for e in image["envs"]]

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--env",
        nargs="+",
        default=all_envs,
        help="Environment name(s) to freeze (default: every env in images.json)",
    )
    args = parser.parse_args()

    subdir = conda_subdir(args.env[0])
    print(f"Platform: {subdir}\n")

    failures = []
    for env in args.env:
        try:
            summary = export(env, subdir)
        except RuntimeError as exc:
            failures.append(env)
            print(f"  {env:20s} FAILED: {exc}".rstrip())
            continue
        line = f"  {summary['env']:20s} conda={summary['conda']:<4d} pip={summary['pip']:<4d}"
        if summary["dropped"]:
            line += f"  DROPPED LOCAL: {', '.join(summary['dropped'])}"
        print(line)

    if failures:
        print(f"\n{len(failures)} environment(s) failed: {', '.join(failures)}")
        return 1
    print(f"\nWrote locks for {len(args.env)} environment(s) to conda-envs/*/lock/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
