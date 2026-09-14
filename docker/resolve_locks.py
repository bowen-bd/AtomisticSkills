"""Resolve pip locks for a platform the environments were never built on.

``export_locks.py`` freezes an environment that exists on this machine. That
only works for the architecture the machine has, so it cannot produce the
linux-64 locks the amd64 images need -- the validated environments live on an
aarch64 workstation.

This script takes the other route: resolve each environment's *declared* pip
requirements for the target platform and write the closed result. That is sound
for amd64 specifically, because the reason the aarch64 environments cannot be
resolved does not apply there. ``fairchem-core`` declares ``torch~=2.8.0`` and
the aarch64 build had to override it, since torch 2.8 ships no sm_121 wheel; on
x86_64 that constraint is satisfiable and the solve is clean. The same holds for
the PyTorch Geometric extensions, which publish x86_64 CUDA wheels and so need
no source build.

Only environments with a usable declarative spec can go through here. The
generative environments (adit, diffcsp, mattergen) were assembled by hand --
``adit-agent`` declares nothing but python, pip and uv -- so their locks must
still be derived from a built environment.

Usage:
    # Env: base-agent
    python docker/resolve_locks.py --env mace-agent
    python docker/resolve_locks.py                  # every supported env

Requirements:
    - Conda environment: base-agent
    - ``uv`` on PATH
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA_ENVS = PROJECT_ROOT / "conda-envs"

# Environments whose core_env.yaml lists the full pip layer. The generative
# environments are deliberately absent; see the module docstring.
SUPPORTED = {
    "mace-agent": "3.12",
    "matgl-agent": "3.12",
    "fairchem-agent": "3.12",
}

TARGET_PLATFORM = "x86_64-unknown-linux-gnu"
SUBDIR = "linux-64"

# A local editable checkout cannot be installed inside an image; point at the
# published distribution instead. Kept in step with export_locks.py.
EDITABLE_ORIGINS = {
    "nvalchemi": "nvalchemi-toolkit==0.1.0",
    "nvalchemi-toolkit": "nvalchemi-toolkit==0.1.0",
}


def pip_requirements(env: str) -> list[str]:
    """Return the ``pip:`` entries of an environment spec, origins substituted.

    The spec is read line-wise rather than with a YAML parser: these files carry
    comments that explain why a pin exists, and the block is a flat list, so a
    parser would buy nothing and add a dependency.
    """
    spec = CONDA_ENVS / env / "core_env.yaml"
    if not spec.is_file():
        sys.exit(f"no spec for {env}: {spec}")

    out: list[str] = []
    in_pip = False
    for raw in spec.read_text().splitlines():
        stripped = raw.strip()
        if re.match(r"^-\s*pip:\s*$", stripped):
            in_pip = True
            continue
        if in_pip:
            # The pip block ends at the first line that is not one of its items.
            if stripped and not stripped.startswith("-"):
                break
            if not stripped or stripped.startswith("#"):
                continue
            item = stripped.lstrip("-").strip().strip('"').strip("'")
            if not item:
                continue
            if item.startswith("-e "):
                name = Path(item[3:].strip()).name.split("[")[0]
                origin = EDITABLE_ORIGINS.get(name)
                if origin is None:
                    sys.exit(
                        f"{env}: editable requirement {item!r} has no published "
                        "origin; add one to EDITABLE_ORIGINS"
                    )
                out.append(origin)
                continue
            out.append(item)
    if not out:
        sys.exit(f"{env}: no pip requirements found in {spec}")
    return out


def resolve(env: str, python: str) -> list[str]:
    """Return the closed, pinned requirement set for the target platform."""
    if shutil.which("uv") is None:
        sys.exit("uv is not on PATH; it is required to resolve locks")

    requirements = pip_requirements(env)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "requirements.in"
        src.write_text("\n".join(requirements) + "\n")
        proc = subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "--quiet",
                "--no-header",
                "--python-platform",
                TARGET_PLATFORM,
                "--python-version",
                python,
                str(src),
                "-o",
                str(Path(tmp) / "out.txt"),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.exit(f"{env}: resolution failed for {SUBDIR}\n{proc.stderr}")
        resolved = (Path(tmp) / "out.txt").read_text().splitlines()

    # Keep only the pins; uv writes provenance comments between them.
    return [line for line in resolved if line and not line.lstrip().startswith("#")]


def write_lock(env: str, pins: list[str]) -> Path:
    """Write the pip lock in the same shape export_locks.py produces."""
    lock_dir = CONDA_ENVS / env / "lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"pip-{SUBDIR}.txt"
    header = [
        f"# {env} pip layer, resolved for {SUBDIR}.",
        "# Generated by docker/resolve_locks.py -- do not edit by hand.",
        "# Resolved from the declared spec rather than frozen from a built",
        "# environment: the validated environments are aarch64, and the",
        "# constraints that cannot be satisfied there are satisfiable here.",
        "# Install with --no-deps; this lock is already closed.",
    ]
    path.write_text("\n".join(header + pins) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--env",
        action="append",
        choices=sorted(SUPPORTED),
        help="environment to resolve (repeatable; default: all supported)",
    )
    args = parser.parse_args()

    for env in args.env or sorted(SUPPORTED):
        pins = resolve(env, SUPPORTED[env])
        path = write_lock(env, pins)
        print(f"{env}: {len(pins)} packages -> {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
