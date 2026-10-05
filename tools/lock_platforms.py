#!/usr/bin/env python3
"""Work out, from each uv.lock, which machines can install which environment.

`uv lock` proves that versions resolve, and `required-environments` proves that
every package has *some* wheel for each architecture. Neither proves that the
wheel installs on a given machine: a manylinux_2_34 wheel needs glibc 2.34, and
a package with only a source distribution needs a compiler and sometimes system
headers. This script answers that question per (venv, extra, architecture) and
records it in ``venv/platforms.tsv``, which ``venv/run`` reads to decide whether
a host can use uv directly or should fall back to a container.

    min_glibc  - the oldest glibc on which every wheel in the set installs
    builds     - packages with no usable wheel that uv compiles from source

Usage:
    venv/run cpu python tools/lock_platforms.py           # rewrite venv/platforms.tsv
    venv/run cpu python tools/lock_platforms.py --check   # fail if it is stale
    venv/run cpu python tools/lock_platforms.py --report  # also list builds per set

Requirements:
    - the ``packaging`` library (present in every uv project) and ``uv`` on PATH
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / "venv"
TABLE = VENV_DIR / "platforms.tsv"
ARCHES = ("x86_64", "aarch64")
LEGACY_MANYLINUX = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}


def projects() -> list[str]:
    """Return the uv project names under venv/."""
    return sorted(p.parent.name for p in VENV_DIR.glob("*/pyproject.toml"))


def python_minor(project: str) -> int:
    """Return the Python 3 minor version a project runs, from requires-python."""
    data = tomllib.loads((VENV_DIR / project / "pyproject.toml").read_text())
    spec = data["project"]["requires-python"]
    return int(re.search(r"3\.(\d+)", spec).group(1))


def arches_of(project: str) -> set[str]:
    """Return the architectures a project resolves for (tool.uv.environments)."""
    data = tomllib.loads((VENV_DIR / project / "pyproject.toml").read_text())
    envs = data.get("tool", {}).get("uv", {}).get("environments", [])
    return {a for a in ARCHES if any(a in e for e in envs)} or set(ARCHES)


def extras_of(project: str) -> list[str]:
    """Return the optional-dependency groups a project declares."""
    data = tomllib.loads((VENV_DIR / project / "pyproject.toml").read_text())
    return sorted(data.get("project", {}).get("optional-dependencies", {}))


def needed(project: str, extra: str | None, arch: str) -> dict[str, str]:
    """Return {name: version} of every registry package the set installs on arch."""
    cmd = [
        "uv",
        "export",
        "--frozen",
        "--no-hashes",
        "--no-header",
        "--no-emit-project",
        "--no-dev",
        "--project",
        str(VENV_DIR / project),
    ]
    if extra:
        cmd += ["--extra", extra]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    env = {
        "sys_platform": "linux",
        "platform_system": "Linux",
        "os_name": "posix",
        "platform_machine": arch,
        "python_version": f"3.{python_minor(project)}",
        "python_full_version": f"3.{python_minor(project)}.0",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "extra": extra or "",
    }
    pins: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e", ".")) or " @ " in line:
            continue
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate(env):
            continue
        version = next((s.version for s in req.specifier if s.operator == "=="), None)
        pins[canonical(req.name)] = version
    return pins


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_glibc(filename: str, arch: str, minor: int = 12) -> tuple[int, int] | None:
    """Return the glibc a wheel needs on arch (CPython 3.minor), or None if it cannot."""
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return None
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    if not any(
        p in ("py3", f"cp3{minor}")
        or (re.fullmatch(r"cp3(\d+)", p) and abi == "abi3" and int(p[3:]) <= minor)
        for p in py.split(".")
    ):
        return None
    best = None
    for tag in plat.split("."):
        if tag == "any":
            return (0, 0)
        # A plain linux_<arch> wheel (PyG's extensions) states no glibc floor;
        # uv installs it on any Linux of that architecture.
        if tag == f"linux_{arch}":
            return (0, 0)
        m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", tag)
        if m and m.group(3) == arch:
            need = (int(m.group(1)), int(m.group(2)))
        else:
            m = re.fullmatch(r"(manylinux1|manylinux2010|manylinux2014)_(\w+)", tag)
            need = LEGACY_MANYLINUX[m.group(1)] if m and m.group(2) == arch else None
        if need is not None and (best is None or need < best):
            best = need
    return best


def analyse(project: str, extra: str | None, arch: str) -> tuple[str, list[str]]:
    """Return (min_glibc or 'unavailable:<pkgs>', [packages built from source])."""
    lock = tomllib.loads((VENV_DIR / project / "uv.lock").read_text())
    by_name: dict[str, list[dict]] = {}
    for pkg in lock["package"]:
        by_name.setdefault(canonical(pkg["name"]), []).append(pkg)

    minor = python_minor(project)
    floor, builds, missing = (0, 0), [], []
    for name, version in sorted(needed(project, extra, arch).items()):
        candidates = [
            p for p in by_name.get(name, []) if version in (None, p.get("version"))
        ]
        if not candidates:
            continue
        pkg = candidates[0]
        source = pkg.get("source", {})
        if any(
            k in source for k in ("git", "path", "directory", "editable", "virtual")
        ):
            continue
        needs = [
            g
            for w in pkg.get("wheels", [])
            if (g := wheel_glibc(w["url"].rsplit("/", 1)[-1], arch, minor)) is not None
        ]
        if needs:
            floor = max(floor, min(needs))
        elif "sdist" in pkg:
            builds.append(f"{name}=={pkg.get('version')}")
        else:
            missing.append(f"{name}=={pkg.get('version')}")
    if missing:
        return "unavailable:" + ",".join(missing), builds
    return f"{floor[0]}.{floor[1]}", builds


def render() -> tuple[str, list[str]]:
    """Return the table text and a human-readable report."""
    rows, report = [], []
    for project in projects():
        for extra in [None, *extras_of(project)]:
            base_builds: set[str] = set()
            for arch in ARCHES:
                if arch not in arches_of(project):
                    rows.append(
                        "\t".join(
                            [
                                project,
                                extra or "-",
                                arch,
                                f"unavailable:not built for {arch}",
                            ]
                        )
                    )
                    continue
                glibc, builds = analyse(project, extra, arch)
                if extra is None:
                    base_builds |= set(builds)
                rows.append("\t".join([project, extra or "-", arch, glibc]))
                extra_builds = sorted(set(builds) - base_builds) if extra else builds
                report.append(
                    f"{project:9} {extra or '-':8} {arch:8} glibc>={glibc:12} "
                    f"builds: {', '.join(extra_builds) or '-'}"
                )
    header = [
        "# Generated by tools/lock_platforms.py from venv/*/uv.lock -- do not edit.",
        "# The oldest glibc on which every wheel of a set installs. venv/run uses",
        "# it to decide whether a host can run the environment with uv or should",
        "# use a container. 'unavailable' means a package has no wheel at all there.",
        "# venv\textra\tarch\tmin_glibc",
    ]
    return "\n".join(header + rows) + "\n", report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="fail if venv/platforms.tsv is stale"
    )
    mode.add_argument(
        "--report", action="store_true", help="print floors and source builds"
    )
    args = parser.parse_args()

    text, report = render()
    if args.report:
        print("\n".join(report))
        return 0
    if args.check:
        if not TABLE.exists() or TABLE.read_text() != text:
            print(
                "venv/platforms.tsv is stale; run: venv/run cpu python tools/lock_platforms.py"
            )
            return 1
        print("venv/platforms.tsv is up to date")
        return 0
    TABLE.write_text(text)
    print(f"Wrote {TABLE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
