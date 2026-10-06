#!/usr/bin/env python3
"""Measure the glibc and libstdc++ a uv project's binary wheels really need.

A wheel's platform tag is a claim: PyG's extensions carry a plain
``linux_x86_64`` tag that states no floor, and DGL's ``manylinux1`` tag is
wrong. This downloads each such wheel of a project's lock (into a cache) and
reports the newest ``GLIBC_`` and ``GLIBCXX_`` symbol versions its shared
libraries import, for ``MEASURED_FLOORS`` in tools/lock_platforms.py.

Usage:
    venv/run cpu python tools/scan_wheel_floors.py scd adit
    venv/run cpu python tools/scan_wheel_floors.py --all msms   # every binary wheel

Requirements:
    - network access to the wheels' indexes
"""

from __future__ import annotations

import argparse
import re
import tomllib
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "atomisticskills" / "wheel-scan"
GLIBC = re.compile(rb"GLIBC_(\d+)\.(\d+)")
GLIBCXX = re.compile(rb"GLIBCXX_3\.4\.(\d+)")
SUSPECT = re.compile(r"-(linux|manylinux1|manylinux2010)_(x86_64|aarch64)\.whl$")


def needs(path: Path) -> tuple[tuple[int, int], int]:
    """Newest GLIBC_ version and GLIBCXX_3.4 minor the wheel's libraries import."""
    glibc, glibcxx = (0, 0), 0
    with zipfile.ZipFile(path) as wheel:
        for name in wheel.namelist():
            if ".so" not in name:
                continue
            data = wheel.read(name)
            for m in GLIBC.finditer(data):
                glibc = max(glibc, (int(m[1]), int(m[2])))
            for m in GLIBCXX.finditer(data):
                glibcxx = max(glibcxx, int(m[1]))
    return glibc, glibcxx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("projects", nargs="+", help="uv projects under venv/")
    parser.add_argument(
        "--all", action="store_true", help="every binary wheel, not only suspect tags"
    )
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    seen = set()
    for project in args.projects:
        lock = tomllib.loads((PROJECT_ROOT / "venv" / project / "uv.lock").read_text())
        for package in lock["package"]:
            for wheel in package.get("wheels", []):
                url = wheel["url"]
                name = url.rsplit("/", 1)[-1].replace("%2B", "+")
                binary = not name.endswith("-none-any.whl")
                if name in seen or not binary or not (args.all or SUSPECT.search(name)):
                    continue
                seen.add(name)
                path = CACHE / name
                if not path.exists():
                    # PyTorch's CDN refuses urllib's default user agent.
                    request = urllib.request.Request(
                        url.replace("+", "%2B"), headers={"User-Agent": "curl/8"}
                    )
                    with urllib.request.urlopen(request) as response:
                        path.write_bytes(response.read())
                glibc, glibcxx = needs(path)
                print(
                    f"{project:10} {name:80} GLIBC_{glibc[0]}.{glibc[1]}  GLIBCXX_3.4.{glibcxx}"
                )


if __name__ == "__main__":
    main()
