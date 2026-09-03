#!/usr/bin/env python3
"""Keep every published manifest on the version recorded in ``VERSION``.

Four files carry the project version, and nothing previously kept them in
agreement -- ``server.json`` sat at 1.0.0 while the repository was tagged
v1.3.4. ``VERSION`` at the repository root is now the single source, and this
tool projects it into:

  * ``.claude-plugin/plugin.json``      -> ``version``
  * ``.claude-plugin/marketplace.json`` -> every ``plugins[].version``
  * ``server.json``                     -> ``version`` and each package's
                                           image tag

Run without arguments to write the files; run with ``--check`` in CI to fail
when any of them has drifted. The release checklist in
``.agents/rules/release-standards.md`` bumps ``VERSION`` and runs this before
the tag is created, so the tag and the manifests cannot disagree.

Usage:
    # Env: base-agent
    python tools/sync_version.py            # rewrite manifests from VERSION
    python tools/sync_version.py --check    # verify only, non-zero on drift
    python tools/sync_version.py --set 1.4.0

Requirements:
    - Conda environment: base-agent (standard library only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "VERSION"
PLUGIN = PROJECT_ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE = PROJECT_ROOT / ".claude-plugin" / "marketplace.json"
SERVER = PROJECT_ROOT / "server.json"

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def read_version() -> str:
    """Return the version recorded in VERSION, validating its shape."""
    if not VERSION_FILE.exists():
        sys.exit(f"missing {VERSION_FILE.relative_to(PROJECT_ROOT)}")
    version = VERSION_FILE.read_text().strip()
    if not SEMVER.match(version):
        sys.exit(f"VERSION must be MAJOR.MINOR.PATCH, got {version!r}")
    return version


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def retag(identifier: str, version: str) -> str:
    """Replace the tag on an OCI reference, leaving any digest form alone."""
    if "@" in identifier:
        return identifier
    repo = (
        identifier.rsplit(":", 1)[0]
        if ":" in identifier.rsplit("/", 1)[-1]
        else identifier
    )
    return f"{repo}:{version}"


def plan(version: str) -> list[tuple[Path, dict, list[str]]]:
    """Return per-file (path, updated document, list of drift descriptions)."""
    out: list[tuple[Path, dict, list[str]]] = []

    plugin = load(PLUGIN)
    drift = []
    if plugin.get("version") != version:
        drift.append(f"plugin.json version {plugin.get('version')!r} != {version!r}")
        plugin["version"] = version
    out.append((PLUGIN, plugin, drift))

    market = load(MARKETPLACE)
    drift = []
    for entry in market.get("plugins", []):
        if entry.get("version") != version:
            drift.append(
                f"marketplace.json plugin {entry.get('name')!r} version "
                f"{entry.get('version')!r} != {version!r}"
            )
            entry["version"] = version
    out.append((MARKETPLACE, market, drift))

    server = load(SERVER)
    drift = []
    if server.get("version") != version:
        drift.append(f"server.json version {server.get('version')!r} != {version!r}")
        server["version"] = version
    for pkg in server.get("packages", []):
        ident = pkg.get("identifier", "")
        wanted = retag(ident, version)
        if ident != wanted:
            drift.append(f"server.json package {ident!r} != {wanted!r}")
            pkg["identifier"] = wanted
    out.append((SERVER, server, drift))

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit non-zero without writing anything",
    )
    parser.add_argument(
        "--set",
        metavar="VERSION",
        help="write this version to VERSION first, then sync the manifests",
    )
    args = parser.parse_args()

    if args.set:
        if not SEMVER.match(args.set):
            sys.exit(f"--set expects MAJOR.MINOR.PATCH, got {args.set!r}")
        VERSION_FILE.write_text(args.set + "\n")
        print(f"VERSION -> {args.set}")

    version = read_version()
    results = plan(version)
    all_drift = [d for _, _, drift in results for d in drift]

    if args.check:
        if all_drift:
            print(f"Version drift against VERSION={version}:")
            for d in all_drift:
                print(f"  - {d}")
            print("\nRun: python tools/sync_version.py")
            return 1
        print(f"All manifests agree on version {version}")
        return 0

    if not all_drift:
        print(f"Already in sync at version {version}")
        return 0

    for path, data, drift in results:
        if drift:
            dump(path, data)
            print(f"Updated {path.relative_to(PROJECT_ROOT)}")
            for d in drift:
                print(f"  {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
