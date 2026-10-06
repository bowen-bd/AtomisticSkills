#!/usr/bin/env python3
"""Promote this CI run's images only after every selected build has succeeded.

Build jobs push run-specific staging tags. Resolve all of them to digests before
changing any release tag, so missing builds cannot be replaced by old images.
The workflow gates this command on successful builds and CPU smoke tests.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def publish(matrix: dict, registry: str, run_id: str, version: str) -> None:
    """Validate the selected platforms, resolve staging digests and publish tags."""
    spec = json.loads((ROOT / "docker/images.json").read_text())
    expected = {image["name"]: set(image["platforms"]) for image in spec["images"]}
    selected: dict[str, set[str]] = {}
    for entry in matrix["include"]:
        selected.setdefault(entry["image"], set()).add(entry["platform"])
    if not selected:
        raise ValueError("No images selected for publication")
    for name, platforms in selected.items():
        if name not in expected or platforms != expected[name]:
            raise ValueError(
                f"Incomplete or unknown image: {name} ({sorted(platforms)})"
            )
    if not run_id.isdigit():
        raise ValueError("run_id must be a GitHub Actions run ID")

    # Finish resolving the entire run before advancing any mutable tag. A retry
    # of failed jobs retains its run ID, so already successful builds are usable.
    resolved: dict[str, dict[str, str]] = {}
    for name, platforms in selected.items():
        base = f"{registry.lower()}/atomisticskills-{name}"
        resolved[base] = {}
        for platform in sorted(platforms):
            arch = platform.split("/")[1]
            staging = f"{base}:build-{run_id}-{arch}"
            result = subprocess.run(
                [
                    "docker",
                    "buildx",
                    "imagetools",
                    "inspect",
                    staging,
                    "--format",
                    "{{json .Manifest}}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            digest = json.loads(result.stdout)["digest"]
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid digest for {staging}: {digest}")
            resolved[base][arch] = f"{base}@{digest}"

    for base, sources in resolved.items():
        subprocess.run(
            [
                "docker",
                "buildx",
                "imagetools",
                "create",
                "-t",
                f"{base}:{version}",
                "-t",
                f"{base}:latest",
                *sources.values(),
            ],
            check=True,
        )
        if len(sources) > 1:
            for arch, source in sources.items():
                subprocess.run(
                    [
                        "docker",
                        "buildx",
                        "imagetools",
                        "create",
                        "--prefer-index=false",
                        "-t",
                        f"{base}:{version}-{arch}",
                        "-t",
                        f"{base}:latest-{arch}",
                        source,
                    ],
                    check=True,
                )


def main() -> None:
    """Read the exact build matrix selected by the workflow's preflight job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="Selected build matrix as JSON")
    parser.add_argument(
        "--registry", required=True, help="Registry and repository owner"
    )
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    publish(
        json.loads(args.matrix),
        args.registry,
        args.run_id,
        (ROOT / "VERSION").read_text().strip(),
    )


if __name__ == "__main__":
    main()
