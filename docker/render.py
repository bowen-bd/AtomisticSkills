#!/usr/bin/env python3
"""Render everything derived from docker/images.json.

``images.json`` is the single source of truth for which MCP server lives in
which container image. This script projects it into the three consumers that
would otherwise drift out of sync:

``server-map <image>``
    A dependency-free ``server:env:module`` table baked into each image and
    read by ``docker/entrypoint.sh``. Shell-parseable on purpose: the
    entrypoint has to resolve a server name before any conda env is active,
    so it cannot rely on a Python interpreter being on PATH.

``plugin-mcp``
    The ``mcpServers`` block for ``.claude-plugin/plugin.json``, wiring every
    server to ``docker run`` against its image. Writes the file in place
    unless ``--stdout`` is given.

``matrix``
    The GitHub Actions build matrix, one entry per (image, platform).

Usage:
    # Env: base-agent
    python docker/render.py server-map lightweight
    python docker/render.py plugin-mcp
    python docker/render.py matrix

Requirements:
    - Conda environment: base-agent (standard library only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_SPEC = PROJECT_ROOT / "docker" / "images.json"
PLUGIN_MANIFEST = PROJECT_ROOT / ".claude-plugin" / "plugin.json"


def load_spec() -> dict:
    """Return the parsed image specification."""
    return json.loads(IMAGES_SPEC.read_text())


def find_image(spec: dict, name: str) -> dict:
    """Return the image entry called ``name``, or exit with the valid names."""
    for image in spec["images"]:
        if image["name"] == name:
            return image
    valid = ", ".join(i["name"] for i in spec["images"])
    sys.exit(f"unknown image {name!r}; expected one of: {valid}")


def normalise_servers(image: dict) -> dict[str, dict]:
    """Return ``{server: {env, module}}`` for either image strategy.

    A merged image names its single conda env once and maps servers straight to
    modules; a locked image carries one env per server and spells both out.
    """
    out = {}
    for server, value in image["servers"].items():
        if isinstance(value, str):
            out[server] = {"env": image["merged_env"], "module": value}
        else:
            out[server] = {"env": value["env"], "module": value["module"]}
    return out


def cmd_server_map(spec: dict, args: argparse.Namespace) -> int:
    """Print the shell-parseable server table for one image."""
    image = find_image(spec, args.image)
    env_vars = image.get("env_vars", {})
    lines = [
        "# server:env:module[:KEY=VAL,KEY=VAL]",
        f"# generated from docker/images.json for image {image['name']}",
    ]
    for server, info in sorted(normalise_servers(image).items()):
        row = f"{server}:{info['env']}:{info['module']}"
        extra = env_vars.get(server)
        if extra:
            row += ":" + ",".join(f"{k}={v}" for k, v in sorted(extra.items()))
        lines.append(row)
    print("\n".join(lines))
    return 0


def per_platform(image: dict, key: str, platform: str, default):
    """Return a setting that may be either shared or keyed by platform.

    Build settings started out identical across architectures and were plain
    scalars. GPU target lists are not: arm64 targets GB10 (sm_121) alone, while
    amd64 has to cover a spread of datacenter and consumer cards. Accept both
    shapes so the scalar form stays valid where nothing differs.
    """
    value = image.get(key, default)
    if isinstance(value, dict):
        return value.get(platform, default)
    return value


def cmd_matrix(spec: dict, args: argparse.Namespace) -> int:
    """Print the GitHub Actions matrix as JSON, one entry per image+platform."""
    runner = {"linux/amd64": "ubuntu-24.04", "linux/arm64": "ubuntu-24.04-arm"}
    subdir = {"linux/amd64": "linux-64", "linux/arm64": "linux-aarch64"}
    include = []
    for image in spec["images"]:
        for platform in image["platforms"]:
            entry = {
                "image": image["name"],
                "platform": platform,
                "arch": platform.split("/")[1],
                "runner": runner[platform],
                "dockerfile": (
                    "docker/Dockerfile.cuda"
                    if image["gpu"]
                    else "docker/Dockerfile.lightweight"
                ),
                # Passed through as --build-arg; the CUDA Dockerfile installs
                # one conda env per name and needs the matching lock subdir.
                "build_args": {
                    "IMAGE_NAME": image["name"],
                    "CONDA_ENVS": " ".join(image["envs"]),
                    "CONDA_SUBDIR": subdir[platform],
                    "PYG_FROM_SOURCE": (
                        "1"
                        if per_platform(image, "pyg_from_source", platform, False)
                        else "0"
                    ),
                    "TORCH_CUDA_ARCH_LIST": per_platform(
                        image, "torch_cuda_arch_list", platform, "12.1"
                    ),
                    # Only consulted where no conda lock exists for the subdir.
                    "CONDA_ENV_PYTHON": " ".join(
                        f"{env}={py}" for env, py in image.get("env_python", {}).items()
                    ),
                    # The PyTorch CUDA index is for the aarch64 wheels. The
                    # amd64 pins resolve against PyPI, whose x86_64 torch wheels
                    # are already CUDA builds; adding a second index there can
                    # silently substitute a different build of the same version.
                    "PIP_EXTRA_INDEX_URL": (
                        ""
                        if platform == "linux/amd64"
                        else "https://download.pytorch.org/whl/cu130"
                    ),
                },
                # Only images built for more than one platform need a manifest
                # list stitched together afterwards.
                "multiarch": len(image["platforms"]) > 1,
            }
            include.append(entry)
    print(json.dumps({"include": include}))
    return 0


def cmd_plugin_mcp(spec: dict, args: argparse.Namespace) -> int:
    """Write (or print) the plugin manifest's mcpServers block."""
    servers: dict[str, dict] = {}

    # Every server goes through docker/run_server.sh rather than an inline
    # `docker run` argument list. Docker and Apptainer take structurally
    # different arguments, and one static list in this manifest cannot serve
    # both -- which mattered the moment the plugin was tested on an HPC node,
    # where there is no Docker daemon and Apptainer is the norm. The script
    # picks the runtime and translates; configuration reaches it through env.
    #
    # The registry is a user config value, not a constant: CI publishes to
    # ghcr.io/<repository_owner>, so a fork's images live in the fork's
    # namespace and a hardcoded owner would point testers at images that do
    # not exist for them.
    for image in spec["images"]:
        ref = (
            "${user_config.image_registry}"
            f"/atomisticskills-{image['name']}"
            ":${user_config.image_tag}"
        )
        for server in sorted(normalise_servers(image)):
            servers[server] = {
                "command": "${CLAUDE_PLUGIN_ROOT}/docker/run_server.sh",
                "args": [server],
                "env": {
                    "ATOMISTIC_RUNTIME": "${user_config.container_runtime}",
                    "ATOMISTIC_IMAGE": ref,
                    "ATOMISTIC_IMAGE_NAME": image["name"],
                    # Lets the launcher refuse an image this architecture
                    # cannot run before downloading gigabytes of it.
                    "ATOMISTIC_PLATFORMS": ",".join(image["platforms"]),
                    "ATOMISTIC_WORK_DIR": "${user_config.work_dir}",
                    # Checkpoints are multi-GB and must outlive plugin updates.
                    "ATOMISTIC_MODEL_CACHE": "${CLAUDE_PLUGIN_DATA}/model-cache",
                    "ATOMISTIC_GPU": "1" if image["gpu"] else "0",
                },
            }

    if args.stdout:
        print(json.dumps({"mcpServers": servers}, indent=2))
        return 0

    manifest = json.loads(PLUGIN_MANIFEST.read_text())
    manifest["mcpServers"] = servers
    PLUGIN_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {len(servers)} servers to "
        f"{PLUGIN_MANIFEST.relative_to(PROJECT_ROOT)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("server-map", help="shell table of servers for one image")
    p.add_argument("image", help="image name from docker/images.json")
    p.set_defaults(func=cmd_server_map)

    p = sub.add_parser("matrix", help="GitHub Actions build matrix as JSON")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("plugin-mcp", help="mcpServers block for plugin.json")
    p.add_argument(
        "--stdout",
        action="store_true",
        help="print instead of writing .claude-plugin/plugin.json",
    )
    p.set_defaults(func=cmd_plugin_mcp)

    args = parser.parse_args()
    return args.func(load_spec(), args)


if __name__ == "__main__":
    sys.exit(main())
