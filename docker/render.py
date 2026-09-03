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
                    "PYG_FROM_SOURCE": "1" if image.get("pyg_from_source") else "0",
                    "TORCH_CUDA_ARCH_LIST": image.get("torch_cuda_arch_list", "12.1"),
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
    registry = spec["registry"]
    servers: dict[str, dict] = {}

    for image in spec["images"]:
        ref = f"{registry}/atomisticskills-{image['name']}:${{user_config.image_tag}}"
        for server in sorted(normalise_servers(image)):
            docker_args = [
                "run",
                "--rm",
                "--interactive",
                # Bind the user's project so skills can read inputs and write
                # results to a path that outlives the container.
                "--volume",
                "${user_config.work_dir}:/work",
                "--workdir",
                "/work",
                # Model checkpoints are multi-GB and must survive both the
                # container and plugin updates.
                "--volume",
                "${CLAUDE_PLUGIN_DATA}/model-cache:/opt/model-cache",
                "--env",
                "HF_HOME=/opt/model-cache/huggingface",
                "--env",
                "TORCH_HOME=/opt/model-cache/torch",
                "--env",
                "MATGL_CACHE=/opt/model-cache/matgl",
            ]
            if image["gpu"]:
                docker_args += ["--gpus", "all"]
            docker_args += [ref, server]

            servers[server] = {
                "command": "${user_config.container_runtime}",
                "args": docker_args,
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
