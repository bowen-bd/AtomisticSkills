"""Everything derived from docker/images.json must agree with it and with reality.

plugin.json, venv/servers.tsv and the in-image server maps are rendered from
images.json; drift between them is silent until a server fails to start on a
user's machine. Image definitions must also point at things that exist: a uv
project for a uv image, lockfiles for a conda-lock image, a module for every
server.

Requirements:
    - Environment: cpu (run with: venv/run cpu python -m pytest tests/test_images_and_manifests.py)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((PROJECT_ROOT / "docker" / "images.json").read_text())
PLUGIN = json.loads((PROJECT_ROOT / ".claude-plugin" / "plugin.json").read_text())
VERSION = (PROJECT_ROOT / "VERSION").read_text().strip()
SUBDIR = {"linux/amd64": "linux-64", "linux/arm64": "linux-aarch64"}

sys.path.insert(0, str(PROJECT_ROOT / "docker"))
import render  # noqa: E402


@pytest.mark.parametrize("what", ["servers", "plugin-mcp"])
def test_rendered_files_are_current(what):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "docker" / "render.py"), what, "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_server_has_a_module():
    for image in SPEC["images"]:
        for server, info in render.normalise_servers(image).items():
            module = PROJECT_ROOT / (info["module"].replace(".", "/") + ".py")
            assert module.is_file(), f"{server}: {module} missing"


def test_uv_images_name_real_projects():
    for image in SPEC["images"]:
        if image["build"] == "uv":
            assert (PROJECT_ROOT / "venv" / image["venv"] / "uv.lock").is_file(), image[
                "name"
            ]


def test_conda_lock_images_have_their_lockfiles():
    problems = []
    for image in SPEC["images"]:
        if image["build"] != "conda-lock":
            continue
        for platform in image["platforms"]:
            for env in image["envs"]:
                for kind in ("pip", "conda"):
                    lock = (
                        PROJECT_ROOT
                        / f"conda-envs/{env}/lock/{kind}-{SUBDIR[platform]}.txt"
                    )
                    if not lock.exists() and not (
                        kind == "conda" and env in image.get("env_python", {})
                    ):
                        problems.append(str(lock.relative_to(PROJECT_ROOT)))
    assert not problems, problems


def test_plugin_servers_go_through_the_launcher():
    for server, config in PLUGIN["mcpServers"].items():
        assert config["command"] == "${CLAUDE_PLUGIN_ROOT}/venv/run", server
        assert config["args"] == ["--server", server]
        # The same project the agent's scripts run in, so paths agree.
        assert config["env"]["ATOMISTIC_WORKSPACE"] == "${CLAUDE_PROJECT_DIR}"
        assert config["env"]["ATOMISTIC_RUNTIME"] == "${user_config.runtime}"


def test_plugin_options_are_optional_with_defaults():
    """A non-interactive install must work without --config flags."""
    options = PLUGIN["userConfig"]
    assert set(options) == {"runtime", "image_registry", "image_tag"}
    for name, option in options.items():
        assert option.get("default") not in (None, ""), name
        assert not option.get("required", False), name
    assert options["runtime"]["default"] == "auto"


def test_default_image_tag_matches_the_release():
    """Pairing new plugin code with old images silently drops this release's fixes."""
    assert PLUGIN["userConfig"]["image_tag"]["default"] == VERSION


def test_registry_packages_are_the_built_images():
    """server.json must list the images CI builds, each with the servers it holds."""
    server_json = json.loads((PROJECT_ROOT / "server.json").read_text())
    packages = {}
    for package in server_json["packages"]:
        repo = package["identifier"].rsplit(":", 1)[0]
        packages[repo.rsplit("atomisticskills-", 1)[1]] = sorted(
            package["packageArguments"][0]["choices"]
        )
    assert packages == {
        image["name"]: sorted(image["servers"]) for image in SPEC["images"]
    }


def test_manifest_versions_agree():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "sync_version.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("path", ["venv/run", "docker/entrypoint.sh"])
def test_launchers_are_executable(path):
    """A non-executable launcher means every server fails to start."""
    assert os.access(PROJECT_ROOT / path, os.X_OK)


def test_dockerfile_copies_existing_files():
    text = (PROJECT_ROOT / "docker" / "Dockerfile").read_text()
    for line in text.splitlines():
        if line.startswith("COPY ") and "--from" not in line:
            for src in line.split()[1:-1]:
                if "$" in src:
                    continue
                assert (PROJECT_ROOT / src).exists(), f"Dockerfile copies missing {src}"


def test_build_matrix_covers_every_image_and_platform():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        render.cmd_matrix(SPEC, None)
    entries = json.loads(buf.getvalue())["include"]
    expected = {(i["name"], p) for i in SPEC["images"] for p in i["platforms"]}
    assert {(e["image"], e["platform"]) for e in entries} == expected
    for e in entries:
        assert (PROJECT_ROOT / e["dockerfile"]).is_file()


class TestPerPlatformSettings:
    """Build settings are scalar where shared, keyed by platform where not."""

    def test_scalar_applies_to_every_platform(self):
        image = {"torch_cuda_arch_list": "12.1"}
        assert (
            render.per_platform(image, "torch_cuda_arch_list", "linux/amd64", "x")
            == "12.1"
        )

    def test_dict_is_looked_up_per_platform(self):
        image = {"torch_cuda_arch_list": {"linux/arm64": "12.1"}}
        assert (
            render.per_platform(image, "torch_cuda_arch_list", "linux/arm64", "x")
            == "12.1"
        )
        assert (
            render.per_platform(image, "torch_cuda_arch_list", "linux/amd64", "x")
            == "x"
        )


def test_research_dir_is_created_under_the_workspace(tmp_path, monkeypatch):
    """Inside a container the repository is read-only; the workspace is not.

    Without ATOMISTIC_WORKSPACE, create_research_dir once failed on Apptainer
    with "[Errno 30] Read-only file system: '/opt/atomisticskills/research'".
    """
    import importlib

    monkeypatch.setenv("ATOMISTIC_WORKSPACE", str(tmp_path))
    research_utils = importlib.import_module("src.utils.research_utils")
    created = research_utils.create_new_research_dir("unit_test_topic")
    assert tmp_path in created.parents and created.is_dir()
