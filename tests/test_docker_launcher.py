"""Tests for docker/run_server.sh, the container launcher behind every MCP server.

The launcher is the single point where a misconfiguration turns into ten dead
servers, and the failures it guards against were all found the hard way on an
HPC node: a missing runtime reported by Claude Code as the meaningless
``Executable not found in $PATH: "stdio"``, gigabytes downloaded for images the
host's architecture cannot run, and mksquashfs exhausting ``ulimit -u`` on a
448-core machine.

None of that needs Docker or Apptainer to test. The launcher's job is to decide
*what command to run*, so these tests put a stub runtime on PATH that records
its arguments and exits, then assert on what the launcher decided.

Requirements:
    - Conda environment: base-agent
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "docker" / "run_server.sh"
IMAGES_SPEC = PROJECT_ROOT / "docker" / "images.json"

HOST_ARCH = "arm64" if os.uname().machine in ("aarch64", "arm64") else "amd64"
OTHER_ARCH = "amd64" if HOST_ARCH == "arm64" else "arm64"


@pytest.fixture
def stub_runtime(tmp_path):
    """Put a fake container runtime on PATH that records how it was invoked.

    Returns a factory: ``stub_runtime("apptainer")`` creates the executable and
    returns a callable giving the recorded argv of the last invocation.
    """

    bindir = tmp_path / "bin"
    bindir.mkdir()

    def make(name: str, exit_code: int = 0):
        argv_log = tmp_path / f"{name}.argv"
        stub = bindir / name
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$@" > "{argv_log}"\n'
            # A build invocation must leave a file behind, or the launcher
            # rightly refuses to exec an image that was never produced.
            'if [[ "${1:-}" == "build" ]]; then : > "$3"; fi\n'
            f"exit {exit_code}\n"
        )
        stub.chmod(0o755)

        def recorded() -> list[str]:
            if not argv_log.exists():
                return []
            return argv_log.read_text().splitlines()

        return recorded

    make.bindir = bindir
    return make


def run_launcher(tmp_path, stub_bindir, server="base", **env_overrides):
    """Invoke the launcher with a controlled environment and return the result."""
    env = dict(os.environ)
    env["PATH"] = f"{stub_bindir}:{env['PATH']}"
    env.update(
        {
            "ATOMISTIC_IMAGE": "ghcr.io/example/atomisticskills-lightweight:9.9.9",
            "ATOMISTIC_IMAGE_NAME": "lightweight",
            "ATOMISTIC_PLATFORMS": f"linux/{HOST_ARCH}",
            "ATOMISTIC_WORK_DIR": str(tmp_path / "work"),
            "ATOMISTIC_MODEL_CACHE": str(tmp_path / "cache"),
            "ATOMISTIC_GPU": "0",
        }
    )
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        ["bash", str(LAUNCHER), server],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestArchitectureGate:
    """An image the host cannot run must be refused before anything downloads."""

    def test_refuses_image_built_for_another_architecture(self, tmp_path, stub_runtime):
        recorded = stub_runtime("docker")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="docker",
            ATOMISTIC_PLATFORMS=f"linux/{OTHER_ARCH}",
        )
        assert result.returncode != 0
        assert "unavailable on this machine" in result.stderr
        assert f"linux/{HOST_ARCH}" in result.stderr
        # The decisive point: the runtime was never invoked, so no bytes moved.
        assert recorded() == [], "runtime was called despite an architecture mismatch"

    def test_allows_multi_arch_image(self, tmp_path, stub_runtime):
        recorded = stub_runtime("docker")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="docker",
            ATOMISTIC_PLATFORMS="linux/amd64,linux/arm64",
        )
        assert result.returncode == 0, result.stderr
        assert recorded(), "runtime should have been invoked"

    def test_no_platform_declared_does_not_block(self, tmp_path, stub_runtime):
        """An unset platform list must not become an accidental deny-all."""
        recorded = stub_runtime("docker")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="docker",
            ATOMISTIC_PLATFORMS=None,
        )
        assert result.returncode == 0, result.stderr
        assert recorded()


class TestMissingRuntime:
    """The 'stdio' error that cost a tester a disassembly session."""

    def test_names_the_missing_runtime_and_lists_alternatives(
        self, tmp_path, stub_runtime
    ):
        stub_runtime("docker")
        result = run_launcher(
            tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="apptainer"
        )
        assert result.returncode != 0
        assert "'apptainer' is not on PATH" in result.stderr
        # It must point at what the host does have, not just what is missing.
        assert "docker" in result.stderr
        assert "container_runtime" in result.stderr

    def test_rejects_unsupported_runtime_name(self, tmp_path, stub_runtime):
        stub_runtime("docker")
        result = run_launcher(tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="chroot")
        assert result.returncode != 0
        assert "unsupported container_runtime" in result.stderr


class TestDockerInvocation:
    def test_builds_expected_docker_command(self, tmp_path, stub_runtime):
        recorded = stub_runtime("docker")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            server="drugdisc",
            ATOMISTIC_RUNTIME="docker",
        )
        assert result.returncode == 0, result.stderr
        argv = recorded()
        assert argv[:3] == ["run", "--rm", "--interactive"]
        assert f"{tmp_path / 'work'}:/work" in argv
        assert "/work" in argv
        # Server name is the final argument, image immediately before it.
        assert argv[-1] == "drugdisc"
        assert argv[-2].endswith("atomisticskills-lightweight:9.9.9")

    def test_gpu_flag_only_when_requested(self, tmp_path, stub_runtime):
        recorded = stub_runtime("docker")
        run_launcher(
            tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="docker", ATOMISTIC_GPU="0"
        )
        assert "--gpus" not in recorded()

        recorded = stub_runtime("docker")
        run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="docker",
            ATOMISTIC_GPU="1",
            ATOMISTIC_PLATFORMS=f"linux/{HOST_ARCH}",
        )
        assert "--gpus" in recorded()

    def test_forwards_credentials_only_when_set(self, tmp_path, stub_runtime):
        recorded = stub_runtime("docker")
        run_launcher(
            tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="docker", MP_API_KEY=None
        )
        assert not any(a.startswith("MP_API_KEY") for a in recorded())

        recorded = stub_runtime("docker")
        run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="docker",
            MP_API_KEY="secret-value",
        )
        assert "MP_API_KEY=secret-value" in recorded()

    def test_empty_credential_is_not_forwarded(self, tmp_path, stub_runtime):
        """An exported-but-empty key must not shadow the server's own default."""
        recorded = stub_runtime("docker")
        run_launcher(
            tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="docker", MP_API_KEY=""
        )
        assert not any(a.startswith("MP_API_KEY") for a in recorded())


class TestApptainerInvocation:
    """Apptainer's arguments are structurally different from Docker's."""

    def test_builds_sif_then_execs_it(self, tmp_path, stub_runtime):
        recorded = stub_runtime("apptainer")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            server="smol",
            ATOMISTIC_RUNTIME="apptainer",
        )
        assert result.returncode == 0, result.stderr
        argv = recorded()
        # The final invocation is the exec, not the build.
        assert argv[0] == "exec"
        assert "--bind" in argv
        assert f"{tmp_path / 'work'}:/work" in argv
        assert "--pwd" in argv
        # Apptainer must be handed the entrypoint explicitly, since `exec`
        # bypasses the image ENTRYPOINT.
        assert "/opt/atomisticskills/docker/entrypoint.sh" in argv
        assert argv[-1] == "smol"

    def test_execs_a_cached_sif_not_a_docker_uri(self, tmp_path, stub_runtime):
        recorded = stub_runtime("apptainer")
        run_launcher(tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="apptainer")
        argv = recorded()
        sif_args = [a for a in argv if a.endswith(".sif")]
        assert sif_args, f"expected a .sif argument, got {argv}"
        assert not any(
            a.startswith("docker://") for a in argv
        ), "exec should use the cached SIF, not re-resolve docker:// each launch"

    def test_reuses_an_existing_sif_without_rebuilding(self, tmp_path, stub_runtime):
        sif_dir = tmp_path / "cache" / "sif"
        sif_dir.mkdir(parents=True)
        sif = sif_dir / "atomisticskills-lightweight-9.9.9.sif"
        sif.write_text("pretend SIF")

        recorded = stub_runtime("apptainer")
        result = run_launcher(
            tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="apptainer"
        )
        assert result.returncode == 0, result.stderr
        assert recorded()[0] == "exec"
        assert "building" not in result.stderr
        assert sif.read_text() == "pretend SIF", "existing SIF was overwritten"

    def test_finds_a_sif_the_prebuild_left_in_the_shared_cache(
        self, tmp_path, stub_runtime, monkeypatch
    ):
        """The pre-build and the launcher must agree without guessing paths.

        prepare_images.sh runs before the first Claude Code session, so
        CLAUDE_PLUGIN_DATA -- and therefore ATOMISTIC_MODEL_CACHE -- does not
        exist yet and it falls back to ~/.cache/atomisticskills. An HPC test
        then sat through four 30s connect timeouts with a valid 1.2 GB SIF
        already on disk. The launcher must look there.
        """
        fake_home = tmp_path / "home"
        shared = fake_home / ".cache" / "atomisticskills" / "sif"
        shared.mkdir(parents=True)
        (shared / "atomisticskills-lightweight-9.9.9.sif").write_text("prebuilt")

        recorded = stub_runtime("apptainer")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="apptainer",
            HOME=str(fake_home),
        )
        assert result.returncode == 0, result.stderr
        argv = recorded()
        assert argv[0] == "exec", f"expected exec, not a rebuild: {argv}"
        assert "building" not in result.stderr
        sif_args = [a for a in argv if a.endswith(".sif")]
        assert (
            sif_args and str(shared) in sif_args[0]
        ), f"launcher ignored the pre-built SIF at {shared}: {argv}"

    def test_model_cache_wins_over_the_shared_fallback(self, tmp_path, stub_runtime):
        """A SIF beside the runtime model cache is preferred when present."""
        fake_home = tmp_path / "home"
        shared = fake_home / ".cache" / "atomisticskills" / "sif"
        shared.mkdir(parents=True)
        (shared / "atomisticskills-lightweight-9.9.9.sif").write_text("shared")

        primary = tmp_path / "cache" / "sif"
        primary.mkdir(parents=True)
        (primary / "atomisticskills-lightweight-9.9.9.sif").write_text("primary")

        recorded = stub_runtime("apptainer")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="apptainer",
            HOME=str(fake_home),
        )
        assert result.returncode == 0, result.stderr
        sif_args = [a for a in recorded() if a.endswith(".sif")]
        assert sif_args and str(primary) in sif_args[0], sif_args

    def test_builds_when_no_prebuild_exists_anywhere(self, tmp_path, stub_runtime):
        """With nothing cached the launcher still builds into the model cache."""
        fake_home = tmp_path / "home"
        (fake_home / ".cache").mkdir(parents=True)

        recorded = stub_runtime("apptainer")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="apptainer",
            HOME=str(fake_home),
        )
        assert result.returncode == 0, result.stderr
        assert "building" in result.stderr
        sif_args = [a for a in recorded() if a.endswith(".sif")]
        assert sif_args and str(tmp_path / "cache") in sif_args[0], sif_args

    def test_gpu_uses_nv_not_gpus_flag(self, tmp_path, stub_runtime):
        recorded = stub_runtime("apptainer")
        run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="apptainer",
            ATOMISTIC_GPU="1",
        )
        argv = recorded()
        assert "--nv" in argv
        assert "--gpus" not in argv

    def test_build_failure_is_reported_not_silently_execd(self, tmp_path, stub_runtime):
        """A failed build must not fall through to exec'ing a nonexistent SIF."""
        recorded = stub_runtime("apptainer", exit_code=1)
        result = run_launcher(
            tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="apptainer"
        )
        assert result.returncode != 0
        assert "failed to build SIF" in result.stderr
        # The thread-exhaustion hint is what makes this actionable on HPC.
        assert "ATOMISTIC_SQUASHFS_PROCS" in result.stderr
        assert recorded()[0] == "build"


class TestSquashfsThreadBounding:
    """mksquashfs defaults to one thread per core and dies against ulimit -u."""

    def test_processors_are_bounded(self, tmp_path, stub_runtime):
        stub_runtime("apptainer")
        probe = tmp_path / "bin" / "apptainer"
        # Record the environment the launcher exports to the runtime.
        env_log = tmp_path / "env.txt"
        probe.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$APPTAINER_MKSQUASHFS_ARGS" > "{env_log}"\n'
            'if [[ "${1:-}" == "build" ]]; then : > "$3"; fi\n'
            "exit 0\n"
        )
        probe.chmod(0o755)
        result = run_launcher(
            tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="apptainer"
        )
        assert result.returncode == 0, result.stderr
        args = env_log.read_text().strip()
        assert "-processors" in args, f"mksquashfs left unbounded: {args!r}"
        count = int(args.split("-processors")[1].split()[0])
        assert 1 <= count <= 8, f"unreasonable processor count {count}"

    def test_explicit_override_is_respected(self, tmp_path, stub_runtime):
        stub_runtime("apptainer")
        env_log = tmp_path / "env.txt"
        probe = tmp_path / "bin" / "apptainer"
        probe.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$APPTAINER_MKSQUASHFS_ARGS" > "{env_log}"\n'
            'if [[ "${1:-}" == "build" ]]; then : > "$3"; fi\n'
            "exit 0\n"
        )
        probe.chmod(0o755)
        run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="apptainer",
            ATOMISTIC_SQUASHFS_PROCS="2",
        )
        assert "-processors 2" in env_log.read_text()

    def test_tmpdir_avoids_system_tmp(self, tmp_path, stub_runtime):
        """Clusters mount /tmp with nodev, which breaks Apptainer builds."""
        stub_runtime("apptainer")
        env_log = tmp_path / "env.txt"
        probe = tmp_path / "bin" / "apptainer"
        probe.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$APPTAINER_TMPDIR" > "{env_log}"\n'
            'if [[ "${1:-}" == "build" ]]; then : > "$3"; fi\n'
            "exit 0\n"
        )
        probe.chmod(0o755)
        run_launcher(tmp_path, stub_runtime.bindir, ATOMISTIC_RUNTIME="apptainer")
        tmpdir = env_log.read_text().strip()
        assert tmpdir, "APPTAINER_TMPDIR was not set"
        # It must live under the configured cache rather than defaulting to the
        # system /tmp, which clusters mount nodev.
        assert tmpdir.startswith(
            str(tmp_path / "cache")
        ), f"build temp should sit under the model cache, got {tmpdir}"


class TestManifestConsistency:
    """plugin.json is generated; drift between it and images.json is silent."""

    def test_every_server_declares_its_platforms_and_image(self):
        spec = json.loads(IMAGES_SPEC.read_text())
        manifest = json.loads(
            (PROJECT_ROOT / ".claude-plugin" / "plugin.json").read_text()
        )
        expected = {
            server: image for image in spec["images"] for server in image["servers"]
        }
        assert set(manifest["mcpServers"]) == set(expected)

        for server, image in expected.items():
            env = manifest["mcpServers"][server]["env"]
            assert env["ATOMISTIC_IMAGE_NAME"] == image["name"]
            assert env["ATOMISTIC_PLATFORMS"] == ",".join(image["platforms"])
            assert env["ATOMISTIC_GPU"] == ("1" if image["gpu"] else "0")

    def test_launcher_is_executable(self):
        """A non-executable launcher means every server fails to start."""
        assert os.access(LAUNCHER, os.X_OK), f"{LAUNCHER} is not executable"

    def test_default_image_tag_matches_the_release(self):
        """A default install must not pull images older than the plugin.

        entrypoint.sh and src/ are baked into the image, so pairing 1.3.5
        plugin code with a 1.3.4 default tag silently ships none of the fixes
        in this release to anyone who does not pass --config image_tag.
        """
        version = (PROJECT_ROOT / "VERSION").read_text().strip()
        manifest = json.loads(
            (PROJECT_ROOT / ".claude-plugin" / "plugin.json").read_text()
        )
        default = manifest["userConfig"]["image_tag"]["default"]
        assert default == version, (
            f"userConfig.image_tag default {default!r} != VERSION {version!r}; "
            "run python tools/sync_version.py"
        )


class TestWorkspaceResolution:
    """The container must not treat the read-only repository as a workspace."""

    def test_workspace_root_honours_the_override(self, tmp_path, monkeypatch):
        """Inside a container the repository is read-only; /work is not.

        Without this override create_research_dir fails on Apptainer with
        "[Errno 30] Read-only file system: '/opt/atomisticskills/research'".
        """
        import importlib

        monkeypatch.setenv("ATOMISTIC_WORKSPACE", str(tmp_path / "work"))
        research_utils = importlib.import_module("src.utils.research_utils")
        assert research_utils.workspace_root() == (tmp_path / "work").absolute()

    def test_workspace_root_defaults_to_repo_for_local_checkouts(self, monkeypatch):
        import importlib

        monkeypatch.delenv("ATOMISTIC_WORKSPACE", raising=False)
        research_utils = importlib.import_module("src.utils.research_utils")
        assert research_utils.workspace_root() == PROJECT_ROOT

    def test_research_dir_is_created_under_the_override(self, tmp_path, monkeypatch):
        import importlib

        workspace = tmp_path / "work"
        workspace.mkdir()
        monkeypatch.setenv("ATOMISTIC_WORKSPACE", str(workspace))
        research_utils = importlib.import_module("src.utils.research_utils")
        created = research_utils.create_new_research_dir("unit_test_topic")
        assert (
            workspace in created.parents
        ), f"research dir {created} escaped the workspace {workspace}"
        assert created.is_dir()


class TestArchGateOrdering:
    """The gate must fire before any other required-variable check."""

    def test_refusal_does_not_require_the_image_reference(self, tmp_path, stub_runtime):
        """Reported from HPC: the gate was unreachable without ATOMISTIC_IMAGE."""
        recorded = stub_runtime("docker")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            server="mace",
            ATOMISTIC_RUNTIME="docker",
            ATOMISTIC_IMAGE=None,
            ATOMISTIC_IMAGE_NAME="mace",
            ATOMISTIC_PLATFORMS=f"linux/{OTHER_ARCH}",
        )
        assert result.returncode != 0
        assert "unavailable on this machine" in result.stderr
        assert "ATOMISTIC_IMAGE is not set" not in result.stderr
        assert recorded() == []

    def test_missing_image_still_reported_when_arch_matches(
        self, tmp_path, stub_runtime
    ):
        recorded = stub_runtime("docker")
        result = run_launcher(
            tmp_path,
            stub_runtime.bindir,
            ATOMISTIC_RUNTIME="docker",
            ATOMISTIC_IMAGE=None,
        )
        assert result.returncode != 0
        assert "ATOMISTIC_IMAGE is not set" in result.stderr
        assert recorded() == []
