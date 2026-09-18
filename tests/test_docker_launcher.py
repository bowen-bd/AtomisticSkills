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
import sys
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


class TestPrebuildSifDiscovery:
    """prepare_images.sh must not rebuild a SIF that already exists elsewhere.

    The two scripts resolve the cache differently on a second run: the first
    pre-build lands in the shared cache (the plugin's data directory does not
    exist until Claude Code has run once), and afterwards that data directory
    does exist and becomes the target. An HPC node spent 15m12s rebuilding a
    1.2 GB image it already had, so both sides search the same list.
    """

    PREPARE = PROJECT_ROOT / "docker" / "prepare_images.sh"

    @staticmethod
    def _run(tmp_path, home, extra_env=None):
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        log = tmp_path / "apptainer.log"
        stub = bindir / "apptainer"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$*" >> "{log}"\n'
            'if [[ "${1:-}" == "build" ]]; then\n'
            '  for a in "$@"; do case "$a" in *.sif|*.partial) : > "$a"; break;; esac; done\n'
            "fi\nexit 0\n"
        )
        stub.chmod(0o755)
        env = {
            "HOME": str(home),
            "PATH": f"{bindir}:/usr/bin:/bin",
        }
        env.update(extra_env or {})
        result = subprocess.run(
            [
                "bash",
                str(TestPrebuildSifDiscovery.PREPARE),
                "--runtime",
                "apptainer",
                "--registry",
                "ghcr.io/example",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        invocations = log.read_text() if log.exists() else ""
        return result, invocations

    def test_does_not_rebuild_a_sif_present_in_the_shared_cache(self, tmp_path):
        version = (PROJECT_ROOT / "VERSION").read_text().strip()
        home = tmp_path / "home"
        shared = home / ".cache" / "atomisticskills" / "sif"
        shared.mkdir(parents=True)
        (shared / f"atomisticskills-lightweight-{version}.sif").write_text("prebuilt")
        # A prior session created the plugin data dir, so CACHE resolves there.
        (
            home
            / ".claude"
            / "plugins"
            / "data"
            / "atomistic-skills-atomistic-skills"
            / "model-cache"
        ).mkdir(parents=True)

        result, invocations = self._run(tmp_path, home)
        assert result.returncode == 0, result.stderr
        assert "have  lightweight" in result.stderr, result.stderr
        assert (
            "atomisticskills-lightweight" not in invocations
        ), f"rebuilt a SIF that already existed: {invocations}"

    def test_builds_when_the_sif_is_genuinely_absent(self, tmp_path):
        home = tmp_path / "home"
        (home / ".cache").mkdir(parents=True)
        result, invocations = self._run(tmp_path, home)
        assert result.returncode == 0, result.stderr
        assert (
            "atomisticskills-lightweight" in invocations
        ), "expected a build when nothing is cached"


class TestMultiArchCoverage:
    """Every declared platform must actually be buildable and reachable.

    Six of ten servers were arm64-only, so an x86_64 cluster -- the common case
    for HPC -- got four working servers and six architecture refusals. Declaring
    a platform without the lockfiles to build it would turn that clear refusal
    into a build failure, and declaring one the plugin never advertises would
    leave the gate refusing an image that exists.
    """

    SUBDIR = {"linux/amd64": "linux-64", "linux/arm64": "linux-aarch64"}

    def test_declared_platforms_have_their_lockfiles(self):
        spec = json.loads(IMAGES_SPEC.read_text())
        problems = []
        for image in spec["images"]:
            if not image["gpu"]:
                continue
            env_python = image.get("env_python", {})
            for platform in image["platforms"]:
                subdir = self.SUBDIR[platform]
                for env in image["envs"]:
                    pip = PROJECT_ROOT / f"conda-envs/{env}/lock/pip-{subdir}.txt"
                    if not pip.exists():
                        problems.append(f"{image['name']}/{platform}: {pip} missing")
                    conda = PROJECT_ROOT / f"conda-envs/{env}/lock/conda-{subdir}.txt"
                    if not conda.exists() and env not in env_python:
                        problems.append(
                            f"{image['name']}/{platform}: no conda lock for {env} "
                            "and no env_python entry to build a bare one"
                        )
        assert not problems, "\n".join(problems)

    def test_plugin_platforms_match_the_image_spec(self):
        """The arch gate reads these; drift refuses an image that exists."""
        spec = json.loads(IMAGES_SPEC.read_text())
        manifest = json.loads(
            (PROJECT_ROOT / ".claude-plugin" / "plugin.json").read_text()
        )
        for image in spec["images"]:
            for server in image["servers"]:
                declared = manifest["mcpServers"][server]["env"]["ATOMISTIC_PLATFORMS"]
                assert declared == ",".join(image["platforms"]), (
                    f"{server}: plugin says {declared!r}, images.json says "
                    f"{','.join(image['platforms'])!r}; run render.py plugin-mcp"
                )

    def test_amd64_gpu_images_do_not_compile_pyg_from_source(self):
        """x86_64 has prebuilt PyG CUDA wheels; compiling them wastes an hour."""
        sys.path.insert(0, str(PROJECT_ROOT / "docker"))
        import render

        spec = json.loads(IMAGES_SPEC.read_text())
        for image in spec["images"]:
            if not image["gpu"] or "linux/amd64" not in image["platforms"]:
                continue
            assert not render.per_platform(
                image, "pyg_from_source", "linux/amd64", False
            ), f"{image['name']} would compile PyG from source on amd64"


class TestPerPlatformSettings:
    """Build settings are scalar where shared, keyed by platform where not."""

    def test_scalar_applies_to_every_platform(self):
        sys.path.insert(0, str(PROJECT_ROOT / "docker"))
        import render

        image = {"torch_cuda_arch_list": "12.1"}
        assert (
            render.per_platform(image, "torch_cuda_arch_list", "linux/amd64", "x")
            == "12.1"
        )
        assert (
            render.per_platform(image, "torch_cuda_arch_list", "linux/arm64", "x")
            == "12.1"
        )

    def test_mapping_selects_by_platform(self):
        sys.path.insert(0, str(PROJECT_ROOT / "docker"))
        import render

        image = {"torch_cuda_arch_list": {"linux/arm64": "12.1", "linux/amd64": "9.0"}}
        assert (
            render.per_platform(image, "torch_cuda_arch_list", "linux/amd64", "x")
            == "9.0"
        )
        assert (
            render.per_platform(image, "torch_cuda_arch_list", "linux/arm64", "x")
            == "12.1"
        )

    def test_missing_key_falls_back_to_the_default(self):
        sys.path.insert(0, str(PROJECT_ROOT / "docker"))
        import render

        assert (
            render.per_platform({}, "absent", "linux/amd64", "fallback") == "fallback"
        )
        image = {"absent": {"linux/arm64": "only-arm"}}
        assert (
            render.per_platform(image, "absent", "linux/amd64", "fallback")
            == "fallback"
        )


class TestEntrypointDispatch:
    """The entrypoint takes a server name, which makes the image hard to inspect.

    `docker run IMG micromamba run -n mace-agent python -c ...` is how you check
    what a stack actually contains, and an ENTRYPOINT that only accepts server
    names answers it with "unknown server 'micromamba'". A GPU tester lost time
    to exactly that. Runnable names now fall through to exec.
    """

    ENTRYPOINT = PROJECT_ROOT / "docker" / "entrypoint.sh"

    @staticmethod
    def _run(tmp_path, *args):
        repo = tmp_path / "repo" / "docker"
        repo.mkdir(parents=True, exist_ok=True)
        (repo / "server-map.txt").write_text(
            "# server:env:module\n"
            "mace:mace-agent:src.mcp_server.mace_server\n"
            "matgl:matgl-agent:src.mcp_server.matgl_server\n"
        )
        env = dict(os.environ)
        env["ATOMISTIC_REPO_DIR"] = str(tmp_path / "repo")
        # Skip the privilege-drop re-exec; it is not what these tests cover.
        env["ATOMISTIC_UID_MATCHED"] = "1"
        return subprocess.run(
            ["bash", str(TestEntrypointDispatch.ENTRYPOINT), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_runnable_name_is_executed_as_a_command(self, tmp_path):
        result = self._run(tmp_path, "bash", "-c", "echo EXECUTED")
        assert result.returncode == 0, result.stderr
        assert "EXECUTED" in result.stdout
        # The fallback must announce itself, so a mistyped server name is never
        # silently some other command.
        assert "not a server here" in result.stderr

    def test_unknown_non_runnable_name_still_errors(self, tmp_path):
        result = self._run(tmp_path, "definitely-not-a-real-binary-xyz")
        assert result.returncode != 0
        assert "unknown server" in result.stderr
        assert "mace matgl" in result.stderr
        assert "--entrypoint" in result.stderr, "should point at the escape hatch"

    def test_no_argument_lists_the_servers(self, tmp_path):
        result = self._run(tmp_path)
        assert result.returncode != 0
        assert "mace matgl" in result.stderr


class TestSquashfsThreadRetry:
    """A thread count that works cannot be predicted, so failure must recover.

    `ulimit -u` counts every process the user has on the node, not just this
    build, and mksquashfs spawns several threads per -processors unit. A cap of
    8 still died with "Failed to create thread" on a busy 448-core login node
    whose limit was 768. Guessing a smaller constant is the same mistake; the
    launcher retries single-threaded instead.
    """

    @staticmethod
    def _stub(tmp_path, succeed_only_at: str):
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        log = tmp_path / "apptainer.log"
        stub = bindir / "apptainer"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            # Log builds only; the final exec also inherits the env var.
            'if [[ "${1:-}" == "build" ]]; then\n'
            f'  printf "%s\\n" "$APPTAINER_MKSQUASHFS_ARGS" >> "{log}"\n'
            f'  if [[ "$APPTAINER_MKSQUASHFS_ARGS" != "{succeed_only_at}" ]]; then\n'
            '    echo "FATAL: while creating squashfs: mksquashfs command failed: '
            'exit status 1: FATAL ERROR: Failed to create thread" >&2\n'
            "    exit 255\n  fi\n"
            '  for a in "$@"; do case "$a" in *.sif|*.partial) : > "$a"; break;; esac; done\n'
            "fi\nexit 0\n"
        )
        stub.chmod(0o755)
        return bindir, log

    def test_falls_back_to_single_threaded(self, tmp_path):
        bindir, log = self._stub(tmp_path, succeed_only_at="-processors 1")
        result = run_launcher(
            tmp_path, bindir, ATOMISTIC_RUNTIME="apptainer", HOME=str(tmp_path / "home")
        )
        assert result.returncode == 0, result.stderr
        assert "retrying single-threaded" in result.stderr
        attempts = [a for a in log.read_text().splitlines() if a.strip()]
        assert attempts[-1] == "-processors 1", attempts
        assert len(attempts) >= 2, "should have tried a higher count first"

    def test_first_attempt_is_bounded_well_below_the_core_count(self, tmp_path):
        bindir, log = self._stub(tmp_path, succeed_only_at="-processors 1")
        run_launcher(
            tmp_path, bindir, ATOMISTIC_RUNTIME="apptainer", HOME=str(tmp_path / "home")
        )
        first = log.read_text().splitlines()[0]
        procs = int(first.rsplit(None, 1)[1])
        assert 1 <= procs <= 4, f"first attempt used {procs}; cap is 4"

    def test_a_non_thread_failure_is_not_retried(self, tmp_path):
        """Only the thread limit is recoverable; masking other errors would hide bugs."""
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        log = tmp_path / "apptainer.log"
        stub = bindir / "apptainer"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == "build" ]]; then '
            f'printf "%s\\n" "$APPTAINER_MKSQUASHFS_ARGS" >> "{log}"; '
            'echo "FATAL: no space left on device" >&2; exit 255; fi\n'
            "exit 0\n"
        )
        stub.chmod(0o755)
        result = run_launcher(
            tmp_path, bindir, ATOMISTIC_RUNTIME="apptainer", HOME=str(tmp_path / "home")
        )
        assert result.returncode != 0
        assert "retrying single-threaded" not in result.stderr
        assert len([a for a in log.read_text().splitlines() if a.strip()]) == 1
