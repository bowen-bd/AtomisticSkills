"""Tests for venv/run, the launcher behind every skill command and MCP server.

The launcher decides *how* code runs -- the host's uv environment or a container
image -- and every misjudgement there turns into broken skills on somebody's
machine. The failures it guards against were found the hard way: a missing
runtime that Claude Code reports as ``Executable not found in $PATH: "stdio"``,
images pulled for an architecture the host cannot run, mksquashfs exhausting
``ulimit -u`` on a 448-core login node, and results written somewhere the agent
could not read them.

None of that needs uv, Docker or Apptainer to test. Each test copies the
launcher into a throwaway repository, puts stub executables on PATH (uv, docker,
apptainer, uname, getconf, nvidia-smi) that record how they were called, and
asserts on what the launcher decided. Stubbing uname and getconf lets every
architecture and glibc be simulated on any host.

Requirements:
    - Environment: cpu (run with: venv/run cpu python -m pytest tests/test_launcher.py)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = (PROJECT_ROOT / "VERSION").read_text().strip()


# Tools the tests decide about. A real one on PATH -- this repository's own
# development machine has /usr/bin/docker -- would turn "no container runtime"
# into a real image pull, so the system tools are offered without them.
HIDDEN = {
    "docker",
    "podman",
    "apptainer",
    "singularity",
    "uv",
    "uvx",
    "nvidia-smi",
    "conda",
    "mamba",
    "micromamba",
}


@pytest.fixture(scope="session")
def sysbin(tmp_path_factory) -> Path:
    """A directory of symlinks to the system's executables, minus HIDDEN."""
    out = tmp_path_factory.mktemp("sysbin")
    for d in ("/usr/local/bin", "/usr/bin", "/bin"):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            target = os.path.join(d, name)
            if (
                name in HIDDEN
                or (out / name).exists()
                or not os.access(target, os.X_OK)
            ):
                continue
            (out / name).symlink_to(target)
    return out


class Host:
    """A fake repository plus stub tools, and a way to run the launcher in it."""

    def __init__(self, tmp_path: Path, sysbin: Path):
        self.sysbin = sysbin
        self.tmp = tmp_path
        self.repo = tmp_path / "repo"
        self.bin = tmp_path / "bin"
        self.home = tmp_path / "home"
        self.logs = tmp_path / "logs"
        for d in (self.bin, self.home, self.logs):
            d.mkdir(parents=True)
        venv = self.repo / "venv"
        venv.mkdir(parents=True)
        shutil.copy(PROJECT_ROOT / "VERSION", self.repo / "VERSION")
        for name in ("run", "servers.tsv", "platforms.tsv"):
            shutil.copy(PROJECT_ROOT / "venv" / name, venv / name)
        for project in sorted(
            p.parent.name for p in (PROJECT_ROOT / "venv").glob("*/pyproject.toml")
        ):
            (venv / project).mkdir()
            for name in ("pyproject.toml", "uv.lock"):
                shutil.copy(
                    PROJECT_ROOT / "venv" / project / name, venv / project / name
                )
        self.launcher = venv / "run"
        self.workspace = tmp_path / "project"
        self.workspace.mkdir()
        self.arch = "aarch64"
        self.glibc = "2.39"
        self.stub_system()

    # --- stubs ---------------------------------------------------------------

    def stub(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)

    def stub_system(self) -> None:
        self.stub(
            "uname",
            f'case "$1" in -m) echo "{self.arch}";; -s) echo Linux;; *) /usr/bin/uname "$@";; esac\n',
        )
        self.stub(
            "getconf",
            f'[[ "$1" == GNU_LIBC_VERSION ]] && echo "glibc {self.glibc}" || /usr/bin/getconf "$@"\n',
        )

    def set_platform(self, arch: str, glibc: str) -> None:
        self.arch, self.glibc = arch, glibc
        self.stub_system()

    def recorder(self, name: str, extra: str = "", exit_code: int = 0) -> Path:
        """Stub `name` to log its argv (one call per line, args NUL-free) and environment."""
        log = self.logs / f"{name}.calls"
        self.stub(
            name,
            f'printf "%s\\x1f" "$@" >> "{log}"; printf "\\n" >> "{log}"\n'
            f'env > "{self.logs}/{name}.env"\n'
            f"{extra}\n"
            f"exit {exit_code}\n",
        )
        return log

    def stub_uv(self, sync_delay: float = 0) -> Path:
        """A uv that records calls and makes `uv sync` create the interpreter."""
        return self.recorder(
            "uv",
            extra=(
                'if [[ "$1" == sync ]]; then\n'
                f"  sleep {sync_delay}\n"
                '  for ((i=1;i<=$#;i++)); do if [[ "${!i}" == --project ]]; then j=$((i+1)); p="${!j}"; fi; done\n'
                '  mkdir -p "$p/.venv/bin" && : > "$p/.venv/bin/python" && chmod +x "$p/.venv/bin/python"\n'
                "fi\n"
                'if [[ "$1" == --version ]]; then echo "uv 0.0.0"; fi\n'
                # One line per call: what each uv invocation was given.
                'echo "$1 pref=${UV_PYTHON_PREFERENCE:-} ssl=${SSL_CERT_FILE:-}"'
                f' >> "{self.logs}/uv.envlines"'
            ),
        )

    def uv_env_lines(self) -> dict[str, str]:
        """{uv subcommand: 'pref=... ssl=...'} for the calls stub_uv saw."""
        path = self.logs / "uv.envlines"
        lines = path.read_text().splitlines() if path.exists() else []
        return {ln.split(" ", 1)[0]: ln.split(" ", 1)[1] for ln in lines}

    def stub_apptainer(self, exit_code: int = 0) -> Path:
        """An apptainer whose `build` leaves the output file, as the real one does."""
        return self.recorder(
            "apptainer",
            extra='if [[ "$1" == build ]]; then : > "$3"; fi',
            exit_code=exit_code,
        )

    @staticmethod
    def calls(log: Path) -> list[list[str]]:
        if not log.exists():
            return []
        return [
            line.split("\x1f")[:-1] for line in log.read_text().splitlines() if line
        ]

    def env_of(self, name: str) -> dict[str, str]:
        path = self.logs / f"{name}.env"
        pairs = [ln.split("=", 1) for ln in path.read_text().splitlines() if "=" in ln]
        return dict(pairs)

    def mark_synced(self, project: str, *extras: str) -> None:
        """Pretend `venv/run --setup` already ran for this project."""
        venv = self.repo / "venv" / project / ".venv"
        (venv / "bin").mkdir(parents=True, exist_ok=True)
        (venv / "bin" / "python").write_text("")
        (venv / "bin" / "python").chmod(0o755)
        digest = subprocess.run(
            ["sha256sum", str(self.repo / "venv" / project / "uv.lock")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout[:16]
        (venv / ".atomisticskills-synced").write_text(
            "\n".join([digest, *extras]) + "\n"
        )

    # --- running -------------------------------------------------------------

    def run(
        self, *args: str, cwd: Path | None = None, **env_overrides
    ) -> subprocess.CompletedProcess:
        env = {
            "PATH": f"{self.bin}:{self.sysbin}",
            "HOME": str(self.home),
            "ATOMISTIC_MODEL_CACHE": str(self.tmp / "cache"),
            "ATOMISTIC_SERVER_SETUP_WAIT": "3",
        }
        for key, value in env_overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return subprocess.run(
            ["bash", str(self.launcher), *args],
            cwd=cwd or self.workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )


@pytest.fixture
def host(tmp_path, sysbin):
    return Host(tmp_path, sysbin)


def flag_value(argv: list[str], flag: str) -> list[str]:
    """Every value that follows `flag` in argv."""
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


# =============================================================================
# Choosing a backend
# =============================================================================


class TestBackendChoice:
    def test_uv_when_the_host_supports_it(self, host):
        log = host.stub_uv()
        host.recorder("docker")
        host.mark_synced("cpu")
        result = host.run("cpu", "python", "script.py", "--x", "1")
        assert result.returncode == 0, result.stderr
        argv = host.calls(log)[-1]
        assert argv[:2] == ["run", "--frozen"]
        assert flag_value(argv, "--project") == [str(host.repo / "venv" / "cpu")]
        assert argv[-4:] == ["python", "script.py", "--x", "1"]
        assert "--no-sync" in argv
        assert not host.calls(
            host.logs / "docker.calls"
        ), "docker must not run when uv works"

    def test_container_when_glibc_is_too_old(self, host):
        host.set_platform("x86_64", "2.17")  # CentOS 7
        host.stub_uv()
        docker = host.recorder("docker")
        result = host.run("mlip", "python", "script.py")
        assert result.returncode == 0, result.stderr
        assert "needs glibc >= 2.28" in result.stderr
        argv = host.calls(docker)[-1]
        assert argv[0] == "run"
        assert f"ghcr.io/learningmatter-mit/atomisticskills-mlip:{VERSION}" in argv

    def test_extra_with_a_higher_floor_falls_back_alone(self, host):
        """glibc 2.31 runs the base cpu set natively, but OpenMM needs 2.34."""
        host.set_platform("x86_64", "2.31")
        uv = host.stub_uv()
        docker = host.recorder("docker")
        host.mark_synced("cpu")
        assert host.run("cpu", "python", "a.py").returncode == 0
        assert host.calls(uv) and not host.calls(docker)
        result = host.run("cpu+openmm", "python", "b.py")
        assert result.returncode == 0, result.stderr
        assert "'cpu+openmm' needs glibc >= 2.34" in result.stderr
        assert host.calls(docker)

    def test_container_when_uv_is_missing(self, host):
        apptainer = host.stub_apptainer()
        result = host.run("cpu", "python", "script.py")
        assert result.returncode == 0, result.stderr
        assert "uv is not installed" in result.stderr
        assert host.calls(apptainer)[-1][0] == "exec"

    def test_nothing_usable_explains_both_remedies(self, host):
        result = host.run("cpu", "python", "script.py")
        assert result.returncode != 0
        assert "install uv" in result.stderr.lower()
        assert "container runtime" in result.stderr

    def test_forced_runtime_missing_lists_what_exists(self, host):
        """The 'stdio' error that cost a tester a disassembly session."""
        host.recorder("docker")
        result = host.run("cpu", "python", "x.py", ATOMISTIC_RUNTIME="apptainer")
        assert result.returncode != 0
        assert "runtime 'apptainer' is not on PATH" in result.stderr
        assert "docker" in result.stderr
        assert "'runtime' option" in result.stderr

    def test_unknown_runtime_is_rejected(self, host):
        result = host.run("cpu", "python", "x.py", ATOMISTIC_RUNTIME="chroot")
        assert result.returncode != 0
        assert "unknown ATOMISTIC_RUNTIME 'chroot'" in result.stderr

    def test_empty_runtime_means_auto(self, host):
        """An unset plugin option arrives as an empty string."""
        host.stub_uv()
        host.mark_synced("cpu")
        assert host.run("cpu", "python", "x.py", ATOMISTIC_RUNTIME="").returncode == 0

    def test_config_file_sets_the_runtime(self, host):
        host.stub_uv()
        docker = host.recorder("docker")
        cfg = host.home / ".config" / "atomistic_skills.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            'MP_API_KEY: "abc"\nATOMISTIC_RUNTIME: docker  # containers\nATOMISTIC_IMAGE_TAG: "7.7.7"\n'
        )
        result = host.run("cpu", "python", "x.py")
        assert result.returncode == 0, result.stderr
        argv = host.calls(docker)[-1]
        assert "ghcr.io/learningmatter-mit/atomisticskills-cpu:7.7.7" in argv
        # The same file carries API keys; containers must be able to read it.
        assert f"{cfg}:/tmp/.config/atomistic_skills.yaml:ro" in flag_value(
            argv, "--volume"
        )

    def test_environment_overrides_the_config_file(self, host):
        uv = host.stub_uv()
        host.recorder("docker")
        host.mark_synced("cpu")
        cfg = host.home / ".config" / "atomistic_skills.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("ATOMISTIC_RUNTIME: docker\n")
        assert host.run("cpu", "python", "x.py", ATOMISTIC_RUNTIME="uv").returncode == 0
        assert host.calls(uv)

    def test_unknown_venv_is_reported(self, host):
        host.stub_uv()
        result = host.run("gpu", "python", "x.py")
        assert result.returncode != 0
        assert "unknown venv 'gpu'" in result.stderr


# =============================================================================
# The uv backend
# =============================================================================


class TestUvBackend:
    def test_first_use_syncs_then_runs(self, host):
        uv = host.stub_uv()
        result = host.run("mlip", "python", "x.py")
        assert result.returncode == 0, result.stderr
        calls = host.calls(uv)
        # Exact: a package dropped from the lock leaves the environment too.
        assert calls[0][0] == "sync" and "--inexact" not in calls[0]
        assert calls[-1][0] == "run"
        marker = host.repo / "venv" / "mlip" / ".venv" / ".atomisticskills-synced"
        assert marker.exists()

    def test_second_use_does_not_sync(self, host):
        uv = host.stub_uv()
        host.run("mlip", "python", "x.py")
        before = len(host.calls(uv))
        host.run("mlip", "python", "y.py")
        new_calls = host.calls(uv)[before:]
        assert [c[0] for c in new_calls] == ["run"]

    def test_a_changed_lock_resyncs(self, host):
        uv = host.stub_uv()
        host.mark_synced("cpu")
        with open(host.repo / "venv" / "cpu" / "uv.lock", "a") as fh:
            fh.write("\n# changed\n")
        host.run("cpu", "python", "x.py")
        assert host.calls(uv)[0][0] == "sync"

    def test_extras_are_passed_and_remembered(self, host):
        uv = host.stub_uv()
        host.mark_synced("cpu")
        assert host.run("cpu+openmm+pymol", "python", "x.py").returncode == 0
        sync = host.calls(uv)[0]
        assert sync[0] == "sync"
        assert flag_value(sync, "--extra") == ["openmm", "pymol"]
        before = len(host.calls(uv))
        host.run("cpu+openmm", "python", "x.py")
        assert [c[0] for c in host.calls(uv)[before:]] == ["run"]

    def test_a_resync_keeps_extras_installed_earlier(self, host):
        """Asking for one extra must never uninstall another."""
        uv = host.stub_uv()
        host.mark_synced("cpu", "openmm")
        with open(host.repo / "venv" / "cpu" / "uv.lock", "a") as fh:
            fh.write("\n# changed\n")
        assert host.run("cpu+pymol", "python", "x.py").returncode == 0
        sync = host.calls(uv)[0]
        assert flag_value(sync, "--extra") == ["openmm", "pymol"]

    def test_no_compiler_means_a_container_for_a_new_environment(self, host):
        host.stub_uv()
        docker = host.recorder("docker")
        hidden = host.tmp / "nocc"
        hidden.mkdir()
        for name in os.listdir(host.sysbin):
            if name not in ("cc", "gcc", "c++", "g++") and not name.startswith(
                ("gcc-", "cc-", "aarch64-linux-gnu-gcc", "x86_64-linux-gnu-gcc")
            ):
                (hidden / name).symlink_to(host.sysbin / name)
        host.sysbin = hidden
        result = host.run("cpu", "python", "x.py")
        assert result.returncode == 0, result.stderr
        assert "needs a C compiler" in result.stderr
        assert host.calls(docker)
        # An environment that already exists needs no compiler. (No GPU on this
        # fake host, so the mlip environment's torch build is cu130.)
        host.mark_synced("mlip", "cu130")
        uv_calls_before = len(host.calls(host.logs / "uv.calls"))
        assert host.run("mlip", "python", "x.py").returncode == 0
        assert len(host.calls(host.logs / "uv.calls")) > uv_calls_before

    def test_docking_on_arm_warns_about_boost(self, host):
        host.stub_uv()
        host.mark_synced("cpu")
        result = host.run("cpu+docking", "python", "x.py")
        if Path("/usr/include/boost").is_dir():
            pytest.skip("this host has the Boost headers")
        assert "Boost headers" in result.stderr

    def test_setup_syncs_every_project(self, host):
        uv = host.stub_uv()
        result = host.run("--setup")
        assert result.returncode == 0, result.stderr
        projects = {
            flag_value(c, "--project")[0] for c in host.calls(uv) if c[0] == "sync"
        }
        assert projects == {
            str(host.repo / "venv" / p) for p in ("cpu", "mlip", "fairchem")
        }


# =============================================================================
# MCP server mode
# =============================================================================


class TestServerMode:
    def test_starts_the_module_in_its_project(self, host):
        uv = host.stub_uv()
        host.mark_synced("mlip")
        result = host.run("--server", "mace", ATOMISTIC_WORKSPACE=str(host.workspace))
        assert result.returncode == 0, result.stderr
        argv = host.calls(uv)[-1]
        assert argv[0] == "run" and "--no-sync" in argv
        assert flag_value(argv, "--project") == [str(host.repo / "venv" / "mlip")]
        assert argv[-3:] == ["python", "-m", "src.mcp_server.mace_server"]
        assert host.env_of("uv")["ATOMISTIC_WORKSPACE"] == str(host.workspace)

    def test_workspace_defaults_to_the_current_directory(self, host):
        host.stub_uv()
        host.mark_synced("cpu")
        assert host.run("--server", "base").returncode == 0
        assert host.env_of("uv")["ATOMISTIC_WORKSPACE"] == str(host.workspace)

    def test_unknown_server_lists_the_real_ones(self, host):
        host.stub_uv()
        result = host.run("--server", "nope")
        assert result.returncode != 0
        assert "unknown MCP server 'nope'" in result.stderr
        assert "mace" in result.stderr and "base" in result.stderr

    def test_quick_first_sync_still_starts_the_server(self, host):
        uv = host.stub_uv(sync_delay=0.2)
        result = host.run("--server", "base")
        assert result.returncode == 0, result.stderr
        assert host.calls(uv)[-1][-3:] == ["python", "-m", "src.mcp_server.base_server"]

    def test_slow_first_sync_continues_in_the_background(self, host):
        """A multi-GB first sync must not die with the MCP client's timeout."""
        host.stub_uv(sync_delay=30)
        result = host.run("--server", "fairchem")
        assert result.returncode != 0
        assert "being created in the background" in result.stderr
        assert "--setup" in result.stderr
        assert (
            host.home / ".local" / "state" / "atomisticskills" / "setup-fairchem.log"
        ).exists()

    def test_generative_servers_run_from_their_uv_project(self, host):
        """adit, diffcsp and mattergen have x86_64 uv projects; the image is the
        fallback elsewhere (aarch64, old glibc)."""
        host.stub_uv()
        host.set_platform("x86_64", "2.39")
        host.mark_synced("adit", "cu130")
        result = host.run("--server", "adit")
        assert result.returncode == 0, result.stderr
        run = [c for c in host.calls(host.logs / "uv.calls") if c and c[0] == "run"][-1]
        assert flag_value(run, "--project")[0].endswith("venv/adit")
        assert run[-2:] == ["-m", "src.mcp_server.adit_server"]

    def test_container_only_server_is_refused_on_the_wrong_architecture(self, host):
        """No bytes may move for an image the host cannot run (29 GB once did).

        The generative image is arm64-only, so an x86_64 host too old for the uv
        project (glibc 2.17) must be refused rather than pull it."""
        host.stub_uv()
        host.set_platform("x86_64", "2.17")
        docker = host.recorder("docker")
        result = host.run("--server", "mattergen")
        assert result.returncode != 0
        assert "unavailable on this machine" in result.stderr
        assert "linux/amd64" in result.stderr
        assert not host.calls(docker)

    def test_container_only_server_runs_its_image(self, host):
        docker = host.recorder("docker")
        host.stub("nvidia-smi", 'echo "GPU 0: NVIDIA GB10"\n')
        result = host.run("--server", "diffcsp")
        assert result.returncode == 0, result.stderr
        argv = host.calls(docker)[-1]
        assert (
            argv[-2]
            == f"ghcr.io/learningmatter-mit/atomisticskills-generative:{VERSION}"
        )
        assert argv[-1] == "diffcsp"
        assert "--gpus" in argv


# =============================================================================
# Containers: what gets mounted, passed and requested
# =============================================================================


class TestDockerInvocation:
    def run_in_docker(self, host, *args, **env):
        docker = host.recorder("docker")
        result = host.run(*args, ATOMISTIC_RUNTIME="docker", **env)
        assert result.returncode == 0, result.stderr
        return host.calls(docker)[-1], host

    def test_same_path_mounts_and_identity(self, host):
        argv, _ = self.run_in_docker(host, "cpu", "python", "x.py")
        volumes = flag_value(argv, "--volume")
        assert f"{host.repo}:{host.repo}:ro" in volumes
        assert f"{host.workspace}:{host.workspace}" in volumes
        assert flag_value(argv, "--workdir") == [str(host.workspace)]
        assert flag_value(argv, "--user") == [f"{os.getuid()}:{os.getgid()}"]
        envs = flag_value(argv, "--env")
        assert f"PYTHONPATH={host.repo}" in envs
        assert f"ATOMISTIC_WORKSPACE={host.workspace}" in envs
        assert argv[-2:] == ["python", "x.py"]

    def test_a_command_names_its_environment(self, host):
        """An image may carry several environments (generative)."""
        argv, _ = self.run_in_docker(host, "mlip", "python", "x.py")
        assert "ATOMISTIC_VENV=mlip" in flag_value(argv, "--env")

    def test_repository_is_writable_when_it_is_the_workspace(self, host):
        argv, _ = self.run_in_docker(
            host, "cpu", "python", "x.py", ATOMISTIC_WORKSPACE=str(host.repo)
        )
        assert f"{host.repo}:{host.repo}:rw" in flag_value(argv, "--volume")

    def test_source_checkouts_are_mounted(self, host):
        """ADiT, DiffCSP++ and MatterGen import repositories that no image
        carries: the default checkout next to the repository, or the one a
        variable names."""
        adit = host.repo.parent / "adit"
        adit.mkdir()
        diffcsp = host.tmp / "elsewhere" / "DiffCSP-PP"
        diffcsp.mkdir(parents=True)
        argv, _ = self.run_in_docker(
            host, "cpu", "python", "x.py", DIFFCSP_REPO=str(diffcsp)
        )
        volumes, envs = flag_value(argv, "--volume"), flag_value(argv, "--env")
        assert f"{adit}:{adit}" in volumes and f"ADIT_REPO={adit}" in envs
        assert not any(e.startswith("MATTERGEN_REPO=") for e in envs), "none here"
        assert f"{diffcsp}:{diffcsp}" in volumes
        assert f"DIFFCSP_REPO={diffcsp}" in envs

    def test_a_cwd_outside_the_workspace_is_mounted_too(self, host):
        elsewhere = host.tmp / "elsewhere"
        elsewhere.mkdir()
        docker = host.recorder("docker")
        result = host.run(
            "cpu",
            "python",
            "x.py",
            cwd=elsewhere,
            ATOMISTIC_RUNTIME="docker",
            ATOMISTIC_WORKSPACE=str(host.workspace),
        )
        assert result.returncode == 0, result.stderr
        assert f"{elsewhere}:{elsewhere}" in flag_value(
            host.calls(docker)[-1], "--volume"
        )

    def test_gpu_only_for_gpu_images_on_gpu_hosts(self, host):
        argv, _ = self.run_in_docker(host, "mlip", "python", "x.py")
        assert "--gpus" not in argv, "no nvidia-smi on this fake host"
        host.stub("nvidia-smi", 'echo "GPU 0: NVIDIA A100"\n')
        argv, _ = self.run_in_docker(host, "mlip", "python", "x.py")
        assert "--gpus" in argv
        argv, _ = self.run_in_docker(host, "cpu", "python", "x.py")
        assert "--gpus" not in argv

    def test_credentials_only_when_set_and_non_empty(self, host):
        argv, _ = self.run_in_docker(host, "cpu", "python", "x.py", MP_API_KEY="secret")
        assert "MP_API_KEY=secret" in flag_value(argv, "--env")
        argv, _ = self.run_in_docker(host, "cpu", "python", "x.py", MP_API_KEY="")
        assert not any(e.startswith("MP_API_KEY") for e in flag_value(argv, "--env"))

    def test_registry_and_tag_are_configurable(self, host):
        argv, _ = self.run_in_docker(
            host,
            "fairchem",
            "python",
            "x.py",
            ATOMISTIC_IMAGE_REGISTRY="ghcr.io/someone",
            ATOMISTIC_IMAGE_TAG="1.2.3",
        )
        assert "ghcr.io/someone/atomisticskills-fairchem:1.2.3" in argv

    def test_server_mode_passes_the_server_name(self, host):
        argv, _ = self.run_in_docker(host, "--server", "smol")
        assert argv[-2:] == [
            f"ghcr.io/learningmatter-mit/atomisticskills-cpu:{VERSION}",
            "smol",
        ]


class TestApptainerInvocation:
    """Apptainer takes structurally different arguments and needs a cached SIF."""

    def run_apptainer(self, host, *args, exit_code=0, **env):
        log = host.stub_apptainer(exit_code=exit_code)
        result = host.run(*args, ATOMISTIC_RUNTIME="apptainer", **env)
        return result, host.calls(log)

    def test_builds_a_sif_then_execs_it(self, host):
        result, calls = self.run_apptainer(host, "--server", "base")
        assert result.returncode == 0, result.stderr
        assert calls[0][0] == "build" and calls[0][-1].startswith("docker://")
        argv = calls[-1]
        assert argv[0] == "exec"
        assert flag_value(argv, "--pwd") == [str(host.workspace)]
        # `exec` bypasses the image ENTRYPOINT, so server mode names it.
        assert "/opt/atomisticskills/docker/entrypoint.sh" in argv
        assert argv[-1] == "base"
        assert not any(a.startswith("docker://") for a in argv)

    def test_command_mode_runs_through_the_entrypoint(self, host):
        """`apptainer exec` skips the ENTRYPOINT, which puts the command's
        environment on PATH (the generative image carries several)."""
        result, calls = self.run_apptainer(host, "cpu", "python", "x.py")
        assert result.returncode == 0, result.stderr
        argv = calls[-1]
        assert argv[-3:] == [
            "/opt/atomisticskills/docker/entrypoint.sh",
            "python",
            "x.py",
        ]
        assert "ATOMISTIC_VENV=cpu" in flag_value(argv, "--env")

    def test_reuses_an_existing_sif(self, host):
        sif = host.tmp / "cache" / "sif" / f"atomisticskills-cpu-{VERSION}.sif"
        sif.parent.mkdir(parents=True)
        sif.write_text("pretend SIF")
        result, calls = self.run_apptainer(host, "cpu", "python", "x.py")
        assert result.returncode == 0, result.stderr
        assert [c[0] for c in calls] == ["exec"]
        assert sif.read_text() == "pretend SIF"

    def test_finds_a_prebuilt_sif_in_the_shared_cache(self, host):
        """The pre-build and the launcher must agree without guessing paths."""
        shared = host.home / ".cache" / "atomisticskills" / "sif"
        shared.mkdir(parents=True)
        (shared / f"atomisticskills-mlip-{VERSION}.sif").write_text("prebuilt")
        result, calls = self.run_apptainer(host, "mlip", "python", "x.py")
        assert result.returncode == 0, result.stderr
        assert [c[0] for c in calls] == ["exec"]
        assert any(str(shared) in a for a in calls[-1])

    def test_gpu_uses_nv(self, host):
        host.stub("nvidia-smi", 'echo "GPU 0: NVIDIA H100"\n')
        result, calls = self.run_apptainer(host, "mlip", "python", "x.py")
        assert result.returncode == 0, result.stderr
        assert "--nv" in calls[-1] and "--gpus" not in calls[-1]

    def test_paths_under_home_are_not_rebound(self, host):
        """Apptainer binds $HOME itself; binding it again only adds noise."""
        workspace = host.home / "project"
        workspace.mkdir()
        result, calls = self.run_apptainer(
            host, "cpu", "python", "x.py", ATOMISTIC_WORKSPACE=str(workspace)
        )
        assert result.returncode == 0, result.stderr
        binds = flag_value(calls[-1], "--bind")
        assert not any(b.startswith(str(host.home)) for b in binds)
        assert f"{host.repo}:{host.repo}" in binds

    def test_build_failure_is_reported_with_the_tuning_hint(self, host):
        result, calls = self.run_apptainer(host, "cpu", "python", "x.py", exit_code=1)
        assert result.returncode != 0
        assert "failed to build" in result.stderr
        assert "ATOMISTIC_SQUASHFS_PROCS" in result.stderr
        assert [c[0] for c in calls] == ["build"]

    def test_mksquashfs_threads_are_bounded(self, host):
        host.stub(
            "apptainer",
            f'printf "%s\\n" "$APPTAINER_MKSQUASHFS_ARGS|$APPTAINER_TMPDIR" >> "{host.logs}/squash"\n'
            'if [[ "$1" == build ]]; then : > "$3"; fi\nexit 0\n',
        )
        assert (
            host.run("cpu", "python", "x.py", ATOMISTIC_RUNTIME="apptainer").returncode
            == 0
        )
        args, tmpdir = (host.logs / "squash").read_text().splitlines()[0].split("|")
        count = int(args.split("-processors")[1].split()[0])
        assert 1 <= count <= 4
        # Clusters mount /tmp nodev, which breaks builds: keep it by the cache.
        assert tmpdir == str(host.tmp / "cache" / "apptainer-tmp")


# =============================================================================
# Workspace: where research/ goes
# =============================================================================


class TestWorkspace:
    def doctor_workspace(self, host, cwd, **env):
        host.stub_uv()
        result = host.run("--doctor", cwd=cwd, **env)
        line = next(
            ln
            for ln in result.stderr.splitlines()
            if ln.strip().startswith("workspace")
        )
        return line.split(":", 1)[1].strip()

    def test_explicit_workspace_wins(self, host):
        assert (
            self.doctor_workspace(host, host.workspace, ATOMISTIC_WORKSPACE="/srv/x")
            == "/srv/x"
        )

    def test_inside_the_checkout_it_is_the_repository(self, host):
        sub = host.repo / "venv"
        assert self.doctor_workspace(host, sub) == str(host.repo)

    def test_elsewhere_it_is_the_current_directory(self, host):
        assert self.doctor_workspace(host, host.workspace) == str(host.workspace)

    @pytest.mark.parametrize("where", ["repo", "outside", "override"])
    def test_python_side_agrees_with_the_launcher(self, tmp_path, where):
        """src/utils/research_utils.workspace_root applies the same rule."""
        cwd = {
            "repo": PROJECT_ROOT / "skills",
            "outside": tmp_path,
            "override": tmp_path,
        }[where]
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
        env.pop("ATOMISTIC_WORKSPACE", None)
        if where == "override":
            env["ATOMISTIC_WORKSPACE"] = "/srv/elsewhere"
        out = subprocess.run(
            [
                "python",
                "-c",
                "from src.utils.research_utils import workspace_root; print(workspace_root())",
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        expected = {
            "repo": str(PROJECT_ROOT),
            "outside": str(tmp_path.resolve()),
            "override": "/srv/elsewhere",
        }
        assert out == expected[where]


class TestHostQuirks:
    """Failures found on an RHEL 8 HPC cluster and its older CentOS 7 GPU nodes."""

    def test_environments_are_built_on_a_managed_python(self, host):
        """A system python3.12 without its -devel package cannot compile smol."""
        host.stub_uv()
        result = host.run("cpu", "python", "x.py", cwd=host.workspace)
        assert result.returncode == 0, result.stderr
        lines = host.uv_env_lines()
        assert lines["sync"].startswith("pref=only-managed")
        # Running an existing environment must not force it: uv would warn on
        # every command for environments built before this default.
        assert lines["run"].startswith("pref= ")

    def test_a_chosen_python_preference_is_kept(self, host):
        host.stub_uv()
        host.run(
            "cpu", "python", "x.py", cwd=host.workspace, UV_PYTHON_PREFERENCE="system"
        )
        assert host.uv_env_lines()["sync"].startswith("pref=system")

    def test_the_uv_backend_gets_the_system_ca_bundle(self, host):
        """A managed CPython does not find RHEL's bundle on its own."""
        host.stub_uv()
        host.mark_synced("cpu")
        host.run("cpu", "python", "x.py", cwd=host.workspace)
        expected = ""
        if not os.path.exists("/etc/ssl/cert.pem"):
            for path in (
                "/etc/pki/tls/certs/ca-bundle.crt",
                "/etc/ssl/certs/ca-certificates.crt",
                "/etc/ssl/ca-bundle.pem",
            ):
                if os.path.exists(path):
                    expected = path
                    break
        assert host.uv_env_lines()["run"].endswith(f"ssl={expected}")

    def test_a_chosen_ca_bundle_is_kept(self, host):
        host.stub_uv()
        host.mark_synced("cpu")
        host.run(
            "cpu", "python", "x.py", cwd=host.workspace, SSL_CERT_FILE="/corp/ca.pem"
        )
        assert host.uv_env_lines()["run"].endswith("ssl=/corp/ca.pem")

    def test_containers_do_not_get_the_host_ca_bundle(self, host):
        """The host path would not exist inside the image."""
        log = host.recorder("docker")
        result = host.run("cpu", "python", "x.py", ATOMISTIC_RUNTIME="docker")
        assert result.returncode == 0, result.stderr
        assert not any("SSL_CERT_FILE" in a for a in host.calls(log)[-1])

    def test_possibly_empty_arrays_are_expanded_safely(self):
        """bash < 4.4 (CentOS 7 ships 4.2) treats "${a[@]}" of an empty array as
        unbound under `set -u`: `venv/run: line 471: extras[@]: unbound variable`."""
        text = (PROJECT_ROOT / "venv" / "run").read_text()
        unsafe = re.findall(r'(?<!\+)"\$\{(extras|previous|binds)\[@\]\}"', text)
        assert not unsafe, f'use ${{a[@]+"${{a[@]}}"}} for: {sorted(set(unsafe))}'

    def test_an_image_that_cannot_be_pulled_says_so(self, host):
        host.recorder(
            "apptainer",
            extra=(
                'if [[ "$1" == build ]]; then echo "FATAL: While performing build: '
                'DENIED: requested access to the resource is denied"; exit 1; fi'
            ),
        )
        result = host.run("cpu", "python", "x.py", ATOMISTIC_RUNTIME="apptainer")
        assert result.returncode != 0
        assert "cannot pull" in result.stderr
        assert "ATOMISTIC_IMAGE_REGISTRY" in result.stderr
        assert "ATOMISTIC_SQUASHFS_PROCS" not in result.stderr

    @pytest.mark.parametrize(
        "driver, expected",
        [
            ("470.256.02", "older than 525"),
            ("550.54.14", "CUDA 12.6 builds (cu126)"),
            ("580.173.02", None),
        ],
    )
    def test_doctor_reports_what_the_driver_allows(self, host, driver, expected):
        host.stub(
            "nvidia-smi",
            f'if [[ "$*" == *name,driver_version* ]]; then echo "NVIDIA GeForce RTX 2080 Ti, {driver}";\n'
            f'elif [[ "$*" == *driver_version* ]]; then echo "{driver}";\n'
            'else echo "GPU 0: NVIDIA GeForce RTX 2080 Ti"; fi\n',
        )
        result = host.run("--doctor")
        if expected:
            assert expected in result.stderr
        else:
            assert "older than" not in result.stderr and "cu126" not in result.stderr

    def test_old_glibc_is_reported_before_a_missing_compiler(self, host, tmp_path):
        """On glibc 2.17 the compiler is not the problem; saying so misleads."""
        no_cc = tmp_path / "no_cc"
        no_cc.mkdir()
        for entry in host.sysbin.iterdir():
            if entry.name not in {
                "cc",
                "gcc",
                "c++",
                "g++",
                "clang",
            } and not entry.name.startswith(("gcc-", "cc-")):
                (no_cc / entry.name).symlink_to(os.readlink(entry))
        host.stub_uv()
        host.set_platform("x86_64", "2.17")
        result = host.run("--doctor", PATH=f"{host.bin}:{no_cc}")
        assert "needs glibc >= 2.28; this host has 2.17" in result.stderr
        assert "C compiler" not in result.stderr

    def test_an_inherited_virtual_env_is_dropped(self, host):
        """A server started from another environment's shell must not make uv warn
        that the active environment does not match the project."""
        host.stub_uv()
        host.mark_synced("mlip")
        host.run(
            "mlip",
            "python",
            "x.py",
            cwd=host.workspace,
            VIRTUAL_ENV=str(host.repo / "venv" / "cpu" / ".venv"),
        )
        assert "VIRTUAL_ENV" not in host.env_of("uv")


class TestTorchCudaBuild:
    """mlip and fairchem lock a CUDA 13 and a CUDA 12.6 torch; venv/run picks one."""

    def stub_driver(self, host, version):
        host.stub(
            "nvidia-smi",
            f'if [[ "$*" == *driver_version* ]]; then echo "{version}"; else echo "GPU 0: NVIDIA A100"; fi\n',
        )

    def sync_extras(self, host):
        syncs = [c for c in host.calls(host.logs / "uv.calls") if c and c[0] == "sync"]
        return [syncs[-1][i + 1] for i, a in enumerate(syncs[-1]) if a == "--extra"]

    @pytest.mark.parametrize(
        "driver, build",
        [(None, "cu130"), ("550.107.02", "cu126"), ("580.65.06", "cu130")],
    )
    def test_the_driver_picks_the_build(self, host, driver, build):
        host.stub_uv()
        if driver:
            self.stub_driver(host, driver)
        result = host.run("mlip", "python", "x.py", cwd=host.workspace)
        assert result.returncode == 0, result.stderr
        assert self.sync_extras(host) == [build]
        run = [c for c in host.calls(host.logs / "uv.calls") if c and c[0] == "run"][-1]
        assert flag_value(run, "--extra") == [build]

    def test_cpu_has_no_build_to_pick(self, host):
        host.stub_uv()
        self.stub_driver(host, "550.107.02")
        host.run("cpu", "python", "x.py", cwd=host.workspace)
        assert self.sync_extras(host) == []

    def test_the_override_wins(self, host):
        host.stub_uv()
        self.stub_driver(host, "580.65.06")
        host.run(
            "fairchem",
            "python",
            "x.py",
            cwd=host.workspace,
            ATOMISTIC_TORCH_CUDA="cu126",
        )
        assert self.sync_extras(host) == ["cu126"]

    def test_a_bad_override_is_refused(self, host):
        result = host.run("mlip", "python", "x.py", ATOMISTIC_TORCH_CUDA="cu999")
        assert result.returncode != 0 and "cu126 or cu130" in result.stderr

    def test_switching_builds_drops_the_other_one(self, host):
        """cu126 and cu130 conflict: carrying the old one over would fail the sync."""
        host.stub_uv()
        host.mark_synced("mlip", "cu130", "openmm")
        self.stub_driver(host, "550.107.02")
        host.run("mlip", "python", "x.py", cwd=host.workspace)
        assert sorted(self.sync_extras(host)) == ["cu126", "openmm"]

    def test_servers_use_the_build_too(self, host):
        host.stub_uv()
        self.stub_driver(host, "550.107.02")
        host.mark_synced("mlip", "cu126")
        host.run("--server", "mace")
        run = [c for c in host.calls(host.logs / "uv.calls") if c and c[0] == "run"][-1]
        assert flag_value(run, "--extra") == ["cu126"]

    def test_doctor_lists_one_gpu_line_on_a_multi_gpu_host(self, host):
        """head -1 closed the pipe on 8 GPUs; under pipefail that also printed 'none'."""
        lines = "\n".join(["NVIDIA A100 80GB PCIe, 550.107.02"] * 8)
        host.stub(
            "nvidia-smi",
            f'if [[ "$*" == *name,driver_version* ]]; then printf "%s\\n" "{lines}";\n'
            'elif [[ "$*" == *driver_version* ]]; then for i in 1 2 3 4 5 6 7 8; do echo 550.107.02; done;\n'
            'else echo "GPU 0: NVIDIA A100"; fi\n',
        )
        result = host.run("--doctor")
        gpu = [ln for ln in result.stderr.splitlines() if ln.strip().startswith("GPU")]
        assert gpu == ["  GPU        : NVIDIA A100 80GB PCIe, 550.107.02"]
        assert "\nnone\n" not in result.stderr
