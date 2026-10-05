"""The uv projects under venv/ must stay installable, consistent and complete.

Each project is resolved separately, so nothing but these tests stops them
drifting apart: a package added to one project's shared CPU set and not the
others, a lock left behind by a pyproject edit, an extra declared in one place
only, or a glibc table that no longer matches the locks it describes.

Requirements:
    - Environment: cpu (run with: venv/run cpu python -m pytest tests/test_uv_projects.py)
    - ``uv`` on PATH for the lock checks (skipped without it)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / "venv"
PROJECTS = ("cpu", "mlip", "fairchem")
ARCHES = {
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'linux' and platform_machine == 'aarch64'",
}
# What each GPU project adds on top of the shared CPU set.
ADDED = {
    "cpu": set(),
    "mlip": {
        "torch",
        "mace-torch",
        "matgl",
        "nvalchemi-toolkit",
        "pytorch-lightning",
        "wandb",
    },
    "fairchem": {"torch", "fairchem-core", "nvalchemi-toolkit"},
}


def load(project: str) -> dict:
    return tomllib.loads((VENV_DIR / project / "pyproject.toml").read_text())


def names(project: str) -> dict[str, str]:
    """{canonical name: requirement string} of a project's base dependencies."""
    out = {}
    for spec in load(project)["project"]["dependencies"]:
        out[Requirement(spec).name.lower().replace("_", "-")] = spec
    return out


needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")


def test_shared_cpu_set_is_identical():
    """A package added to one project's CPU set and not the others is drift."""
    shared = {
        p: {k: v for k, v in names(p).items() if k not in ADDED[p]} for p in PROJECTS
    }
    assert shared["cpu"] == shared["mlip"] == shared["fairchem"], {
        p: sorted(set(shared[p]) ^ set(shared["cpu"])) for p in PROJECTS
    }


@pytest.mark.parametrize("project", PROJECTS)
def test_the_repository_is_an_editable_dependency(project):
    """Without it, `from src...` fails whenever the cwd is not the repo root."""
    data = load(project)
    assert "atomisticskills" in names(project)
    assert data["tool"]["uv"]["sources"]["atomisticskills"] == {
        "path": "../..",
        "editable": True,
    }


@pytest.mark.parametrize("project", PROJECTS)
def test_both_architectures_are_required(project):
    """`environments` limits the resolver; only `required-environments` checks wheels."""
    uv = load(project)["tool"]["uv"]
    assert set(uv["environments"]) == ARCHES
    assert set(uv["required-environments"]) == ARCHES


@pytest.mark.parametrize("project", PROJECTS)
def test_extras_are_the_same_everywhere(project):
    extras = dict(load(project)["project"]["optional-dependencies"])
    # The GPU projects add their two torch builds; everything else matches cpu.
    if project in ("mlip", "fairchem"):
        assert extras.pop("cu126") == extras.pop("cu130") == ["torch", "torchvision"]
    assert extras == load("cpu")["project"]["optional-dependencies"]
    assert set(extras) == {"openmm", "pymol", "docking", "void", "transport"}


def test_gpu_projects_lock_both_torch_cuda_builds():
    """cu130 for drivers >= 580 (and GB10's sm_121), cu126 for older drivers; the
    builds exclude each other, and both come from PyTorch's index on both arches
    (PyPI's aarch64 torch is not always a CUDA build)."""
    for project in ("mlip", "fairchem"):
        uv = load(project)["tool"]["uv"]
        # torchvision is compiled against one torch build: same index as torch.
        for package in ("torch", "torchvision"):
            assert uv["sources"][package] == [
                {"index": "pytorch-cu126", "extra": "cu126"},
                {"index": "pytorch-cu130", "extra": "cu130"},
            ]
        assert uv["conflicts"] == [[{"extra": "cu126"}, {"extra": "cu130"}]]
        lock = (VENV_DIR / project / "uv.lock").read_text()
        assert "+cu126" in lock and "+cu130" in lock


def test_versions_are_floors_not_pins():
    """Everything tracks its latest release; only these pins have a reason.

    mcp<2: the servers use the 1.x FastMCP API, which 2.x renamed and changed.
    pymol-open-source==3.2.0a0: the only release with Python 3.12 wheels.
    """
    allowed = {"mcp<2", "pymol-open-source==3.2.0a0"}
    for project in PROJECTS:
        data = load(project)
        specs = list(data["project"]["dependencies"])
        for extra in data["project"]["optional-dependencies"].values():
            specs += extra
        capped = [
            s
            for s in specs
            if any(op in s.split(";")[0] for op in ("==", "<", "~=", "!="))
            and s.split(";")[0].strip() not in allowed
        ]
        assert not capped, f"{project}: unexplained pins or caps {capped}"
        assert "override-dependencies" not in data["tool"]["uv"], project


@needs_uv
@pytest.mark.parametrize("project", PROJECTS)
def test_lock_matches_pyproject(project):
    result = subprocess.run(
        ["uv", "lock", "--check", "--project", str(VENV_DIR / project)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@needs_uv
def test_glibc_table_matches_the_locks():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "lock_platforms.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_base_sets_install_on_rhel8_era_x86():
    """The point of keeping OpenMM, PyMOL and Vina out of the base sets."""
    table = (VENV_DIR / "platforms.tsv").read_text().splitlines()
    rows = [ln.split("\t") for ln in table if ln and not ln.startswith("#")]
    for venv, extra, arch, glibc in rows:
        if extra == "-" and arch == "x86_64":
            assert tuple(map(int, glibc.split("."))) <= (
                2,
                28,
            ), f"{venv} needs glibc {glibc} on x86_64"
