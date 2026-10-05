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
# Research stacks with their own uv project each (the generative servers, ...):
# pinned to the environment they were verified in, rather than tracking the
# latest releases like the shared projects.
STACKS = tuple(
    sorted(
        p.parent.name
        for p in VENV_DIR.glob("*/pyproject.toml")
        if p.parent.name not in PROJECTS
    )
)
ALL = PROJECTS + STACKS
# Stacks resolved from upstream's own requirements (ms-pred at a pinned commit;
# SelfConditionedDenoisingAtoms' requirements.txt) rather than a verified closed
# set: their uv.lock is the pin.
LOCK_PINNED = ("msms", "scd")
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


@pytest.mark.parametrize("project", ALL)
def test_the_repository_is_an_editable_dependency(project):
    """Without it, `from src...` fails whenever the cwd is not the repo root."""
    data = load(project)
    assert "atomisticskills" in names(project)
    assert data["tool"]["uv"]["sources"]["atomisticskills"] == {
        "path": "../..",
        "editable": True,
    }


@pytest.mark.parametrize("project", ALL)
def test_both_architectures_are_required(project):
    """`environments` limits the resolver; only `required-environments` checks wheels.

    The shared projects cover both architectures. A research stack may be
    x86_64-only (its compiled dependencies have no aarch64 wheels); venv/run
    then uses its container image on aarch64."""
    uv = load(project)["tool"]["uv"]
    assert set(uv["environments"]) == set(uv["required-environments"])
    if project in PROJECTS:
        assert set(uv["environments"]) == ARCHES
    else:
        assert (
            "sys_platform == 'linux' and platform_machine == 'x86_64'"
            in uv["environments"]
        )


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
@pytest.mark.parametrize("project", ALL)
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
        if venv in PROJECTS and extra == "-" and arch == "x86_64":
            assert tuple(map(int, glibc.split("."))) <= (
                2,
                28,
            ), f"{venv} needs glibc {glibc} on x86_64"


@pytest.mark.parametrize("project", [s for s in STACKS if s not in LOCK_PINNED])
def test_research_stacks_reproduce_their_verified_environment(project):
    """A closed set like the verified `pip install --no-deps`, with the torch build
    still chosen per driver."""
    data = load(project)
    uv = data["tool"]["uv"]
    pins = [d for d in data["project"]["dependencies"] if d != "atomisticskills"]
    assert pins and all("==" in d for d in pins), project
    # Every pin is an override, so upstream constraints cannot move it...
    torchy = {"torch", "torchvision"}
    assert {p for p in pins if Requirement(p).name not in torchy} <= set(
        uv["override-dependencies"]
    ), project
    # ...but torch is not: an override would drop its per-extra index.
    overridden = {Requirement(o).name for o in uv["override-dependencies"]}
    assert not overridden & {"torch", "torchvision"}, project
    extras = data["project"].get("optional-dependencies", {})
    if not extras:
        return  # a single torch build (e.g. reactot: torch 2.2 has no CUDA 12.6/13)
    assert set(extras) == {"cu126", "cu130"} and extras["cu126"] == extras["cu130"]
    assert uv["conflicts"] == [[{"extra": "cu126"}, {"extra": "cu130"}]]
    # Each package of the torch build comes from the index matching its extra.
    for package in (Requirement(r).name for r in extras["cu126"]):
        assert [src["extra"] for src in uv["sources"][package]] == ["cu126", "cu130"], (
            project,
            package,
        )


@pytest.mark.parametrize("project", STACKS)
def test_research_stack_builds_install_on_rhel8_era_x86(project):
    table = (VENV_DIR / "platforms.tsv").read_text().splitlines()
    rows = [ln.split("\t") for ln in table if ln and not ln.startswith("#")]
    has_builds = bool(load(project)["project"].get("optional-dependencies"))
    wanted = ("cu126", "cu130") if has_builds else ("-",)
    floors = {
        e: g for v, e, a, g in rows if v == project and a == "x86_64" and e in wanted
    }
    assert set(floors) == set(wanted), f"{project} missing from venv/platforms.tsv"
    for extra, glibc in floors.items():
        assert not glibc.startswith("unavailable"), (project, extra, glibc)
        assert tuple(map(int, glibc.split("."))) <= (2, 28), (project, extra, glibc)
