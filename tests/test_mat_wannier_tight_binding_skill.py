"""Scientific and CLI regressions for the Wannier skill, independent of ignored outputs.

Run with venv/run cpu python -m pytest tests/test_mat_wannier_tight_binding_skill.py.
Set WANNIER90_TEST_BINARY to a working executable for optional native integration.
"""

from __future__ import annotations

import json
import hashlib
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/mat-wannier-tight-binding"
SCRIPTS = SKILL / "scripts"
SI = SKILL / "examples/silicon-sp3"
GAAS = SKILL / "examples/gaas-valence"
sys.path.insert(0, str(SCRIPTS))

from generate_kmesh import uniform_mesh
from parse_hr import parse_hamiltonian
from parse_wout import parse_wout
from wannier_tb.hamiltonian import interpolate
from wannier_tb.io import (
    read_bands,
    read_cell,
    read_hr,
    read_kpoints,
    read_wsvec,
    resolve_wsvec,
    sha256,
)


def cli(script: str, *args, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Exercise the public CLI in the same interpreter, without assuming cwd."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def interpolation_args(out: Path) -> list:
    """Return the documented native-reference regression arguments."""
    return [
        SI / "silicon_hr.dat",
        "--kpoints",
        SI / "silicon_band.kpt",
        "--ref-bands",
        SI / "silicon_band.dat",
        "--max-error",
        "5e-5",
        "--output-dir",
        out,
    ]


@pytest.mark.parametrize(
    "folder,seed,nw,total,dis",
    [(GAAS, "gaas", 4, 4.466881009, False), (SI, "silicon", 8, 14.499574503, True)],
)
def test_native_final_state(folder, seed, nw, total, dis):
    result = parse_wout((folder / f"{seed}.wout").read_text())
    assert result["num_wann"] == nw
    assert result["omega_total"] == pytest.approx(total, abs=1e-8)
    assert result["localization_status"] == "converged"
    assert result["has_disentanglement"] is dis
    assert result["disentanglement_status"] == (
        "converged" if dis else "not_applicable"
    )
    assert np.sum(result["wf_spreads"]) == pytest.approx(total, abs=1e-7)


@pytest.mark.parametrize(
    "corruption", ["empty", "truncated", "missing_orbital", "inconsistent_total"]
)
def test_invalid_final_state_is_rejected(corruption):
    text = (GAAS / "gaas.wout").read_text()
    if corruption == "empty":
        text = ""
    elif corruption == "truncated":
        text = text.split("All done:")[0]
    elif corruption == "missing_orbital":
        text = text[: text.rfind("Final State")] + re.sub(
            r"^.*WF centre and spread\s+4.*\n",
            "",
            text[text.rfind("Final State") :],
            flags=re.M,
        )
    else:
        text = re.sub(r"Omega Total\s*=\s*\S+", "Omega Total = 100.0", text)
    with pytest.raises(ValueError):
        parse_wout(text)


def test_convergence_is_not_inferred_from_small_spread(tmp_path):
    source = tmp_path / "fixed.wout"
    source.write_text(
        (GAAS / "gaas.wout")
        .read_text()
        .replace("Wannierisation convergence criteria satisfied", "Fixed iteration run")
    )
    assert parse_wout(source.read_text())["localization_status"] != "converged"
    run = cli("parse_wout.py", source, "--require-converged", "--output-dir", tmp_path)
    assert run.returncode != 0
    assert (
        json.loads((tmp_path / "wout_summary.json").read_text())["validation"]["status"]
        == "FAILED"
    )


def test_empty_wout_cli_cannot_pass(tmp_path):
    source = tmp_path / "empty.wout"
    source.touch()
    run = cli(
        "parse_wout.py", source, "--max-omega-tot", "20", "--output-dir", tmp_path
    )
    assert run.returncode != 0
    assert "PASSED" not in run.stdout


def test_invalid_rerun_replaces_stale_success(tmp_path):
    source = tmp_path / "run.wout"
    shutil.copyfile(GAAS / "gaas.wout", source)
    assert cli("parse_wout.py", source, "--output-dir", tmp_path).returncode == 0
    source.write_text("")
    assert cli("parse_wout.py", source, "--output-dir", tmp_path).returncode != 0
    assert (
        json.loads((tmp_path / "wout_summary.json").read_text())["validation"]["status"]
        == "FAILED"
    )
    assert cli("interpolate_bands.py", *interpolation_args(tmp_path)).returncode == 0
    args = interpolation_args(tmp_path)
    args[args.index("--ref-bands") + 1] = tmp_path / "absent.dat"
    assert cli("interpolate_bands.py", *args).returncode != 0
    assert (
        json.loads((tmp_path / "band_comparison.json").read_text())["status"]
        == "FAILED"
    )


def test_modern_interpolation_matches_native_and_needs_ws_shifts():
    nw, nr, deg, r, h = read_hr(SI / "silicon_hr.dat")
    enabled, shifts = read_wsvec(SI / "silicon_wsvec.dat", r, nw)
    assert enabled and (nw, nr) == (8, 93)
    k = read_kpoints(SI / "silicon_band.kpt")
    _, reference = read_bands(SI / "silicon_band.dat")
    calculated, residual = interpolate(r, h, deg, k, shifts)
    assert calculated.shape == (380, 8)
    assert residual < 1e-5
    assert np.max(np.abs(calculated - reference)) < 5e-5
    unshifted, _ = interpolate(r, h, deg, k)
    assert np.max(np.abs(unshifted - reference)) > 1e-2
    _, old_reference = read_bands(SI / "silicon_legacy_band.dat")
    assert np.max(np.abs(unshifted - old_reference)) < 5e-5


def test_analytic_complex_chain_and_degeneracies():
    r = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    h = np.array([[[2 - 2j]], [[3]], [[2 + 2j]]])
    k = np.column_stack((np.linspace(0, 1, 13), np.zeros((13, 2))))
    energies, _ = interpolate(r, h, np.array([2, 1, 2]), k)
    phase = 2 * np.pi * k[:, 0]
    assert np.allclose(energies[:, 0], 3 + 2 * np.cos(phase) - 2 * np.sin(phase))


def test_ws_translation_multiplicity_normalization():
    r = np.array([[0, 0, 0]])
    h = np.array([[[2.0]]])
    shifts = {(0, 0, 0, 0, 0): np.array([[-1, 0, 0], [1, 0, 0]])}
    k = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
    energies, _ = interpolate(r, h, np.ones(1), k, shifts)
    assert np.allclose(energies[:, 0], [2, 0, -2])


def test_nonhermitian_model_is_not_silently_repaired():
    with pytest.raises(ValueError, match="not Hermitian"):
        interpolate(
            np.zeros((1, 3)), np.array([[[1 + 0.1j]]]), np.ones(1), np.zeros((1, 3))
        )


@pytest.mark.parametrize(
    "body",
    [
        "comment\n1\n1\n0\n0 0 0 1 1 1.0 0.0\n",
        "comment\n1\n1\n1\n0 0 0 2 1 1.0 0.0\n",
        "comment\n1\n1\n1\n0 0 0 1 1 nan 0.0\n",
        "comment\n1\n2\n1 1\n0 0 0 1 1 1.0 0.0\n0 0 0 1 1 1.0 0.0\n",
        "comment\n1\n1\n1\n",
    ],
)
def test_malformed_hr_rejected(tmp_path, body):
    source = tmp_path / "bad_hr.dat"
    source.write_text(body)
    with pytest.raises(ValueError):
        read_hr(source)


def test_missing_or_incomplete_sidecar_rejected(tmp_path):
    source = tmp_path / "copied_hr.dat"
    shutil.copyfile(SI / "silicon_hr.dat", source)
    nw, _, _, r, _ = read_hr(source)
    with pytest.raises(FileNotFoundError):
        resolve_wsvec(source, r, nw)
    assert resolve_wsvec(source, r, nw, legacy=True)[0] is None
    with pytest.raises(ValueError, match="conflicts"):
        resolve_wsvec(SI / "silicon_hr.dat", r, nw, legacy=True)
    incomplete = tmp_path / "copied_wsvec.dat"
    incomplete.write_text(
        "\n".join((SI / "silicon_wsvec.dat").read_text().splitlines()[:-1])
    )
    with pytest.raises(ValueError):
        read_wsvec(incomplete, r, nw)


def test_physical_hopping_distances_use_cell_and_centres():
    centres = np.asarray(parse_wout((SI / "silicon.wout").read_text())["wf_centres"])
    cell = read_cell(SI / "silicon.win")
    summary, hops = parse_hamiltonian(
        SI / "silicon_hr.dat", cell_matrix=cell, centres=centres
    )
    assert summary["distance_kind"] == "orbital_pair"
    row = next(
        row
        for row in hops
        if (row["Rx"], row["Ry"], row["Rz"], row["orb_m"], row["orb_n"])
        == (0, 0, 0, 1, 2)
    )
    assert row["distance_angstrom"] == pytest.approx(
        np.linalg.norm(centres[1] - centres[0]), abs=1e-6
    )
    no_cell, hops = parse_hamiltonian(SI / "silicon_hr.dat", threshold=100)
    assert no_cell["distance_kind"] == "not_computed"
    assert no_cell["decay_profile"] == [] and hops == []
    assert no_cell["max_hopping_intercell_eV"] > 1  # Independent of display threshold.


def test_cli_regression_and_config_persistence_from_empty_directory(tmp_path):
    run = cli("interpolate_bands.py", *interpolation_args(tmp_path), cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    run = cli(
        "parse_wout.py",
        SI / "silicon.wout",
        "--require-converged",
        "--output-dir",
        tmp_path,
        cwd=tmp_path,
    )
    assert run.returncode == 0, run.stderr
    data = yaml.safe_load((tmp_path / "input_configs.yaml").read_text())["stages"]
    assert set(data) == {"interpolate_bands", "parse_wout"}
    assert data["interpolate_bands"]["hermiticity_tol"] == 1e-5
    assert data["interpolate_bands"]["ref_energy_shift"] == 0.0
    assert data["parse_wout"]["require_converged"] is True
    report = json.loads((tmp_path / "band_comparison.json").read_text())
    assert report["status"] == "PASSED" and report["reference_kind"] == "wannier90"
    assert read_bands(tmp_path / "tb_interpolated_bands.dat")[1].shape == (380, 8)


@pytest.mark.parametrize(
    "issue", ["missing_reference", "wrong_points", "wrong_shape", "wrong_energy"]
)
def test_invalid_reference_never_passes(tmp_path, issue):
    arguments = interpolation_args(tmp_path)
    if issue == "missing_reference":
        arguments[arguments.index("--ref-bands") + 1] = tmp_path / "missing.dat"
    elif issue == "wrong_points":
        k = read_kpoints(SI / "silicon_band.kpt")[::-1]
        np.savetxt(tmp_path / "wrong.kpt", k)
        arguments += ["--ref-kpoints", tmp_path / "wrong.kpt"]
    elif issue == "wrong_shape":
        (tmp_path / "bad.dat").write_text("0 1\n")
        arguments[arguments.index("--ref-bands") + 1] = tmp_path / "bad.dat"
        arguments += ["--ref-kpoints", SI / "silicon_band.kpt"]
    else:
        arguments += ["--ref-energy-shift", "1"]
    run = cli("interpolate_bands.py", *arguments)
    assert run.returncode != 0
    if (tmp_path / "band_comparison.json").exists():
        assert (
            json.loads((tmp_path / "band_comparison.json").read_text())["status"]
            != "PASSED"
        )


def test_energy_window_and_explicit_dft_reference_label(tmp_path):
    run = cli(
        "interpolate_bands.py",
        *interpolation_args(tmp_path),
        "--reference-kind",
        "dft",
        "--energy-window",
        "-6",
        "6.4",
    )
    assert run.returncode == 0, run.stderr
    report = json.loads((tmp_path / "band_comparison.json").read_text())
    assert report["reference_kind"] == "dft"
    assert 0 < report["compared_eigenvalues"] < 380 * 8


def test_qe_and_wannier_templates_match_exactly():
    folder = SKILL / "resources/templates"
    nscf = read(folder / "template_nscf.in", format="espresso-in")
    scf = read(folder / "template_scf.in", format="espresso-in")
    cell = read_cell(folder / "template.win")
    assert np.allclose(nscf.cell.array, cell, atol=1e-10)
    assert np.allclose(scf.cell.array, cell, atol=1e-10)
    assert np.allclose(nscf.get_scaled_positions(), [[0, 0, 0], [0.25, 0.25, 0.25]])
    wtext = (folder / "template.win").read_text()
    block = re.search(r"begin kpoints\n(.*?)end kpoints", wtext, re.S)[1]
    wpoints = np.array(
        [list(map(float, line.split())) for line in block.strip().splitlines()]
    )
    qblock = (
        (folder / "template_nscf.in")
        .read_text()
        .split("K_POINTS crystal\n")[1]
        .splitlines()
    )
    assert int(qblock[0]) == 64
    qpoints = np.array([list(map(float, line.split())) for line in qblock[1:]])
    assert np.allclose(wpoints, qpoints[:, :3], atol=1e-12)
    assert np.sum(qpoints[:, 3]) == pytest.approx(1)
    assert len(np.unique(wpoints, axis=0)) == 64
    shifted = uniform_mesh((2, 3, 4), (0.5, 0, 0.5))
    assert shifted.shape == (24, 3) and not np.any(np.all(shifted == 0, axis=1))


@pytest.mark.parametrize("folder", [GAAS, SI])
def test_upstream_inputs_match_recorded_provenance(folder):
    provenance = json.loads((folder / "provenance.json").read_text())
    for name, info in provenance["unmodified_upstream_inputs"].items():
        source = folder / info.get("bundled_file", name)
        if info.get("compression") == "xz":
            assert sha256(source) == info["bundled_sha256"]
            with lzma.open(source, "rb") as stream:
                assert hashlib.sha256(stream.read()).hexdigest() == info["sha256"]
        else:
            assert sha256(source) == info["sha256"]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("WANNIER90_TEST_BINARY"),
    reason="Set WANNIER90_TEST_BINARY for native integration",
)
@pytest.mark.parametrize("example", ["gaas-valence", "silicon-sp3"])
def test_native_example_from_scratch(tmp_path, example):
    run = cli(
        "run_example.py",
        example,
        "--wannier90",
        os.environ["WANNIER90_TEST_BINARY"],
        "--output-dir",
        tmp_path / example,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((tmp_path / example / "execution.json").read_text())
    assert report["status"] == "PASSED" and report["dft_executed"] is False


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("WANNIER90_TEST_BINARY"),
    reason="Set WANNIER90_TEST_BINARY for native integration",
)
def test_relative_binary_and_native_template_preprocessing(tmp_path):
    binary = os.path.relpath(os.environ["WANNIER90_TEST_BINARY"], ROOT)
    run = cli(
        "run_example.py",
        "gaas-valence",
        "--wannier90",
        binary,
        "--output-dir",
        tmp_path / "relative-run",
        cwd=ROOT,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    shutil.copyfile(
        SKILL / "resources/templates/template.win", tmp_path / "template.win"
    )
    run = subprocess.run(
        [os.environ["WANNIER90_TEST_BINARY"], "-pp", "template"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / "template.nnkp").exists()
