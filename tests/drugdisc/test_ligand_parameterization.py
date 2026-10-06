"""Ligand parameterization runs, not just parses its arguments.

Builds the drug-complex-system-builder example (HIV-1 protease and a docked
ligand) with OpenFF Sage and RDKit charges, which need nothing outside the
openmm extra. Run in: venv/run cpu+openmm python -m pytest tests/drugdisc
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.drugdisc
pytest.importorskip("openff.toolkit", reason="needs the openmm extra (cpu+openmm)")

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "skills" / "drug-complex-system-builder" / "examples" / "hiv1-protease"
SCRIPT = (
    ROOT / "skills" / "drug-complex-system-builder" / "scripts" / "build_complex.py"
)


def build(tmp_path, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--receptor",
            str(EXAMPLE / "1HSG_prepared.pdb"),
            "--ligand",
            str(EXAMPLE / "ligand.sdf"),
            "--output_dir",
            str(tmp_path / "system"),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_builds_a_solvated_complex_with_rdkit_charges(tmp_path):
    result = build(tmp_path, "--charge_method", "gasteiger", "--box_padding", "8")
    assert result.returncode == 0, result.stdout + result.stderr
    provenance = json.loads((tmp_path / "system" / "build_provenance.json").read_text())
    assert provenance["n_ligand_atoms"] == 18 and provenance["n_protein_atoms"] == 3509
    assert provenance["n_total_atoms"] > 30000
    assert (tmp_path / "system" / "system.xml").stat().st_size > 1_000_000


@pytest.mark.skipif(shutil.which("sqm") is not None, reason="AmberTools is installed")
def test_am1bcc_without_ambertools_says_so(tmp_path):
    result = build(tmp_path)
    assert result.returncode != 0
    assert "AM1-BCC charges need AmberTools (sqm) on PATH" in result.stderr
