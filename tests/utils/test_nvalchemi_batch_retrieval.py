"""Unit and integration tests for retrieving ASE simulation results from batched NValchemi runs.

Verifies that trajectory databases (.traj), step-by-step logs (.log), and final structures
are correctly reconstructed and saved with proper physical properties (velocities, energies, forces).
"""

from __future__ import annotations

import os
import pytest
import numpy as np
from ase.io import read


def _make_cu_bulk(scale: float = 1.0):
    from ase.build import bulk

    # Use repeat(2) to generate 8 atoms so that center-of-mass subtraction
    # does not zero out velocities.
    return bulk("Cu", "fcc", a=3.6 * scale).repeat(2)


def _make_structures(n: int = 2):
    scales = [1.0, 0.98, 1.02][:n]
    return [_make_cu_bulk(s) for s in scales]


@pytest.mark.mace
class TestBatchRetrieval:
    def test_batch_relaxation_retrieval(self, tmp_path):
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        structure_names = ["structure_0", "structure_1"]

        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.1,
            steps=5,
            output_dir=str(tmp_path),
        )

        assert result["status"] == "success" if "status" in result else True
        assert result["mode"] == "batch"
        assert result["successful"] == 0
        assert result["not_converged"] == 2
        assert result["failed"] == 0

        for r in result["results"]:
            name = r["structure_name"]
            out_dir = r["output_dir"]
            assert name in structure_names
            assert r["status"] == "not_converged"
            assert r["converged"] is False

            # Check files existence
            assert os.path.exists(r["trajectory_path"])
            assert os.path.exists(r["log_path"])
            assert os.path.exists(r["cif_path"])
            assert os.path.exists(os.path.join(out_dir, "relaxed_energy.txt"))

            # Verify trajectory file content
            traj_atoms = read(r["trajectory_path"], index=":")
            assert len(traj_atoms) > 0

            # Verify that single-point properties exist
            for atoms in traj_atoms:
                assert atoms.calc is not None
                assert "energy" in atoms.calc.results
                assert "forces" in atoms.calc.results
                # Forces shape should match number of atoms
                assert atoms.calc.results["forces"].shape == (len(atoms), 3)

            # Verify log file format
            with open(r["log_path"]) as f:
                lines = f.readlines()
                assert len(lines) > 1
                assert "Step" in lines[0]
                assert "Energy" in lines[0]
                assert "fmax" in lines[0]
                for line in lines[1:]:
                    parts = line.split()
                    assert len(parts) >= 3
                    # Step is integer
                    int(parts[1])
                    # Energy and fmax are float
                    float(parts[2])
                    float(parts[3])

    def test_batch_md_retrieval(self, tmp_path):
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        structure_names = ["structure_0", "structure_1"]

        # Use NVT Nose-Hoover to verify thermostat behavior
        result = wrapper.run_md(
            structure_data=structures,
            temperature=300,
            steps=10,
            timestep=1.0,
            ensemble="nvt_nose_hoover",
            output_dir=str(tmp_path),
            log_interval=2,
        )

        assert result["mode"] == "batch"
        assert result["successful"] == 2

        for r in result["results"]:
            name = r["structure_name"]
            out_dir = r["output_dir"]
            assert name in structure_names
            assert r["status"] == "success"

            # Check files existence
            assert os.path.exists(r["trajectory_path"])
            assert os.path.exists(r["log_path"])
            assert os.path.exists(r["cif_path"])
            assert os.path.exists(os.path.join(out_dir, "energy.txt"))

            # Verify trajectory file content
            traj_atoms = read(r["trajectory_path"], index=":")
            # We ran 10 steps with log_interval=2, starting at step 0:
            # step 0, 2, 4, 6, 8, 10 -> should be 6 frames
            assert len(traj_atoms) == 6

            # Verify single-point properties, especially velocities
            for atoms in traj_atoms:
                assert atoms.calc is not None
                assert "energy" in atoms.calc.results
                assert "forces" in atoms.calc.results
                # Verify velocities are preserved in traj
                velocities = atoms.get_velocities()
                assert velocities is not None
                assert velocities.shape == (len(atoms), 3)
                # Verify velocities are not collapsed to exactly zero
                assert np.linalg.norm(velocities) > 0.0

            # Verify log file format
            with open(r["log_path"]) as f:
                lines = f.readlines()
                assert len(lines) > 1
                assert "Time[ps]" in lines[0]
                assert "Etot[eV]" in lines[0]
                assert "Epot[eV]" in lines[0]
                assert "Ekin[eV]" in lines[0]
                assert "T[K]" in lines[0]
                for line in lines[1:]:
                    parts = line.split()
                    assert len(parts) >= 5
                    # Time, Etot, Epot, Ekin, Temp are floats
                    for p in parts[:5]:
                        float(p)
                    # Temperature should be around 300K, definitely not 0K
                    temp = float(parts[4])
                    assert temp > 50.0

    def test_batch_relaxation_no_extraction(self, tmp_path):
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        structure_names = ["structure_0", "structure_1"]

        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.1,
            steps=5,
            output_dir=str(tmp_path),
            extract_batch_results=False,
        )

        assert result["status"] == "success" if "status" in result else True
        assert result["mode"] == "batch"
        assert result["successful"] == 0
        assert result["not_converged"] == 2
        assert result["failed"] == 0

        for r in result["results"]:
            name = r["structure_name"]
            out_dir = r["output_dir"]
            assert name in structure_names
            assert r["status"] == "not_converged"
            assert r["converged"] is False

            # Check that traj and log paths are not in the result, and files do not exist
            assert "trajectory_path" not in r
            assert "log_path" not in r
            assert not os.path.exists(os.path.join(out_dir, "relax.traj"))
            assert not os.path.exists(os.path.join(out_dir, "relax.log"))

            # Check that final structures and energies are still saved
            assert os.path.exists(r["cif_path"])
            assert os.path.exists(os.path.join(out_dir, "relaxed_energy.txt"))

            # Read relaxed energy
            with open(os.path.join(out_dir, "relaxed_energy.txt")) as f:
                energy_val = float(f.read().strip())
                assert energy_val < 0.0

    def test_batch_md_no_extraction(self, tmp_path):
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        structure_names = ["structure_0", "structure_1"]

        result = wrapper.run_md(
            structure_data=structures,
            temperature=300,
            steps=10,
            timestep=1.0,
            ensemble="nvt_nose_hoover",
            output_dir=str(tmp_path),
            log_interval=2,
            extract_batch_results=False,
        )

        assert result["mode"] == "batch"
        assert result["successful"] == 2

        for r in result["results"]:
            name = r["structure_name"]
            out_dir = r["output_dir"]
            assert name in structure_names
            assert r["status"] == "success"

            # Check that traj and log paths are not in the result, and files do not exist
            assert "trajectory_path" not in r
            assert "log_path" not in r

            # Find and assert no .traj or .log exists
            for f in os.listdir(out_dir):
                assert not f.endswith(".traj")
                assert not f.endswith(".log")

            # Check that final structures and energies are still saved
            assert os.path.exists(r["cif_path"])
            assert os.path.exists(os.path.join(out_dir, "energy.txt"))

            # Read final energy
            with open(os.path.join(out_dir, "energy.txt")) as f:
                energy_val = float(f.read().strip())
                assert energy_val < 0.0
