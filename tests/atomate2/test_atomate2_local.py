import sys

import pytest
import shutil
from pathlib import Path
from pymatgen.core import Structure, Lattice

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.dft.atomate2_utils import Atomate2Handler


def test_atomate2_local_si_scf(tmp_path):
    """
    Test a local Atomate2 VASP calculation for Silicon with 3 SCF steps.
    Runs VASP, so it needs a VASP binary and POTCARs on this machine.
    """
    # 1. Setup test directory
    test_dir = tmp_path / "atomate2_si"
    test_dir.mkdir(parents=True)

    try:
        # 2. Create Silicon structure (diamond)
        lattice = Lattice.cubic(5.43)
        structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        structure_path = test_dir / "Si.cif"
        structure.to(filename=str(structure_path))

        # 3. Initialize handler
        handler = Atomate2Handler(str(test_dir))

        # 4. Check environment: skip where VASP is not installed (CI)
        env = handler.check_environment()
        assert env["atomate2"] is True
        if not (env["vasp"] and env["potcar"]):
            pytest.skip("needs a VASP binary and POTCARs")

        # 5. Define custom config for 3 SCF steps
        config = {
            "NELM": 3,
            "NSW": 0,  # Static calculation
            "EDIFF": 1e-4,
        }

        # 6. Create flow maker
        maker = handler.get_flow_maker(
            preset_type="mp", calculation_type="static", config=config
        )

        # 7. Run locally
        from jobflow import run_locally

        flow = maker.make(structure)
        # We set ensure_success=False because 3 SCF steps will likely not converge,
        # and we want to verify the run completed regardless of convergence.
        responses = run_locally(
            flow,
            create_folders=True,
            ensure_success=False,
            root_dir=str(test_dir / "run"),
        )

        # 8. Verify outputs
        # Find vasprun.xml in run directory
        vasprun_files = list((test_dir / "run").rglob("vasprun.xml"))
        assert len(vasprun_files) > 0, "vasprun.xml should have been created"

        # 9. Verify extraction (idiomatic parsing)
        # We need to create the responses.pkl for the handler to find it
        job_id = "test_si_job"
        all_responses = {
            "test_flow_uuid": {"responses": responses, "dir": str(test_dir / "run")}
        }
        import pickle

        with open(test_dir / f"{job_id}_responses.pkl", "wb") as f:
            pickle.dump(all_responses, f)

        results = handler.extract_results(job_id)
        assert "results" in results
        assert len(results["results"]) > 0
        print(f"Extracted energy: {results['results'][0]['energy']}")
        assert results["results"][0]["energy"] is not None

    finally:
        # Cleanup temporary test directory
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    # Allows running standalone
    test_atomate2_local_si_scf()
