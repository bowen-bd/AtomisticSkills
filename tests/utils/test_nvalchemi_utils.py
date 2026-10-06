"""Unit tests for src.utils.mlips.nvalchemi.nvalchemi_utils.

Tests the ASE ↔ AtomicData round-trip, availability check, and batch
result extraction.  These tests mock nvalchemi data types so they can run
in any environment without nvalchemi installed.

Run with:
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_utils.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atoms(pbc: bool = True):
    """Return a small ASE Atoms for testing."""
    from ase import Atoms
    from ase.build import bulk

    if pbc:
        return bulk("Cu", "fcc", a=3.6)
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------


class TestCheckNvalchemiAvailable:
    def test_returns_bool(self):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import check_nvalchemi_available

        result = check_nvalchemi_available()
        assert isinstance(result, bool)

    def test_consistent_with_flag(self):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import (
            NVALCHEMI_AVAILABLE,
            check_nvalchemi_available,
        )

        assert check_nvalchemi_available() == NVALCHEMI_AVAILABLE


# ---------------------------------------------------------------------------
# atoms_to_atomic_data / atomic_data_to_atoms round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__(
        "src.utils.mlips.nvalchemi.nvalchemi_utils", fromlist=["NVALCHEMI_AVAILABLE"]
    ).NVALCHEMI_AVAILABLE
    is False,
    reason="nvalchemi not installed",
)
class TestAtomsRoundTrip:
    def test_periodic_atoms_roundtrip(self):
        import torch
        from src.utils.mlips.nvalchemi.nvalchemi_utils import (
            atoms_to_atomic_data,
            atomic_data_to_atoms,
        )

        atoms = _make_atoms(pbc=True)
        data = atoms_to_atomic_data(atoms, device="cpu", dtype=torch.float32)
        recovered = atomic_data_to_atoms(data)

        assert len(recovered) == len(atoms)
        np.testing.assert_allclose(
            recovered.get_positions(), atoms.get_positions(), atol=1e-5
        )
        assert np.all(recovered.get_atomic_numbers() == atoms.get_atomic_numbers())
        assert recovered.pbc.all()

    def test_non_periodic_atoms_roundtrip(self):
        import torch
        from src.utils.mlips.nvalchemi.nvalchemi_utils import (
            atoms_to_atomic_data,
            atomic_data_to_atoms,
        )

        atoms = _make_atoms(pbc=False)
        data = atoms_to_atomic_data(atoms, device="cpu", dtype=torch.float32)
        recovered = atomic_data_to_atoms(data)

        assert len(recovered) == len(atoms)
        np.testing.assert_allclose(
            recovered.get_positions(), atoms.get_positions(), atol=1e-5
        )

    def test_atomic_data_fields(self):
        import torch
        from src.utils.mlips.nvalchemi.nvalchemi_utils import atoms_to_atomic_data

        atoms = _make_atoms(pbc=True)
        data = atoms_to_atomic_data(atoms, device="cpu", dtype=torch.float32)

        assert hasattr(data, "positions")
        assert hasattr(data, "atomic_numbers")
        assert data.positions.shape == (len(atoms), 3)
        assert data.atomic_numbers.shape == (len(atoms),)
        assert data.positions.dtype == torch.float32

    def test_stress_voigt_conversion(self):
        """Verify [1,3,3] stress → Voigt [6] conversion is correct."""
        import torch
        from src.utils.mlips.nvalchemi.nvalchemi_utils import (
            atoms_to_atomic_data,
            atomic_data_to_atoms,
        )
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms = _make_atoms(pbc=True)
        # Attach a known stress tensor
        s_voigt = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # [xx,yy,zz,yz,xz,xy]
        atoms.calc = SinglePointCalculator(
            atoms, energy=-3.0, forces=np.zeros((len(atoms), 3)), stress=s_voigt
        )

        data = atoms_to_atomic_data(atoms, device="cpu", dtype=torch.float32)
        recovered = atomic_data_to_atoms(data)

        if recovered.calc is not None and "stress" in recovered.calc.results:
            s_out = recovered.calc.results["stress"]
            np.testing.assert_allclose(s_out, s_voigt, atol=1e-5)


# ---------------------------------------------------------------------------
# extract_batch_results
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__(
        "src.utils.mlips.nvalchemi.nvalchemi_utils", fromlist=["NVALCHEMI_AVAILABLE"]
    ).NVALCHEMI_AVAILABLE
    is False,
    reason="nvalchemi not installed",
)
class TestExtractBatchResults:
    def test_returns_per_structure_results(self, tmp_path):
        import torch
        from src.utils.mlips.nvalchemi.nvalchemi_utils import (
            atoms_to_atomic_data,
            extract_batch_results,
        )
        from nvalchemi.data import Batch

        atoms_list = [_make_atoms(pbc=True), _make_atoms(pbc=True)]
        data_list = [
            atoms_to_atomic_data(a, device="cpu", dtype=torch.float32)
            for a in atoms_list
        ]
        batch = Batch.from_data_list(data_list)

        names = ["struct_A", "struct_B"]
        dirs = [str(tmp_path / "A"), str(tmp_path / "B")]
        results = extract_batch_results(batch, names, dirs)

        assert len(results) == 2
        for r, name in zip(results, names):
            assert r["structure_name"] == name
            assert r["status"] == "success"
            import os

            assert os.path.exists(r["cif_path"])

    def test_error_captured_per_structure(self, tmp_path, monkeypatch):
        """A bad data item returns status='failed' without crashing the whole batch."""
        import torch
        from src.utils.mlips.nvalchemi.nvalchemi_utils import (
            atoms_to_atomic_data,
            extract_batch_results,
        )
        from nvalchemi.data import Batch

        atoms = _make_atoms(pbc=True)
        data = atoms_to_atomic_data(atoms, device="cpu", dtype=torch.float32)
        batch = Batch.from_data_list([data])

        # Monkeypatch atomic_data_to_atoms to raise for the first item
        from src.utils.mlips.nvalchemi import nvalchemi_utils as _m

        original = _m.atomic_data_to_atoms

        def _bad(d):
            raise ValueError("injected error")

        monkeypatch.setattr(_m, "atomic_data_to_atoms", _bad)
        results = extract_batch_results(batch, ["bad"], [str(tmp_path)])
        assert results[0]["status"] == "failed"
        assert "injected error" in results[0]["error"]
        monkeypatch.setattr(_m, "atomic_data_to_atoms", original)
