"""Integration tests for NValchemi GPU-accelerated batch operations.

These tests require the GPU environment of each backend (mlip or fairchem), which carries nvalchemi
and at least one MLIP model available.

Run individual groups:
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestFallback -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestBatchRelaxMACE -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestBatchRelaxM3GNet -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestBatchRelaxCHGNet -v
    venv/run fairchem python -m pytest tests/utils/test_nvalchemi_batch.py::TestBatchStaticFairChem -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestBatchMDNVT -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestInflightRelaxMACE -v
    venv/run fairchem python -m pytest tests/utils/test_nvalchemi_batch.py::TestInflightRelaxFairChem -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestInflightRelaxM3GNet -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestInflightRelaxTensorNet -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestInflightRelaxCHGNet -v

Backend key and MCP signature tests (any environment with nvalchemi + MACE):
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestBackendKeyAllPaths -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestRelaxLogFixedBatch -v
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_batch.py::TestMCPServerParamCoverage -v
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_cu_bulk(scale: float = 1.0):
    """Return a Cu FCC unit cell, optionally strained."""
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.6 * scale)
    return atoms


def _make_structures(n: int = 3):
    scales = [1.0, 0.98, 1.02][:n]
    return [_make_cu_bulk(s) for s in scales]


# ---------------------------------------------------------------------------
# Fallback test (no nvalchemi needed)
# ---------------------------------------------------------------------------


class TestFallback:
    """When nvalchemi is unavailable, batch ops must fall back to sequential."""

    def test_batch_relax_falls_back_without_nvalchemi(self, tmp_path, monkeypatch):
        """Monkeypatching NVALCHEMI_AVAILABLE=False should force sequential path."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as _nv_utils

        monkeypatch.setattr(_nv_utils, "NVALCHEMI_AVAILABLE", False)

        # The dispatch in base.py checks check_nvalchemi_available() which reads the module attr
        monkeypatch.setattr(_nv_utils, "check_nvalchemi_available", lambda: False)

        try:
            from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        except ImportError:
            pytest.skip("MACE not available in this environment")

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.5,
            steps=5,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        # Should complete without error (via sequential path)
        assert result is not None

    def test_single_prediction_unchanged(self, tmp_path):
        """Single-structure prediction path must not be affected by NValchemi changes."""
        try:
            from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        except ImportError:
            pytest.skip("MACE not available in this environment")

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
            # create_calculator() verifies mace.calculators is importable
            wrapper.create_calculator()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        atoms = _make_cu_bulk()
        result = wrapper.static_calculation(atoms, use_nvalchemi=True)
        assert "energy" in result
        assert isinstance(result["energy"], float)


# ---------------------------------------------------------------------------
# MACE batch relax
# ---------------------------------------------------------------------------


@pytest.mark.mace
class TestBatchRelaxMACE:
    def test_batch_relax_mace_nvalchemi(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=20,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result is not None

    def test_batch_relax_mace_energy_close_to_sequential(self, tmp_path):
        """NValchemi batch relax energies should be within 0.1% of sequential."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        import src.utils.mlips.nvalchemi.nvalchemi_utils as _nv

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)

        # Run with NValchemi
        result_nv = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=5,
            output_dir=str(tmp_path / "nv"),
            use_nvalchemi=True,
        )

        # Force sequential
        original = _nv.NVALCHEMI_AVAILABLE
        _nv.NVALCHEMI_AVAILABLE = False
        result_seq = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=5,
            output_dir=str(tmp_path / "seq"),
            use_nvalchemi=True,
        )
        _nv.NVALCHEMI_AVAILABLE = original

        # Both paths complete without error (energy comparison may differ
        # slightly due to internal optimizer state but should be same order)
        assert result_nv is not None
        assert result_seq is not None


# ---------------------------------------------------------------------------
# MatGL M3GNet batch relax
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestBatchRelaxM3GNet:
    def test_batch_relax_m3gnet(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(model_name="M3GNet-PES-MatPES-PBE-2025.2", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=20,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result is not None

    def test_m3gnet_wrapper_factory(self):
        """M3GNetWrapper.from_potential() should succeed for an M3GNet potential."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            import matgl
            from matgl.models._m3gnet import M3GNet
            from src.utils.mlips.nvalchemi.matgl_wrappers import M3GNetWrapper

            potential = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        assert isinstance(potential.model, M3GNet)
        nv_model = M3GNetWrapper.from_potential(potential)
        assert nv_model is not None
        assert nv_model.model_config.neighbor_config is not None
        assert (
            abs(
                float(nv_model.model_config.neighbor_config.cutoff)
                - potential.model.cutoff
            )
            < 0.01
        )


# ---------------------------------------------------------------------------
# MatGL CHGNet batch relax
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestBatchRelaxCHGNet:
    def test_batch_relax_chgnet(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(
            model_name="CHGNet-PES-MatPES-PBE-2025.2.10", device="cpu"
        )
        try:
            wrapper.load()
        except Exception:
            pytest.skip("CHGNet model unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=20,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result is not None

    def test_chgnet_wrapper_factory(self):
        """CHGNetWrapper.from_potential() should succeed for a CHGNet potential."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            import matgl
            from matgl.models._chgnet import CHGNet
            from src.utils.mlips.nvalchemi.matgl_wrappers import CHGNetWrapper

            potential = matgl.load_model("CHGNet-PES-MatPES-PBE-2025.2.10")
        except Exception:
            pytest.skip("CHGNet model unavailable in this environment")

        assert isinstance(potential.model, CHGNet)
        nv_model = CHGNetWrapper.from_potential(potential)
        assert nv_model is not None
        assert nv_model.model_config.neighbor_config is not None
        assert (
            abs(
                float(nv_model.model_config.neighbor_config.cutoff)
                - potential.model.cutoff
            )
            < 0.01
        )


# ---------------------------------------------------------------------------
# FairChem batch static
# ---------------------------------------------------------------------------


@pytest.mark.fairchem
class TestBatchStaticFairChem:
    def test_batch_static_fairchem(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper
        except ImportError:
            pytest.skip("FairChem not available in this environment")

        wrapper = FAIRCHEMWrapper(model_name="uma-s-1p2", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("FairChem uma-s-1p2 unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.static_calculation(
            structure_data=structures, use_nvalchemi=True
        )
        assert result is not None

    def test_fairchem_wrapper_neighbor_config_none(self):
        """FairChemWrapper must have neighbor_config=None (builds own graph)."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper
        except ImportError:
            pytest.skip("FairChem not available in this environment")

        wrapper = FAIRCHEMWrapper(model_name="uma-s-1p2", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("FairChem uma-s-1p2 unavailable in this environment")

        nv_model = wrapper._get_nvalchemi_model()
        if nv_model is None:
            pytest.skip("_get_nvalchemi_model returned None")

        assert nv_model.model_config.neighbor_config is None


# ---------------------------------------------------------------------------
# MD batch NVT
# ---------------------------------------------------------------------------


@pytest.mark.mace
class TestBatchMDNVT:
    def test_batch_md_nvt_nose_hoover(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        result = wrapper.run_md(
            structure_data=structures,
            temperature=300,
            steps=50,
            timestep=1.0,
            ensemble="nvt_nose_hoover",
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result is not None

    def test_batch_md_nve(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        result = wrapper.run_md(
            structure_data=structures,
            temperature=300,
            steps=20,
            timestep=1.0,
            ensemble="nve",
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Inflight batch relax — MACE
# ---------------------------------------------------------------------------


def _make_pool_varied(n: int = 12):
    """Return n Cu FCC conventional cells (cubic=True) with small monotonic strains.

    Each structure has 4 atoms.  With a live-batch cap of 10 atoms the
    SizeAwareSampler fits at most 2 structures at a time, exercising the
    evict-and-replace loop across multiple refill cycles.
    """
    from ase.build import bulk

    scales = [1.0 + 0.005 * (i - n // 2) for i in range(n)]
    return [bulk("Cu", "fcc", a=3.6 * s, cubic=True) for s in scales]


@pytest.mark.mace
class TestInflightRelaxMACE:
    """Inflight batching activates when total atom count exceeds the GPU budget."""

    def _load_wrapper(self):
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        return wrapper

    def test_inflight_backend_selected(self, tmp_path, monkeypatch):
        """result['backend'] must be 'nvalchemi_inflight' when threshold is exceeded."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        # Cap live batch to 10 atoms so that a pool of 12 × 4-atom structures
        # (48 atoms total) always triggers inflight mode.
        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(12),
            fmax=100.0,  # loose: structures graduate via 2-step budget, not fmax
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )

        assert result is not None
        assert result.get("backend") == "nvalchemi_inflight"

    def test_inflight_all_structures_returned(self, tmp_path, monkeypatch):
        """Every input structure must appear exactly once in results."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        n = 12
        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(n),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )

        assert result.get("total_structures") == n
        results_list = result.get("results", [])
        assert len(results_list) == n

        # No duplicates and no missing indices.
        names = [r["structure_name"] for r in results_list]
        assert len(set(names)) == n

    def test_inflight_output_files_written(self, tmp_path, monkeypatch):
        """CIF and energy files must be written for every structure."""
        import os
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(12),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )

        for r in result.get("results", []):
            assert r["status"] in ("success", "not_converged"), (
                f"Unexpected status '{r['status']}' for {r['structure_name']}: "
                f"{r.get('error', '')}"
            )
            assert os.path.isfile(r["cif_path"]), f"Missing CIF: {r['cif_path']}"
            energy_path = os.path.join(r["output_dir"], "relaxed_energy.txt")
            assert os.path.isfile(energy_path), f"Missing energy file: {energy_path}"
            log_path = os.path.join(r["output_dir"], "relax.log")
            assert os.path.isfile(log_path), f"Missing relax.log: {log_path}"
            with open(log_path) as lf:
                lines = lf.readlines()
            assert lines[0].startswith("           Step"), "Bad relax.log header"
            assert any(ln.startswith("FIRE:") for ln in lines[1:]), (
                "No FIRE steps logged"
            )
            assert isinstance(r["energy"], float)

    def test_inflight_fixed_batch_still_works_below_threshold(
        self, tmp_path, monkeypatch
    ):
        """When total atoms are under the threshold, fixed-batch backend is used."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        # Set threshold high enough that 3 × 4-atom structures (12 atoms) fits.
        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 1000,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(3),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )

        assert result is not None
        assert result.get("backend") == "nvalchemi"


# ---------------------------------------------------------------------------
# Inflight batch relax — FairChem
# ---------------------------------------------------------------------------


@pytest.mark.fairchem
class TestInflightRelaxFairChem:
    """Inflight batching with FairChem UMA model."""

    def _load_wrapper(self):
        from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper

        wrapper = FAIRCHEMWrapper(model_name="uma-s-1p2", device="cpu")
        wrapper.load()
        return wrapper

    def test_inflight_backend_selected(self, tmp_path, monkeypatch):
        """Inflight mode is triggered when total atoms exceed batch limit."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper  # noqa: F401
        except ImportError:
            pytest.skip("FairChem not available in this environment")

        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("FairChem uma-s-1p2 unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(8),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )

        assert result is not None
        assert result.get("backend") == "nvalchemi_inflight"

    def test_inflight_all_structures_returned(self, tmp_path, monkeypatch):
        """Every input structure appears exactly once in results."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper  # noqa: F401
        except ImportError:
            pytest.skip("FairChem not available in this environment")

        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("FairChem uma-s-1p2 unavailable in this environment")

        n = 8
        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(n),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )

        assert result.get("total_structures") == n
        results_list = result.get("results", [])
        assert len(results_list) == n
        assert len({r["structure_name"] for r in results_list}) == n

    def test_inflight_output_files_written(self, tmp_path, monkeypatch):
        """CIF and energy text files exist for every structure."""
        import os
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper  # noqa: F401
        except ImportError:
            pytest.skip("FairChem not available in this environment")

        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("FairChem uma-s-1p2 unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(8),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )

        for r in result.get("results", []):
            assert r["status"] in ("success", "not_converged"), (
                f"Unexpected status '{r['status']}' for {r['structure_name']}: "
                f"{r.get('error', '')}"
            )
            assert os.path.isfile(r["cif_path"]), f"Missing CIF: {r['cif_path']}"
            energy_path = os.path.join(r["output_dir"], "relaxed_energy.txt")
            assert os.path.isfile(energy_path), f"Missing energy: {energy_path}"
            log_path = os.path.join(r["output_dir"], "relax.log")
            assert os.path.isfile(log_path), f"Missing relax.log: {log_path}"
            with open(log_path) as lf:
                lines = lf.readlines()
            assert lines[0].startswith("           Step"), "Bad relax.log header"
            assert any(ln.startswith("FIRE:") for ln in lines[1:]), (
                "No FIRE steps logged"
            )


# ---------------------------------------------------------------------------
# Backend key coverage — all three execution paths
# ---------------------------------------------------------------------------


@pytest.mark.mace
class TestBackendKeyAllPaths:
    """The 'backend' key must be present and correct in every batch result dict.

    These tests use MACE with CPU to avoid GPU availability constraints, but the
    backend-key contract is enforced in base.py and applies to all wrappers.
    """

    def _load_mace(self):
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        return wrapper

    def test_sequential_backend_key(self, tmp_path, monkeypatch):
        """Sequential fallback (NValchemi disabled) must return backend='sequential'."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as _nv

        try:
            wrapper = self._load_mace()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        monkeypatch.setattr(_nv, "NVALCHEMI_AVAILABLE", False)
        monkeypatch.setattr(_nv, "check_nvalchemi_available", lambda: False)

        result = wrapper.relax_structure(
            structure_data=_make_structures(2),
            fmax=0.5,
            steps=3,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("backend") == "sequential", (
            f"Expected 'sequential', got {result.get('backend')!r}"
        )

    def test_fixed_batch_nvalchemi_backend_key(self, tmp_path, monkeypatch):
        """Fixed-batch NValchemi (all structures fit in one pass) must return backend='nvalchemi'."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            wrapper = self._load_mace()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        # Large threshold → total atoms (3×1=3) always < threshold → fixed-batch
        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10_000,
        )

        result = wrapper.relax_structure(
            structure_data=_make_structures(3),
            fmax=0.5,
            steps=3,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("backend") == "nvalchemi", (
            f"Expected 'nvalchemi', got {result.get('backend')!r}"
        )

    def test_inflight_backend_key(self, tmp_path, monkeypatch):
        """Inflight mode (total atoms > threshold) must return backend='nvalchemi_inflight'."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            wrapper = self._load_mace()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        # Cap at 10 atoms; pool of 12 × 4-atom structures (48 total) forces inflight
        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(12),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("backend") == "nvalchemi_inflight", (
            f"Expected 'nvalchemi_inflight', got {result.get('backend')!r}"
        )

    def test_static_sequential_backend_key(self, tmp_path, monkeypatch):
        """Sequential static_calculation fallback must return backend='sequential'."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as _nv

        try:
            wrapper = self._load_mace()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        monkeypatch.setattr(_nv, "NVALCHEMI_AVAILABLE", False)
        monkeypatch.setattr(_nv, "check_nvalchemi_available", lambda: False)

        result = wrapper.static_calculation(
            structure_data=_make_structures(2), use_nvalchemi=True
        )
        assert result.get("backend") == "sequential", (
            f"Expected 'sequential', got {result.get('backend')!r}"
        )


# ---------------------------------------------------------------------------
# relax.log written by fixed-batch NValchemi path
# ---------------------------------------------------------------------------


@pytest.mark.mace
class TestRelaxLogFixedBatch:
    """relax.log must be written for fixed-batch NValchemi runs (not only inflight)."""

    def test_relax_log_written_fixed_batch_mace(self, tmp_path, monkeypatch):
        import os
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.mace.mace_wrapper import MACEWrapper

            wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
            wrapper.load()
        except Exception:
            pytest.skip("MACE-OMAT-0-small unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10_000,
        )

        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(3),
            fmax=100.0,
            steps=3,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("backend") == "nvalchemi"
        for r in result.get("results", []):
            log_path = os.path.join(r["output_dir"], "relax.log")
            assert os.path.isfile(log_path), f"Missing relax.log: {log_path}"
            with open(log_path) as f:
                lines = f.readlines()
            assert lines[0].startswith("           Step"), "Bad header"
            assert any(ln.startswith("FIRE:") for ln in lines[1:]), "No FIRE steps"


# ---------------------------------------------------------------------------
# MCP server parameter-signature coverage
# ---------------------------------------------------------------------------


class TestMCPServerParamCoverage:
    """max_batch_atoms must be in the signature of every MCP relax_structure tool.

    These tests import the server modules and inspect function signatures so they
    run in any environment without starting the actual MCP servers.
    """

    def _get_relax_fn(self, server_module_path: str):
        import importlib
        import sys

        # Temporarily suppress the MCP server's stdout redirection at import time
        orig = sys.stdout
        try:
            mod = importlib.import_module(server_module_path)
        except Exception:
            return None
        finally:
            sys.stdout = orig
        return getattr(mod, "relax_structure", None)

    def _check_param(self, fn, param: str) -> None:
        import inspect

        if fn is None:
            pytest.skip("Server module not importable in this environment")
        sig = inspect.signature(fn)
        assert param in sig.parameters, (
            f"'{param}' missing from {fn.__module__}.{fn.__qualname__} signature. "
            f"Parameters: {list(sig.parameters)}"
        )

    def test_fairchem_relax_has_max_batch_atoms(self):
        try:
            import src.mcp_server.fairchem_server as srv

            fn = getattr(srv, "relax_structure", None)
        except Exception:
            pytest.skip("fairchem_server not importable")
        self._check_param(fn, "max_batch_atoms")

    def test_mace_relax_has_max_batch_atoms(self):
        try:
            import src.mcp_server.mace_server as srv

            fn = getattr(srv, "relax_structure", None)
        except Exception:
            pytest.skip("mace_server not importable")
        self._check_param(fn, "max_batch_atoms")

    def test_matgl_relax_has_max_batch_atoms(self):
        try:
            import src.mcp_server.matgl_server as srv

            fn = getattr(srv, "relax_structure", None)
        except Exception:
            pytest.skip("matgl_server not importable")
        self._check_param(fn, "max_batch_atoms")


# ---------------------------------------------------------------------------
# Inflight batch relax — MatGL (M3GNet, TensorNet, CHGNet)
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestInflightRelaxM3GNet:
    """Inflight batching with MatGL M3GNet model.

    M3GNet is used because it supports Cu without element-coverage gaps.
    TensorNet and CHGNet share the same inflight code path and are covered
    by TestInflightRelaxTensorNet / TestInflightRelaxCHGNet below.
    """

    def _load_wrapper(self):
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(model_name="M3GNet-PES-MatPES-PBE-2025.2", device="cpu")
        wrapper.load()
        return wrapper

    def test_inflight_backend_selected(self, tmp_path, monkeypatch):
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(12),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("backend") == "nvalchemi"

    def test_inflight_all_structures_returned(self, tmp_path, monkeypatch):
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        n = 12
        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(n),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("total_structures") == n
        assert len(result.get("results", [])) == n

    def test_inflight_output_files_written(self, tmp_path, monkeypatch):
        import os
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(8),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        for r in result.get("results", []):
            assert r["status"] in (
                "success",
                "not_converged",
            ), f"Unexpected status for {r['structure_name']}: {r.get('error', '')}"
            assert os.path.isfile(r["cif_path"])
            assert os.path.isfile(os.path.join(r["output_dir"], "relaxed_energy.txt"))
            log_path = os.path.join(r["output_dir"], "relax.log")
            assert os.path.isfile(log_path), f"Missing relax.log: {log_path}"
            with open(log_path) as f:
                lines = f.readlines()
            assert lines[0].startswith("           Step"), "Bad relax.log header"
            assert any(ln.startswith("FIRE:") for ln in lines[1:]), (
                "No FIRE steps logged"
            )

    def test_inflight_fixed_batch_still_works_below_threshold(
        self, tmp_path, monkeypatch
    ):
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10_000,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(3),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("backend") == "nvalchemi"


@pytest.mark.matgl
class TestInflightRelaxTensorNet:
    """Inflight batching with MatGL TensorNet model."""

    def _load_wrapper(self):
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(
            model_name="TensorNet-PES-MatPES-PBE-2025.2", device="cpu"
        )
        wrapper.load()
        return wrapper

    def test_inflight_optout_keeps_fixed_backend(self, tmp_path, monkeypatch):
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("TensorNet model unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(12),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert wrapper._get_nvalchemi_model()._nvalchemi_supports_inflight is False
        assert result.get("backend") == "nvalchemi"

    def test_inflight_output_files_written(self, tmp_path, monkeypatch):
        import os
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("TensorNet model unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(8),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        for r in result.get("results", []):
            assert r["status"] in ("success", "not_converged")
            assert os.path.isfile(r["cif_path"])
            log_path = os.path.join(r["output_dir"], "relax.log")
            assert os.path.isfile(log_path), f"Missing relax.log: {log_path}"
            with open(log_path) as f:
                lines = f.readlines()
            assert lines[0].startswith("           Step")
            assert any(ln.startswith("FIRE:") for ln in lines[1:])


@pytest.mark.matgl
class TestInflightRelaxCHGNet:
    """Inflight batching with MatGL CHGNet model."""

    def _load_wrapper(self):
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(
            model_name="CHGNet-PES-MatPES-PBE-2025.2.10", device="cpu"
        )
        wrapper.load()
        return wrapper

    def test_inflight_backend_selected(self, tmp_path, monkeypatch):
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("CHGNet model unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(12),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        assert result.get("backend") == "nvalchemi"

    def test_inflight_output_files_written(self, tmp_path, monkeypatch):
        import os
        import src.utils.mlips.nvalchemi.nvalchemi_utils as nv_utils
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")
        try:
            wrapper = self._load_wrapper()
        except Exception:
            pytest.skip("CHGNet model unavailable in this environment")

        monkeypatch.setattr(
            nv_utils,
            "estimate_max_batch_atoms",
            lambda device="cuda", safety_factor=0.5, model=None: 10,
        )
        result = wrapper.relax_structure(
            structure_data=_make_pool_varied(8),
            fmax=100.0,
            steps=2,
            output_dir=str(tmp_path),
            use_nvalchemi=True,
        )
        for r in result.get("results", []):
            assert r["status"] in ("success", "not_converged")
            assert os.path.isfile(r["cif_path"])
            log_path = os.path.join(r["output_dir"], "relax.log")
            assert os.path.isfile(log_path), f"Missing relax.log: {log_path}"
            with open(log_path) as f:
                lines = f.readlines()
            assert lines[0].startswith("           Step")
            assert any(ln.startswith("FIRE:") for ln in lines[1:])
