"""
Tests for MACE accelerator selection and its fallbacks.

CuEquivariance is an optimisation, never a requirement: a CPU-only host, an
environment that did not sync the cu130 extra, or a card its kernels do not
cover must all still be able to run MACE. These tests pin that behaviour with
a fake calculator factory, so they need no GPU, no cuequivariance install and
no checkpoint download.

Run in: venv/run mlip python -m pytest tests/mace/test_mace_accelerator.py
"""

import logging

import pytest

from src.utils.mlips.mace import mace_wrapper as mw


def fake_calculator(**kwargs):
    """Stand-in for a MACE calculator that records how it was built."""
    return {"kwargs": kwargs}


@pytest.fixture
def wrapper_cuda():
    """A wrapper pointed at CUDA, without touching any hardware."""
    w = mw.MACEWrapper(model_name="MACE-MH-1", device="cuda")
    w.is_loaded = True
    return w


@pytest.mark.mace
class TestAcceleratorSelection:
    def test_auto_on_cpu_never_requests_cueq(self):
        w = mw.MACEWrapper(model_name="MACE-MH-1", device="cpu")
        built = w._build_calculator(fake_calculator, device="cpu")
        assert "enable_cueq" not in built["kwargs"]
        assert w.accelerator_effective == "none"

    def test_auto_without_cueq_installed_degrades(self, wrapper_cuda, monkeypatch):
        monkeypatch.setattr(mw, "cueq_installed", lambda: False)
        built = wrapper_cuda._build_calculator(fake_calculator, device="cuda")
        assert "enable_cueq" not in built["kwargs"]
        assert wrapper_cuda.accelerator_effective == "none"

    def test_auto_with_cueq_requests_it(self, wrapper_cuda, monkeypatch):
        monkeypatch.setattr(mw, "cueq_installed", lambda: True)
        built = wrapper_cuda._build_calculator(fake_calculator, device="cuda")
        assert built["kwargs"]["enable_cueq"] is True
        assert wrapper_cuda.accelerator_effective == "cueq"

    def test_none_never_attempts_even_when_installed(self, monkeypatch):
        monkeypatch.setattr(mw, "cueq_installed", lambda: True)
        w = mw.MACEWrapper(model_name="MACE-MH-1", device="cuda", accelerator="none")
        built = w._build_calculator(fake_calculator, device="cuda")
        assert "enable_cueq" not in built["kwargs"]
        assert w.accelerator_effective == "none"

    def test_explicit_cueq_request_still_degrades(self, wrapper_cuda, monkeypatch):
        """Asking for cueq is a request, not a promise."""
        monkeypatch.setattr(mw, "cueq_installed", lambda: False)
        w = mw.MACEWrapper(model_name="MACE-MH-1", device="cuda", accelerator="cueq")
        built = w._build_calculator(fake_calculator, device="cuda")
        assert "enable_cueq" not in built["kwargs"]
        assert w.accelerator_effective == "none"

    def test_unknown_accelerator_rejected(self):
        with pytest.raises(ValueError, match="Unknown accelerator"):
            mw.MACEWrapper(model_name="MACE-MH-1", accelerator="fastest")


@pytest.mark.mace
class TestAcceleratorFallback:
    def test_incompatible_card_degrades_instead_of_raising(
        self, wrapper_cuda, monkeypatch, caplog
    ):
        """A GPU cuequivariance has no kernels for must not take MACE down."""
        monkeypatch.setattr(mw, "cueq_installed", lambda: True)

        def factory(**kwargs):
            if kwargs.get("enable_cueq"):
                raise RuntimeError("CUDA error: no kernel image is available")
            return fake_calculator(**kwargs)

        with caplog.at_level(logging.WARNING):
            built = wrapper_cuda._build_calculator(factory, device="cuda")

        assert "enable_cueq" not in built["kwargs"]
        assert wrapper_cuda.accelerator_effective == "none"
        assert any("no kernel image" in r.getMessage() for r in caplog.records)

    def test_missing_module_degrades(self, wrapper_cuda, monkeypatch):
        """mace raises ImportError when cuequivariance is absent; absorb it."""
        monkeypatch.setattr(mw, "cueq_installed", lambda: True)

        def factory(**kwargs):
            if kwargs.get("enable_cueq"):
                raise ImportError("cuequivariance is not installed")
            return fake_calculator(**kwargs)

        built = wrapper_cuda._build_calculator(factory, device="cuda")
        assert "enable_cueq" not in built["kwargs"]

    def test_real_error_on_the_plain_path_surfaces(self, wrapper_cuda, monkeypatch):
        """The plain path is unguarded, so genuine bugs still raise."""
        monkeypatch.setattr(mw, "cueq_installed", lambda: True)

        def factory(**kwargs):
            raise TypeError("model_paths must be a path")

        with pytest.raises(TypeError, match="model_paths"):
            wrapper_cuda._build_calculator(factory, device="cuda")

    def test_fallback_is_remembered_for_later_calls(self, wrapper_cuda, monkeypatch):
        """One failed attempt must not cost a rebuild on every structure."""
        monkeypatch.setattr(mw, "cueq_installed", lambda: True)
        attempts = []

        def factory(**kwargs):
            attempts.append(kwargs.get("enable_cueq", False))
            if kwargs.get("enable_cueq"):
                raise RuntimeError("no kernel image")
            return fake_calculator(**kwargs)

        wrapper_cuda._build_calculator(factory, device="cuda")
        wrapper_cuda._build_calculator(factory, device="cuda")
        assert attempts == [True, False, False]


@pytest.mark.mace
class TestDescribeAccelerator:
    def test_reports_cpu_as_none(self):
        w = mw.MACEWrapper(model_name="MACE-MH-1", device="cpu")
        assert "CPU" in w.describe_accelerator() or "cpu" in w.describe_accelerator()

    def test_reports_missing_package(self, wrapper_cuda, monkeypatch):
        monkeypatch.setattr(mw, "cueq_installed", lambda: False)
        assert "not installed" in wrapper_cuda.describe_accelerator()

    def test_reports_cueq_available(self, wrapper_cuda, monkeypatch):
        monkeypatch.setattr(mw, "cueq_installed", lambda: True)
        assert wrapper_cuda.describe_accelerator().startswith("cueq")
