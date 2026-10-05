"""Tests for src/utils/mlips/device_utils.get_best_device.

nvidia-smi can list a GPU that torch cannot use -- typically a driver older than
the torch build's CUDA needs (driver 550 with CUDA 13 builds, seen on two lab
machines). Device selection must then fall back to the CPU instead of handing a
CUDA device to the model loader.

Requirements:
    - Environment: mlip (run with: venv/run mlip python -m pytest tests/test_device_utils.py)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.utils.mlips import device_utils  # noqa: E402


@pytest.fixture
def no_usable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_auto_falls_back_to_cpu_when_torch_cannot_use_the_gpu(no_usable_cuda):
    assert device_utils.get_best_device("auto") == "cpu"


@pytest.mark.parametrize("request_", ["cuda", "cuda:1"])
def test_an_explicit_cuda_request_falls_back_too(no_usable_cuda, request_):
    assert device_utils.get_best_device(request_) == "cpu"


def test_cpu_is_honoured():
    assert device_utils.get_best_device("cpu") == "cpu"
