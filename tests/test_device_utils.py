"""Tests for src/utils/mlips/device_utils.get_best_device.

nvidia-smi can list a GPU that torch cannot use -- typically a driver older than
the torch build's CUDA needs (driver 550 with CUDA 13 builds, seen on two lab
machines). Device selection must then fall back to the CPU instead of handing a
CUDA device to the model loader.

Requirements:
    - Environment: mlip (run with: venv/run mlip python -m pytest tests/test_device_utils.py)
"""

from __future__ import annotations

import subprocess

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


class _SmiResult:
    returncode = 0
    # index, memory.used [MiB], memory.total [MiB]: GPU 1 has the most free memory.
    stdout = "0, 4000, 81920\n1, 1000, 81920\n2, 8000, 81920\n"


@pytest.fixture
def three_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _SmiResult())


@pytest.mark.parametrize(
    "visible, count, expected",
    [
        (None, 3, "cuda:1"),  # all visible: physical 1 is ordinal 1
        ("0", 1, "cuda:0"),  # only physical 0 visible: never cuda:1
        ("2,1", 2, "cuda:1"),  # physical 1 is torch ordinal 1 in "2,1"
        ("2,0", 2, "cuda:1"),  # physical 0 (more free than 2) is ordinal 1
    ],
)
def test_nvidia_smi_indices_follow_cuda_visible_devices(
    three_gpus, monkeypatch, visible, count, expected
):
    """With CUDA_VISIBLE_DEVICES=0, picking nvidia-smi's GPU 1 gave
    'CUDA error: invalid device ordinal' (seen on an 8-GPU workstation)."""
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    assert device_utils.get_best_device("auto") == expected
