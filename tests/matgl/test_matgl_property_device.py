"""MatGL property models predict wherever the model is, whichever GPU is visible.

MatGL 4.1 builds the graph on the default CUDA device whenever one is visible,
so a model on the CPU (or on another GPU) once failed with "Expected all tensors
to be on the same device". GPU-free CI cannot see this; run on a GPU host:
    venv/run mlip python -m pytest tests/matgl/test_matgl_property_device.py
"""

import pytest
import torch
from pymatgen.core import Lattice, Structure

from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

pytestmark = pytest.mark.matgl

SI = Structure(Lattice.cubic(5.431), ["Si"] * 2, [[0, 0, 0], [0.25, 0.25, 0.25]])
DEVICES = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "model_name, key",
    [
        ("MEGNet-MP-2019.4.1-BandGap-mfi", "bandgap"),
        ("MEGNet-Eform-MP-2018.6.1", "formation_energy"),
    ],
)
def test_property_prediction_on_each_device(model_name, key, device):
    wrapper = MatGLWrapper(model_name=model_name, device=device)
    wrapper.load()
    result = wrapper.static_calculation(SI.to_ase_atoms())
    assert "error" not in result, result
    assert result[key] == pytest.approx(
        _reference(model_name, key), abs=1e-3
    ), f"{model_name} on {device}"


_REFERENCE = {}


def _reference(model_name: str, key: str) -> float:
    """The CPU prediction: every device must agree with it."""
    if model_name not in _REFERENCE:
        wrapper = MatGLWrapper(model_name=model_name, device="cpu")
        wrapper.load()
        _REFERENCE[model_name] = wrapper.static_calculation(SI.to_ase_atoms())[key]
    return _REFERENCE[model_name]
