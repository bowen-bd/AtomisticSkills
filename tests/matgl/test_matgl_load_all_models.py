"""
Test that every pretrained model matgl lists loads, the way the matgl MCP
server loads it (MatGLWrapper, which allowlists what the checkpoints pickle).

Run in: venv/run mlip python -m pytest tests/matgl/test_matgl_load_all_models.py
"""

import pytest
import matgl

from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper


ALL_MODELS = matgl.get_available_pretrained_models()


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_load_pretrained_model(model_name):
    """Verify every pretrained model can be loaded without error."""
    wrapper = MatGLWrapper(model_name=model_name, device="cpu")
    wrapper.load()
    model = wrapper.model
    assert model is not None, f"load_model('{model_name}') returned None"
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, f"Model '{model_name}' has no parameters"
