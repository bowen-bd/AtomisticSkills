"""Batch backend selection must be explicit, even with NValchemi installed."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.utils.mlips.base import MLIPModel


class RoutingModel(MLIPModel):
    """Exercise real public dispatch without loading model weights or GPU code."""

    def load(self, model_path=None):
        self.is_loaded = True

    def create_calculator(self):
        raise AssertionError("The routing test should not evaluate a model")

    def save_checkpoint(self, checkpoint_path):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass

    def predict_atomic_features(self, structure_data):
        pass


@pytest.mark.parametrize(
    "operation", ["static_calculation", "relax_structure", "run_md"]
)
@pytest.mark.parametrize("input_kind", ["list", "directory"])
@pytest.mark.parametrize("choice", [None, False, True])
def test_batch_requires_explicit_opt_in(
    monkeypatch, tmp_path, caplog, operation, input_kind, choice
):
    """Installation, cached adapters and list/directory inputs cannot enable it."""
    available = Mock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "src.utils.mlips.nvalchemi.nvalchemi_utils",
        SimpleNamespace(check_nvalchemi_available=available),
    )
    model = RoutingModel("routing-test")
    model.load()
    model._get_nvalchemi_model = Mock(return_value=object())
    for name in [
        "_batch_static_nvalchemi",
        "_batch_relax_nvalchemi",
        "_batch_md_nvalchemi",
    ]:
        setattr(model, name, Mock(return_value={"backend": "nvalchemi"}))
    for name in ["_single_static_calculation", "_single_relax", "_single_run_md"]:
        setattr(
            model,
            name,
            Mock(side_effect=lambda *a, **k: {"status": "success", "energy": 0.0}),
        )
    inputs = ["one.cif", "two.cif"]
    if input_kind == "directory":
        directory = tmp_path / "inputs"
        directory.mkdir()
        for name in inputs:
            (directory / name).touch()
        inputs = str(directory)
    kwargs = {} if choice is None else {"use_nvalchemi": choice}
    if operation != "static_calculation":
        kwargs["output_dir"] = str(tmp_path / "output")
    result = getattr(model, operation)(inputs, **kwargs)
    assert result["backend"] == ("nvalchemi" if choice else "sequential")
    assert (
        available.call_count
        == model._get_nvalchemi_model.call_count
        == int(bool(choice))
    )
    assert ("experimental NValchemi" in caplog.text) == bool(choice)
    # An explicit call must not switch later requests to NValchemi implicitly.
    if choice:
        result = getattr(model, operation)(
            inputs, **{k: v for k, v in kwargs.items() if k != "use_nvalchemi"}
        )
        assert result["backend"] == "sequential"
        assert model._get_nvalchemi_model.call_count == 1


@pytest.mark.parametrize("server_name", ["mace", "matgl", "fairchem"])
@pytest.mark.parametrize(
    "tool, method",
    [
        ("predict_structure", "static_calculation"),
        ("relax_structure", "relax_structure"),
        ("run_md", "run_md"),
    ],
)
@pytest.mark.parametrize("choice", [None, True])
def test_mcp_exposes_and_forwards_opt_in(
    monkeypatch, tmp_path, server_name, tool, method, choice
):
    """Check registered tool schemas and execute the actual MCP server function."""
    from src.utils import mcp_utils

    monkeypatch.setattr(mcp_utils, "setup_mcp_stdout", lambda: None)
    # Relaxation cleanup imports torch; CPU runtime need not install it.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    server = importlib.import_module(f"src.mcp_server.{server_name}_server")
    wrapper = Mock(is_loaded=True)
    getattr(wrapper, method).return_value = {"backend": "sequential"}
    monkeypatch.setattr(server, "wrapper", wrapper)
    schema = server.mcp._tool_manager.get_tool(tool).parameters
    assert schema["properties"]["use_nvalchemi"]["default"] is False
    kwargs = {} if choice is None else {"use_nvalchemi": choice}
    if tool != "predict_structure":
        kwargs["output_dir"] = str(tmp_path)
    getattr(server, tool)(["one.cif", "two.cif"], **kwargs)
    assert getattr(wrapper, method).call_args.kwargs["use_nvalchemi"] is bool(choice)
