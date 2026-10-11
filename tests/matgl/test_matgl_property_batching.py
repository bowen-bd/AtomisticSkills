"""Batched property predictions must match one-structure-at-a-time evaluation.

Regression test for ml-property-predictor's ``train_matgl_property.py``: CHGNet
returns one value per graph with shape [B], and dividing it by a [B, 1] atom-count
tensor broadcast to [B, B], so losses and errors were wrong for batches of
structures with different atom counts.

Run in: venv/run mlip python -m pytest tests/matgl/test_matgl_property_batching.py
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "ml-property-predictor"
    / "scripts"
    / "train_matgl_property.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("train_matgl_property", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structures():
    """Three cells with unequal atom counts (2, 4 and 8 atoms)."""
    from pymatgen.core import Lattice, Structure

    nacl = Structure(
        Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()
    nacl_2x = nacl.copy()
    nacl_2x.make_supercell([2, 1, 1])
    mgo = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return [nacl, nacl_2x, mgo]


@pytest.mark.matgl
class TestPropertyBatching:
    """predict_batch and evaluate agree with per-structure evaluation."""

    def _setup(self):
        import torch
        from matgl.ext.pymatgen import Structure2Graph
        from matgl.models import CHGNet

        mod = _load_script()
        torch.manual_seed(0)
        model = CHGNet(element_types=("Na", "Cl", "Mg", "O")).eval()
        converter = Structure2Graph(
            element_types=model.element_types, cutoff=model.cutoff
        )
        structures = _structures()
        assert len({len(s) for s in structures}) == len(structures)
        labels = [1.5, -0.25, 3.0]
        items = [
            (*mod.structure_to_graph(s, converter), y)
            for s, y in zip(structures, labels)
        ]
        items = [(g, y, st) for g, st, y in items]
        return mod, model, items, labels

    @pytest.mark.parametrize("per_atom", [False, True])
    def test_batched_predictions_match_single(self, per_atom, skip_if_wrong_env):
        import torch

        mod, model, items, _ = self._setup()
        with torch.no_grad():
            graph, _, states = mod.collate_graphs(items)
            batched = mod.predict_batch(model, graph, states, per_atom)
            single = torch.cat(
                [
                    mod.predict_batch(model, *mod.collate_graphs([it])[::2], per_atom)
                    for it in items
                ]
            )
        assert batched.shape == (len(items),)
        assert torch.allclose(batched, single, atol=1e-5)

    def test_evaluate_weights_structures_not_batches(self, skip_if_wrong_env):
        import torch
        from torch.utils.data import DataLoader

        mod, model, items, labels = self._setup()
        with torch.no_grad():
            preds = [
                float(mod.predict_batch(model, *mod.collate_graphs([it])[::2], True))
                for it in items
            ]
        errors = torch.tensor(preds) - torch.tensor(labels)
        expected_mse = float((errors**2).mean())
        expected_mae = float(errors.abs().mean())

        # batch_size=2 gives uneven batches (2 + 1 structures).
        loader = DataLoader(items, batch_size=2, collate_fn=mod.collate_graphs)
        mse, mae = mod.evaluate(model, loader, "cpu", per_atom=True)
        assert mse == pytest.approx(expected_mse, rel=1e-5)
        assert mae == pytest.approx(expected_mae, rel=1e-5)
