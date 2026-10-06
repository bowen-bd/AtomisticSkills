"""Convergence and geometry regressions independent of downloaded models."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read

torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

from nvalchemi.data import Batch
from nvalchemi.dynamics.sinks import HostMemory
from src.utils.mlips.base import MLIPModel
from src.utils.mlips.nvalchemi.nvalchemi_utils import (
    atoms_to_atomic_data,
    extract_batch_results,
)


@pytest.mark.parametrize("history", [False, True])
def test_relax_extraction_reports_final_convergence(tmp_path, history):
    """Saving a geometry successfully does not imply optimizer convergence."""
    batch = Batch.from_data_list(
        [atoms_to_atomic_data(bulk("Cu", cubic=True)) for _ in range(2)]
    )
    batch["energy"] = torch.tensor([[-1.0], [-2.0]])
    batch["forces"] = torch.zeros(8, 3)
    batch["status"] = torch.tensor([0, 0], dtype=torch.int32)
    sink = HostMemory(capacity=2) if history else None
    if sink is not None:
        sink.write(batch)
    # The final state need not have a snapshot (e.g. a sparse history).
    batch["status"] = torch.tensor([0, 1], dtype=torch.int32)
    batch["energy"] = torch.tensor([[-3.0], [-4.0]])
    batch.positions += 0.01
    results = extract_batch_results(
        batch,
        ["a", "b"],
        [str(tmp_path / "a"), str(tmp_path / "b")],
        mode="relax",
        memory_sink=sink,
    )
    assert [r["status"] for r in results] == ["not_converged", "success"]
    assert [r["converged"] for r in results] == [False, True]
    assert [r["energy"] for r in results] == [-3.0, -4.0]
    if history:
        assert read(results[1]["trajectory_path"], -1).get_potential_energy() == -4.0


@pytest.mark.parametrize("steps, expected", [(0, "not_converged"), (200, "success")])
def test_sequential_relax_reports_optimizer_result(tmp_path, steps, expected):
    """Exercise ASE FIRE instead of mocking its convergence flag."""
    model = SimpleNamespace(
        create_calculator=EMT,
        check_structure_data=lambda atoms: atoms.copy(),
    )
    atoms = bulk("Cu", cubic=True)
    atoms.positions[0, 0] += 0.1
    result = MLIPModel._single_relax(
        model, atoms, 0.01, steps, "FIRE", False, str(tmp_path), None
    )
    assert "error" not in result
    assert result["status"] == expected
    assert result["converged"] == (expected == "success")
    assert result["steps"] <= steps


def test_cell_stress_prevents_convergence_after_graduation():
    """Status=1 is ambiguous in FusedStage: a budget exit is not convergence."""
    from src.utils.mlips.nvalchemi.nvalchemi_utils import relax_convergence_hook

    batch = Batch.from_data_list([atoms_to_atomic_data(bulk("Cu", cubic=True))])
    batch["forces"] = torch.zeros(4, 3)
    batch["stress"] = torch.eye(3).unsqueeze(0)
    batch["status"] = torch.ones(1, dtype=torch.int32)
    assert relax_convergence_hook(0.01, False).evaluate(batch) is not None
    assert relax_convergence_hook(0.01, True).evaluate(batch) is None


@pytest.mark.mace
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["fixed", "inflight", "sequential"])
def test_step_limit_is_not_success(tmp_path, mode):
    """A five-step budget on perturbed crystals must report nonconvergence."""
    from src.utils.mlips.loader import load_wrapper

    wrapper = load_wrapper("mace", "MACE-OMAT-0-small", device="cuda")
    structures = [bulk("Cu", cubic=True), bulk("Si", cubic=True)]
    for atoms in structures:
        atoms.positions[0, 0] += 0.1
        atoms.set_cell(atoms.cell * 1.03, scale_atoms=True)
    if mode == "sequential":
        result = wrapper._batch_relax_sequential(
            structures, 1e-6, 5, "FIRE", True, str(tmp_path)
        )
    else:
        result = wrapper.relax_structure(
            structures,
            fmax=1e-6,
            steps=5,
            relax_cell=True,
            output_dir=str(tmp_path),
            max_batch_atoms=8 if mode == "inflight" else 100,
        )
        assert result["backend"] == (
            "nvalchemi_inflight" if mode == "inflight" else "nvalchemi"
        )
    assert result["successful"] == result["failed"] == 0
    assert result["not_converged"] == 2
    assert all(r["status"] == "not_converged" for r in result["results"])
    assert all(not r["converged"] for r in result["results"])
    assert all(r["steps"] == 5 for r in result["results"])


@pytest.mark.mace
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_variable_cell_relax_recovers_strained_supercell(tmp_path):
    """The unscaled upstream step must relax a sheared cell without distortion."""
    from src.utils.mlips.loader import load_wrapper

    wrapper = load_wrapper("mace", "MACE-OMAT-0-small", device="cuda")
    atoms = bulk("Si", cubic=True).repeat(2)  # 64 atoms
    atoms.set_cell(
        atoms.cell.array @ np.array([[1.03, 0.05, 0], [0, 0.98, 0.02], [0, 0, 1.01]]),
        scale_atoms=True,
    )
    atoms.rattle(stdev=0.02, seed=47)
    result = wrapper.relax_structure(
        [atoms.copy()],
        fmax=0.005,
        steps=600,
        relax_cell=True,
        output_dir=str(tmp_path / "batch"),
    )
    baseline = wrapper._single_relax(
        atoms.copy(), 0.005, 600, "FIRE", True, str(tmp_path / "ase"), None
    )
    assert result["backend"] == "nvalchemi"
    row = result["results"][0]
    assert row["status"] == baseline["status"] == "success"
    optimized, reference = read(row["cif_path"]), read(baseline["cif_path"])
    assert abs(optimized.get_volume() / reference.get_volume() - 1) <= 0.005
    optimized.calc = wrapper.create_calculator()
    assert (
        abs(optimized.get_potential_energy() - baseline["energy"]) / len(atoms) <= 0.001
    )
    assert np.linalg.norm(optimized.get_forces(), axis=1).max() <= 0.0051
    assert row["steps"] <= 4 * baseline["steps"]
