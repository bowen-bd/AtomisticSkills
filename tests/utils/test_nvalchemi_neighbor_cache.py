"""Neighbor topology must follow live geometry during batched dynamics."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

from ase.build import bulk
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.base import NeighborConfig, NeighborListFormat

from src.utils.mlips.nvalchemi.neighbor_list import make_neighbor_list_hook
from src.utils.mlips.nvalchemi.nvalchemi_utils import (
    atoms_to_atomic_data,
    warp_on_torch_stream,
)


def edge_keys(batch):
    """Canonicalize directed edges including periodic image shifts."""
    edges = batch.neighbor_list.long()
    shifts = batch.neighbor_list_shifts.long() + 32
    return (
        (
            (
                ((edges[:, 0] * batch.num_nodes + edges[:, 1]) * 64 + shifts[:, 0]) * 64
                + shifts[:, 1]
            )
            * 64
            + shifts[:, 2]
        )
        .sort()
        .values
    )


@pytest.mark.mace
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mace_inflight_refill_matches_fresh_neighbors(tmp_path, monkeypatch):
    """Audit actual MACE inputs after same-size refills through the public API."""
    from src.utils.mlips.loader import load_wrapper

    wrapper = load_wrapper("mace", "MACE-OMAT-0-small", device="cuda:0")
    model = wrapper._get_nvalchemi_model()
    config = model.model_config.neighbor_config
    assert config is not None
    forward = model.forward
    observed = []

    def audited_forward(batch, *args, **kwargs):
        reference = batch.clone()
        NeighborListHook(config)._rebuild(reference)
        ids = tuple(batch.orig_idx.flatten().tolist())
        observed.append((ids, torch.equal(edge_keys(batch), edge_keys(reference))))
        return forward(batch, *args, **kwargs)

    monkeypatch.setattr(model, "forward", audited_forward)
    hcp = bulk("Si", "hcp", a=2.5, c=4.0)
    displaced = hcp.copy()
    displaced.positions[0, 0] += 0.2
    structures = [hcp, displaced, bulk("Si"), bulk("Si", "bcc", a=3.4, cubic=True)]
    result = wrapper.relax_structure(
        structures,
        fmax=0.01,
        steps=3,
        relax_cell=False,
        max_batch_atoms=4,
        output_dir=str(tmp_path),
    )
    assert result.get("backend") == "nvalchemi_inflight", result
    assert result["failed"] == 0, result
    assert observed, "MACE forward was not exercised"
    initial_ids = observed[0][0]
    refilled = [
        (ids, equal) for ids, equal in observed if len(ids) == 2 and ids != initial_ids
    ]
    assert refilled, observed
    assert all(equal for _, equal in refilled), observed


@pytest.mark.parametrize(
    "release, guarded",
    [
        ("0.1.0", False),
        ("0.2.0", True),
        ("0.2.0rc1", True),
        ("0.2.9", True),
        ("0.3.0", False),
        ("1.0.0", False),
    ],
)
def test_guard_is_limited_to_toolkit_02(monkeypatch, release, guarded):
    """A future dependency lock must not silently retain private 0.2 handling."""
    from src.utils.mlips.nvalchemi import neighbor_list

    monkeypatch.setattr(neighbor_list, "version", lambda name: release)
    hook = make_neighbor_list_hook(NeighborConfig(cutoff=5.0))
    assert (type(hook) is not NeighborListHook) == guarded


def tiny_batch(counts=(2, 2), length=2.0, periodic=True):
    """Create same-total-size batches with controlled cells and atom partitions."""
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10,
                atomic_numbers=torch.full((n,), 29),
                cell=torch.eye(3).unsqueeze(0) * length,
                pbc=torch.full((1, 3), periodic, dtype=torch.bool),
            )
            for n in counts
        ]
    ).to("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("skin", [0.0, 1.0])
@pytest.mark.parametrize("change", ["cell", "pbc", "partition"])
def test_geometry_changes_match_fresh_hook(change, skin):
    """Cell/PBC/partition edits invalidate both allocations and Verlet reuse."""
    config = NeighborConfig(cutoff=5.0, format=NeighborListFormat.COO)
    stage = DynamicsStage.BEFORE_COMPUTE
    hook = make_neighbor_list_hook(config, skin=skin, max_neighbors=1024)
    batch = tiny_batch(
        length=8.0 if change == "cell" else 2.0, periodic=change != "pbc"
    )
    with warp_on_torch_stream("cuda"):
        hook(HookContext(batch=batch), stage)
        if change == "cell":
            batch.cell.div_(4)
        elif change == "pbc":
            batch.pbc.fill_(True)
        else:
            batch = tiny_batch(counts=(1, 3))
        hook(HookContext(batch=batch), stage)
        fresh = batch.clone()
        NeighborListHook(config, skin=skin, max_neighbors=1024)(
            HookContext(batch=fresh), stage
        )
        assert torch.equal(edge_keys(batch), edge_keys(fresh))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fixed_cell_steps_reuse_buffers():
    """Ordinary fixed-cell movement does not force full reallocation."""
    config = NeighborConfig(cutoff=5.0, format=NeighborListFormat.COO)
    hook = make_neighbor_list_hook(config)
    batch = tiny_batch(length=8.0)
    ctx = HookContext(batch=batch)
    stage = DynamicsStage.BEFORE_COMPUTE
    with warp_on_torch_stream("cuda"):
        hook(ctx, stage)
        matrix, staging = (
            hook._neighbor_matrix.data_ptr(),
            hook._buf_positions.data_ptr(),
        )
        for _ in range(5):
            batch.positions.add_(0.01)
            hook(ctx, stage)
            assert hook._neighbor_matrix.data_ptr() == matrix
            assert hook._buf_positions.data_ptr() == staging


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "species, cutoff, method, steps",
    [
        ("Cu", 5.0, None, 4),
        ("NaCl", 5.0, "batch_naive", 12),
        ("Si", 6.0, None, 10),
    ],
)
def test_gradual_compression_matches_fresh_neighbors(species, cutoff, method, steps):
    """One-percent cell steps can cross a cached allocation threshold."""
    atoms = (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True)
        if species == "NaCl"
        else bulk(species, cubic=True)
    )
    shear = np.eye(3)
    shear[1, 0] = 0.35
    atoms.set_cell(atoms.cell.array @ shear, scale_atoms=True)
    config = NeighborConfig(cutoff=cutoff, format=NeighborListFormat.COO)
    hook = make_neighbor_list_hook(config, method=method, max_neighbors=1024)
    batch = Batch.from_data_list(
        [atoms_to_atomic_data(atoms, device="cuda") for _ in range(2)]
    )
    with warp_on_torch_stream("cuda"):
        for step in range(steps + 1):
            changed = atoms.copy()
            changed.set_cell(atoms.cell * (1 - step * 0.01), scale_atoms=True)
            changed.wrap()
            fresh = Batch.from_data_list(
                [atoms_to_atomic_data(changed, device="cuda") for _ in range(2)]
            )
            batch.cell.copy_(fresh.cell)
            batch.positions.copy_(fresh.positions)
            hook(HookContext(batch=batch), DynamicsStage.BEFORE_COMPUTE)
            NeighborListHook(config, method=method, max_neighbors=1024)._rebuild(fresh)
            assert torch.equal(edge_keys(batch), edge_keys(fresh)), step


@pytest.mark.mace
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ensemble", ["nve", "nvt", "nvt_langevin", "npt"])
def test_batch_md_uses_guarded_neighbors(tmp_path, monkeypatch, caplog, ensemble):
    """Audit MD topology and guarded routing, including the existing NPT fallback."""
    from src.utils.mlips.loader import load_wrapper
    from src.utils.mlips.nvalchemi import neighbor_list

    wrapper = load_wrapper("mace", "MACE-OMAT-0-small", device="cuda:0")
    model = wrapper._get_nvalchemi_model()
    forward = model.forward
    observed, cells, hooks = [], [], []
    factory = neighbor_list.make_neighbor_list_hook

    def record_hook(*args, **kwargs):
        hook = factory(*args, **kwargs)
        hooks.append(hook)
        return hook

    def audit(batch, *args, **kwargs):
        reference = batch.clone()
        NeighborListHook(model.model_config.neighbor_config)._rebuild(reference)
        observed.append(torch.equal(edge_keys(batch), edge_keys(reference)))
        cells.append(batch.cell.detach().clone())
        return forward(batch, *args, **kwargs)

    monkeypatch.setattr(neighbor_list, "make_neighbor_list_hook", record_hook)
    monkeypatch.setattr(model, "forward", audit)
    result = wrapper.run_md(
        [bulk("Cu", cubic=True), bulk("Cu", cubic=True)],
        temperature=10,
        steps=3,
        timestep=0.1,
        ensemble=ensemble,
        log_interval=1,
        output_dir=str(tmp_path),
    )
    assert hooks and all(type(h) is not NeighborListHook for h in hooks)
    assert result["failed"] == 0, result
    if ensemble == "npt" and result.get("backend") != "nvalchemi":
        # The existing 0.2 integration does not publish initial stress, so NPT
        # currently takes the ASE fallback before advancing a cell. Ensure its
        # registered neighbor hook is guarded too, without masking other errors.
        assert "'Batch' has no attribute 'stress'" in caplog.text
        assert result["successful"] == 2
        return
    assert result.get("backend") == "nvalchemi", result
    assert observed and all(observed)
    assert torch.equal(cells[0], cells[-1]) == (ensemble != "npt")
