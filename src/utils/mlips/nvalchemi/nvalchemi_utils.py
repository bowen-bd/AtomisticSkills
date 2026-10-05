"""Core utilities for NValchemi GPU-accelerated MLIP batch operations."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

try:
    import torch
    from nvalchemi.data import AtomicData
    from nvalchemi.dynamics.base import ConvergenceHook, _ConvergenceCriterion
    from nvalchemi.dynamics.sinks import HostMemory
    from nvalchemi.dynamics.optimizers.fire2 import FIRE2VariableCell

    NVALCHEMI_AVAILABLE = True
except ImportError:
    NVALCHEMI_AVAILABLE = False

    class AtomicData:  # type: ignore
        pass

    class _ConvergenceCriterion:  # type: ignore
        pass

    class HostMemory:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FIRE2VariableCell:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


if TYPE_CHECKING:
    from ase import Atoms


def check_nvalchemi_available() -> bool:
    """Return True if nvalchemi-toolkit is importable."""
    return NVALCHEMI_AVAILABLE


@contextmanager
def warp_on_torch_stream(device: Any) -> Iterator[None]:
    """Launch every Warp kernel inside the block on PyTorch's current CUDA stream.

    Entering an nvalchemi dynamics object (``with optimizer:``) switches PyTorch
    to a dedicated CUDA stream, but several Warp kernels -- nvalchemiops' batched
    cell list among them (0.4) -- launch on Warp's stream, which stays the
    default one. The PyTorch op that reads the cell list's output,
    ``num_neighbors.max()`` in ``NeighborListHook``, then races the kernel: the
    hook sizes its neighbor matrix from a partial count and overflows on a later
    step ("number of neighbors is larger than the maximum allowed"), or turns a
    half-filled matrix into an edge list with neighbors missing. Enter this
    *after* the dynamics object, so Warp follows the dynamics stream and every
    launch is ordered with the PyTorch ops around it.

    Args:
        device: The device the batch lives on; CPU devices are a no-op.
    """
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        yield
        return
    import warp as wp

    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(torch_device))):
        yield


def atoms_to_atomic_data(
    atoms: "Atoms",
    device: str = "cpu",
    dtype: Optional[Any] = None,
) -> Any:
    """Convert an ASE Atoms object to a NValchemi AtomicData.

    Parameters
    ----------
    atoms : ase.Atoms
    device : str
        Target torch device string.
    dtype : torch.dtype, optional
        Floating-point dtype; defaults to torch.float32.
    """
    if not NVALCHEMI_AVAILABLE:
        raise ImportError(
            "nvalchemi-toolkit is required. "
            "Install with: pip install nvalchemi-toolkit"
        )
    if dtype is None:
        dtype = torch.float32
    # Wrap coordinates inside unit cell boundary to prevent OOM from huge neighbor list image creation
    if hasattr(atoms, "pbc") and any(atoms.pbc):
        atoms.wrap()
    return AtomicData.from_atoms(atoms, device=device, dtype=dtype)


def atomic_data_to_atoms(data: Any) -> "Atoms":
    """Reconstruct an ASE Atoms object from a NValchemi AtomicData.

    Attaches a SinglePointCalculator with energy/forces/stress if those
    fields are populated in the AtomicData.

    Parameters
    ----------
    data : AtomicData
        A single-graph NValchemi AtomicData (from Batch.get_data()).
    """
    if not NVALCHEMI_AVAILABLE:
        raise ImportError("nvalchemi-toolkit is required.")

    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    pos = data.positions.detach().cpu().numpy()
    numbers = data.atomic_numbers.detach().cpu().numpy()

    cell_tensor = getattr(data, "cell", None)
    if cell_tensor is not None:
        cell = cell_tensor.squeeze().detach().cpu().numpy()
        pbc = True
    else:
        cell = None
        pbc = False

    atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=pbc)

    # Attach last-step energetics if available
    calc_results: dict[str, Any] = {}

    energy_tensor = getattr(data, "energy", None)
    if energy_tensor is not None:
        calc_results["energy"] = float(energy_tensor.detach().cpu().sum().item())

    forces_tensor = getattr(data, "forces", None)
    if forces_tensor is not None:
        calc_results["forces"] = forces_tensor.detach().cpu().numpy()

    stress_tensor = getattr(data, "stress", None)
    if stress_tensor is not None:
        # NValchemi stress [1,3,3] eV/Å³ (Cauchy, tensile convention)
        # → ASE Voigt 6-vector [xx, yy, zz, yz, xz, xy] eV/Å³
        s = stress_tensor.squeeze(0).detach().cpu().numpy()
        calc_results["stress"] = [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]

    if calc_results:
        atoms.calc = SinglePointCalculator(atoms, **calc_results)

    return atoms


def extract_batch_results(
    final_batch: Any,
    structure_names: list[str],
    output_dirs: list[str],
    mode: str = "static",
    memory_sink: Optional[Any] = None,
    log_interval: int = 10,
    timestep: float = 1.0,
    temperature: float = 300.0,
    ensemble: str = "nvt",
) -> list[dict[str, Any]]:
    """Extract per-structure results from a NValchemi Batch or memory sink after dynamics.

    Supports static, relax, and md modes.
    """
    if not NVALCHEMI_AVAILABLE:
        raise ImportError("nvalchemi-toolkit is required.")

    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write
    import numpy as np

    results: list[dict[str, Any]] = []

    if mode == "static":
        num_graphs = final_batch.num_graphs
        data_list = final_batch.to_data_list()
        for i in range(num_graphs):
            struct_name = (
                structure_names[i] if i < len(structure_names) else f"structure_{i}"
            )
            out_dir = output_dirs[i] if i < len(output_dirs) else output_dirs[0]
            os.makedirs(out_dir, exist_ok=True)
            try:
                atoms = atomic_data_to_atoms(data_list[i])
                structure = AseAtomsAdaptor.get_structure(atoms)
                cif_path = os.path.join(out_dir, "final_structure.cif")
                structure.to(filename=cif_path)
                energy = None
                if atoms.calc is not None and "energy" in atoms.calc.results:
                    energy = atoms.calc.results["energy"]
                    with open(os.path.join(out_dir, "energy.txt"), "w") as f:
                        f.write(f"{energy}\n")
                results.append(
                    {
                        "structure_name": struct_name,
                        "status": "success",
                        "energy": energy,
                        "cif_path": cif_path,
                        "output_dir": out_dir,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "structure_name": struct_name,
                        "status": "failed",
                        "error": str(e),
                        "output_dir": out_dir,
                    }
                )
    elif mode == "relax":
        if memory_sink is not None:
            stored_batch = memory_sink.read()
            data_list = stored_batch.to_data_list()
            n_total_frames = len(data_list)
            n_structures = len(structure_names)
            n_snapshots = n_total_frames // n_structures
        else:
            data_list = final_batch.to_data_list()
            n_structures = len(structure_names)
            n_snapshots = 1

        for i in range(n_structures):
            struct_name = structure_names[i]
            out_dir = output_dirs[i]
            os.makedirs(out_dir, exist_ok=True)
            try:
                struct_atoms_list = []
                if memory_sink is not None:
                    for s in range(n_snapshots):
                        idx = s * n_structures + i
                        if idx < n_total_frames:
                            frame_data = data_list[idx]
                            struct_atoms_list.append(atomic_data_to_atoms(frame_data))
                            status_val = getattr(frame_data, "status", None)
                            if status_val is not None:
                                val = int(
                                    status_val.item()
                                    if hasattr(status_val, "item")
                                    else status_val
                                )
                                if val >= 1:
                                    break
                else:
                    struct_atoms_list.append(atomic_data_to_atoms(data_list[i]))

                if not struct_atoms_list:
                    raise RuntimeError("No trajectory snapshots recorded.")

                traj_file = None
                log_file = None
                if memory_sink is not None:
                    traj_file = os.path.join(out_dir, "relax.traj")
                    write(traj_file, struct_atoms_list, format="traj")

                    log_file = os.path.join(out_dir, "relax.log")
                    with open(log_file, "w") as f:
                        f.write("           Step          Energy         fmax\n")
                        for step_idx, atoms in enumerate(struct_atoms_list):
                            energy = atoms.get_potential_energy()
                            forces = atoms.get_forces()
                            fmax = np.sqrt((forces**2).sum(axis=-1)).max()
                            f.write(
                                f"FIRE:  {step_idx:9d}  {energy:14.6f}  {fmax:12.6f}\n"
                            )

                final_atoms = struct_atoms_list[-1]
                structure = AseAtomsAdaptor.get_structure(final_atoms)
                cif_path = os.path.join(out_dir, "relaxed_structure.cif")
                structure.to(filename=cif_path)

                final_energy = final_atoms.get_potential_energy()
                with open(os.path.join(out_dir, "relaxed_energy.txt"), "w") as f:
                    f.write(f"{final_energy}\n")

                res_dict = {
                    "structure_name": struct_name,
                    "status": "success",
                    "energy": final_energy,
                    "cif_path": cif_path,
                    "output_dir": out_dir,
                }
                if traj_file is not None:
                    res_dict["trajectory_path"] = traj_file
                if log_file is not None:
                    res_dict["log_path"] = log_file

                results.append(res_dict)
            except Exception as e:
                results.append(
                    {
                        "structure_name": struct_name,
                        "status": "failed",
                        "error": str(e),
                        "output_dir": out_dir,
                    }
                )
    elif mode == "md":
        if memory_sink is not None:
            stored_batch = memory_sink.read()
            data_list = stored_batch.to_data_list()
            n_total_frames = len(data_list)
            n_structures = len(structure_names)
            n_snapshots = n_total_frames // n_structures
        else:
            data_list = final_batch.to_data_list()
            n_structures = len(structure_names)
            n_snapshots = 1

        for i in range(n_structures):
            struct_name = structure_names[i]
            out_dir = output_dirs[i]
            os.makedirs(out_dir, exist_ok=True)
            try:
                struct_atoms_list = []
                if memory_sink is not None:
                    for s in range(n_snapshots):
                        idx = s * n_structures + i
                        if idx < n_total_frames:
                            data = data_list[idx]
                            atoms = atomic_data_to_atoms(data)
                            velocities_tensor = getattr(data, "velocities", None)
                            if velocities_tensor is not None:
                                atoms.set_velocities(
                                    velocities_tensor.detach().cpu().numpy()
                                )
                            struct_atoms_list.append(atoms)
                else:
                    data = data_list[i]
                    atoms = atomic_data_to_atoms(data)
                    velocities_tensor = getattr(data, "velocities", None)
                    if velocities_tensor is not None:
                        atoms.set_velocities(velocities_tensor.detach().cpu().numpy())
                    struct_atoms_list.append(atoms)

                if not struct_atoms_list:
                    raise RuntimeError("No trajectory snapshots recorded.")

                formula = struct_atoms_list[-1].get_chemical_formula()
                filename_base = f"{formula}_{temperature}K_{ensemble}"

                traj_file = None
                log_file = None
                if memory_sink is not None:
                    traj_file = os.path.join(out_dir, f"{filename_base}.traj")
                    log_file = os.path.join(out_dir, f"{filename_base}.log")
                    write(traj_file, struct_atoms_list, format="traj")

                    with open(log_file, "w") as f:
                        f.write(
                            "Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n"
                        )
                        for step_idx, atoms in enumerate(struct_atoms_list):
                            time_ps = (step_idx * log_interval * timestep) / 1000.0
                            ep = atoms.get_potential_energy()
                            ek = atoms.get_kinetic_energy()
                            etot = ep + ek
                            temp = atoms.get_temperature()
                            f.write(
                                f"{time_ps:.4f}      {etot:11.4f}  {ep:11.4f}  {ek:11.4f}   {temp:5.1f}\n"
                            )

                final_atoms = struct_atoms_list[-1]
                structure = AseAtomsAdaptor.get_structure(final_atoms)
                cif_path = os.path.join(out_dir, "final_structure.cif")
                structure.to(filename=cif_path)

                final_energy = final_atoms.get_potential_energy()
                with open(os.path.join(out_dir, "energy.txt"), "w") as f:
                    f.write(f"{final_energy}\n")

                res_dict = {
                    "structure_name": struct_name,
                    "status": "success",
                    "cif_path": cif_path,
                    "output_dir": out_dir,
                }
                if traj_file is not None:
                    res_dict["trajectory_path"] = traj_file
                if log_file is not None:
                    res_dict["log_path"] = log_file

                results.append(res_dict)
            except Exception as e:
                results.append(
                    {
                        "structure_name": struct_name,
                        "status": "failed",
                        "error": str(e),
                        "output_dir": out_dir,
                    }
                )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return results


# ---------------------------------------------------------------------------
# Inflight batching helpers
# ---------------------------------------------------------------------------

# Per-architecture bytes-per-param-per-atom factors.
# Substring matched against every class name in the model's MRO (see _detect_bytes_per_atom).
# Calibrated for NValchemi FIRE relaxation (torch.no_grad(), no gradient buffers).
#
# The dominant memory drivers are:
#   - edge count/atom  ∝ r_cut³ × density
#   - feature width / tensor order
# Hence models with long cutoffs (FairChem 12 Å) use far more memory per atom than
# short-cutoff models (MACE ~5 Å) despite having more parameters.
_ARCH_BYTES_PER_PARAM_ATOM: list[tuple[str, float]] = [
    # FairChem eSEN/UMA: 12 Å cutoff, ~400 edges/atom; attention weights shared → lower factor.
    ("FairChemWrapper", 0.15),
    # MACE equivariant GNN: ~5 Å cutoff, ~30 edges/atom; L=2 tensor features → moderate factor.
    # Matches NVMACEWrapper and the dynamically-created HeadAwareMACEWrapper subclass.
    ("MACEWrapper", 0.5),
    # MatGL TensorNet: compact model, short cutoff.
    ("TensorNetWrapper", 1.5),
    # MatGL M3GNet / CHGNet: very few params, per-atom overhead is relatively large.
    ("M3GNetWrapper", 4.0),
    ("CHGNetWrapper", 3.0),
]
_FALLBACK_BYTES_PER_ATOM = 5_000_000  # 5 MB/atom for unrecognised architectures


def _detect_bytes_per_atom(model: Any) -> int:
    """Estimate GPU bytes per atom for a NValchemi model.

    Walks the model's class MRO looking for a known architecture pattern,
    then computes ``bytes_per_atom = num_params × factor``.  Returns the
    fallback (5 MB/atom) for unrecognised classes.
    """
    for klass in type(model).__mro__:
        name = klass.__name__
        for pattern, factor in _ARCH_BYTES_PER_PARAM_ATOM:
            if pattern in name:
                num_params = sum(p.numel() for p in model.parameters())
                return max(500_000, int(num_params * factor))
    return _FALLBACK_BYTES_PER_ATOM


def estimate_max_batch_atoms(
    device: str = "cuda",
    safety_factor: float = 0.5,
    model: Optional[Any] = None,
) -> int:
    """Estimate max total atoms for a single fixed GPU batch from free VRAM.

    Used as the threshold for switching from fixed-batch to inflight mode:
    if the total atom count across all structures exceeds this, inflight
    batching is used so only a subset occupies GPU memory at any time.

    Parameters
    ----------
    device : str
        Torch device string. Non-CUDA devices return a conservative fallback.
    safety_factor : float
        Fraction of free VRAM to budget for the live batch. Default 0.5.
    model : nn.Module, optional
        The NValchemi model object.  When provided, ``_detect_bytes_per_atom``
        inspects the class hierarchy to pick a calibrated bytes/atom estimate
        rather than the generic fallback.

    Returns
    -------
    int
        Estimated atom budget. Minimum 200.
    """
    if not NVALCHEMI_AVAILABLE:
        return 2000
    if not (torch.cuda.is_available() and device not in ("cpu", "mps")):
        return 2000

    free_bytes, _ = torch.cuda.mem_get_info()
    bytes_per_atom = (
        _detect_bytes_per_atom(model) if model is not None else _FALLBACK_BYTES_PER_ATOM
    )
    return max(200, int(free_bytes * safety_factor / bytes_per_atom))


class AtomsDataset:
    """Dataset adapter wrapping pre-loaded ASE Atoms for SizeAwareSampler.

    SizeAwareSampler requires three methods:
      * ``__len__()``
      * ``get_metadata(idx) -> (num_atoms, num_edges)``  — must be O(1)
      * ``__getitem__(idx) -> (AtomicData, dict)``

    ``get_metadata`` is called for every sample at sampler construction to
    build atom-count bins for bin-packing.  Sizes are pre-computed at init
    so the call is a trivial list lookup.

    Parameters
    ----------
    atoms_list : list[Atoms]
        Pre-loaded ASE Atoms objects.
    names : list[str]
        Corresponding structure names (same order as atoms_list).
    device : str
        Target torch device for AtomicData conversion.
    dtype : torch.dtype, optional
        Floating-point dtype. Defaults to torch.float32.
    """

    def __init__(
        self,
        atoms_list: list,
        names: list[str],
        device: str = "cuda",
        dtype: Any = None,
        relax_cell: bool = False,
    ) -> None:
        self._atoms = atoms_list
        self._names = names
        self._device = device
        self._dtype = dtype
        self._relax_cell = relax_cell
        # Wrap coordinates inside unit cell boundary to prevent OOM from huge neighbor list image creation
        for a in self._atoms:
            if hasattr(a, "pbc") and any(a.pbc):
                a.wrap()
        # Pre-computed for O(1) get_metadata — no tensor allocation at query time.
        self._sizes: list[tuple[int, int]] = [(len(a), 0) for a in atoms_list]

    def __len__(self) -> int:
        return len(self._atoms)

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` without constructing AtomicData."""
        return self._sizes[idx]

    def __getitem__(self, idx: int) -> tuple:
        """Return ``(AtomicData, metadata_dict)`` for sample *idx*."""
        if not NVALCHEMI_AVAILABLE:
            raise ImportError("nvalchemi-toolkit is required.")
        dtype = self._dtype if self._dtype is not None else torch.float32
        data = AtomicData.from_atoms(self._atoms[idx], device=self._device, dtype=dtype)
        # Pre-allocate model output tensors as zeros.  SegmentedLevelStorage.concatenate()
        # only keeps keys common to both batches, so replacements added during inflight
        # refill must carry the same keys as the live batch or forces/energy will be dropped.
        n_atoms = len(self._atoms[idx])
        data["forces"] = torch.zeros(n_atoms, 3, device=self._device, dtype=dtype)
        # Energy shape (1, 1) satisfies AtomicData pydantic validation Float[Tensor, 'B 1']
        data["energy"] = torch.zeros(1, 1, device=self._device, dtype=dtype)
        if self._relax_cell:
            data["stress"] = torch.zeros(1, 3, 3, device=self._device, dtype=dtype)
        # Register orig_idx as a system-level property so it survives from_data_list/
        # to_data_list round-trips via the system_group.  _bookkeeping_keys only
        # overwrites "status" and "system_id", so orig_idx tracks the original input
        # index through refill_check even when n_remaining=0 resets all system_ids.
        data.add_system_property(
            "orig_idx",
            torch.tensor([[idx]], dtype=torch.long, device=self._device),
        )
        return data, {"name": self._names[idx], "index": idx}


class HostMemoryWithSystemId(HostMemory):
    """HostMemory that preserves ``system_id`` across the unbatch/rebatch round-trip.

    When FusedStage writes graduated structures to sinks via
    ``_overflow_to_sinks``, the batch is unbatched with ``to_data_list()``.
    During reconstruction the ``__system_keys__`` registration on each
    ``AtomicData`` is reset to defaults, so ``system_id`` (stamped by
    ``SizeAwareSampler``) is silently dropped.  This subclass extracts
    ``system_id`` *before* unbatching and re-registers it on each individual
    ``AtomicData`` so that it survives ``read()`` / ``drain()``.

    Parameters
    ----------
    capacity : int
        Maximum number of graduated samples to buffer in CPU memory.
    on_graduate : callable(orig_idx: int, data_cpu: AtomicData) -> None, optional
        Callback fired immediately when each structure graduates from the live
        batch.  Receives the dataset index (``orig_idx``) and the structure's
        ``AtomicData`` already moved to CPU.  Use this to write results to disk
        incrementally so partial results survive an OOM abort.
    """

    def __init__(
        self,
        capacity: int,
        on_graduate: Optional[Callable[[int, Any], None]] = None,
    ) -> None:
        super().__init__(capacity)
        self._on_graduate = on_graduate

    def write(self, batch: Any, mask: Any = None) -> None:  # type: ignore[override]
        if not NVALCHEMI_AVAILABLE:
            raise ImportError("nvalchemi-toolkit is required.")

        num_total = batch.num_graphs or 0
        if num_total == 0:
            return

        if mask is not None:
            mask = mask.to(device=batch.device, dtype=torch.bool)
            if mask.shape[0] != num_total:
                raise ValueError(
                    f"mask length {mask.shape[0]} != num_graphs {num_total}"
                )
            num_selected = int(mask.sum().item())
            if num_selected == 0:
                return
            if num_selected < num_total:
                indices = torch.nonzero(mask, as_tuple=True)[0]
                _ = batch.batch_ptr  # trigger lazy init for SegmentedLevelStorage
                batch = batch.index_select(indices)

        # Normalise system tensors before to_data_list() to satisfy AtomicData validators.
        # `batch` is already a local copy from index_select above, so in-place
        # modification is safe and does not affect the live batch.
        # - Cast float64 → float32 (FairChem and similar models output float64 stress).
        # - Reshape stress (B, 9) → (B, 3, 3) if a model outputs Voigt notation.
        sys_group = batch._system_group
        if sys_group is not None:
            for key in list(sys_group.keys()):
                t = sys_group[key]
                if not isinstance(t, torch.Tensor):
                    continue
                if t.dtype == torch.float64:
                    t = t.float()
                if key == "stress" and t.dim() == 2 and t.shape[-1] == 9:
                    t = t.reshape(t.shape[0], 3, 3)
                sys_group[key] = t

        # Extract system_id before to_data_list() drops it.
        system_ids = batch["system_id"].detach().to("cpu")
        data_list = batch.to_data_list()

        if len(self._data_list) + len(data_list) > self._capacity:
            raise RuntimeError(
                f"HostMemoryWithSystemId is full: cannot add {len(data_list)} "
                f"samples to buffer with {len(self._data_list)}/{self._capacity}."
            )

        for i, data in enumerate(data_list):
            data_cpu = data.to(self._device)
            data_cpu.add_system_property("system_id", system_ids[i : i + 1])
            self._data_list.append(data_cpu)
            if self._on_graduate is not None:
                orig_idx = int(data_cpu["orig_idx"].squeeze().item())
                self._on_graduate(orig_idx, data_cpu)

        # Explicitly release the GPU data_list and return cached CUDA allocator
        # blocks back to the driver.  On DGX Spark's unified memory pool this
        # prevents the allocator from silently consuming the entire pool over
        # thousands of FIRE steps (each adapt_input builds large temporary graphs).
        del data_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RelaxLogHook:
    """ASE-style per-structure relax.log writer for inflight batch relaxation.

    Writes ``{output_dir}/relax.log`` for each structure in the same format
    as the fixed-batch NValchemi path, enabling step-by-step monitoring of
    relaxations running inside the FusedStage engine.

    Implements the NValchemi Hook protocol (``frequency``, ``stage``,
    ``__call__``) and the Python context-manager protocol.  FusedStage calls
    ``__enter__``/``__exit__`` automatically via ``_open_hooks``/
    ``_close_hooks``, so file handles are flushed and closed without extra
    bookkeeping in ``_batch_relax_nvalchemi_inflight``.

    Parameters
    ----------
    output_dirs : list[str]
        Per-structure output directories indexed by ``orig_idx``.
    stage : DynamicsStage
        Hook stage — pass ``DynamicsStage.AFTER_COMPUTE``.
    """

    frequency: int = 1

    def __init__(self, output_dirs: list, stage: Any) -> None:
        self.stage = stage
        self._output_dirs = output_dirs
        self._step_counts: dict[int, int] = {}
        self._file_handles: dict[int, Any] = {}

    def __call__(self, ctx: Any, stage: Any) -> None:
        if not NVALCHEMI_AVAILABLE:
            return
        batch = ctx.batch
        n_graphs = batch.num_graphs
        if n_graphs == 0:
            return

        # Per-graph orig_idx and energy (both system-level, shape [B, 1])
        orig_idxs = batch["orig_idx"].squeeze(-1).tolist()
        energies = batch["energy"].squeeze(-1).tolist()

        # Per-graph max force norm via scatter_reduce_ over atom-level forces
        force_norms = batch["forces"].norm(dim=-1)  # (N_total,)
        batch_idx = batch.batch_idx  # (N_total,) — atom→graph mapping
        fmax_per_graph = torch.zeros(n_graphs, device=force_norms.device)
        fmax_per_graph.scatter_reduce_(
            0, batch_idx, force_norms, reduce="amax", include_self=False
        )
        fmax_list = fmax_per_graph.tolist()

        for g in range(n_graphs):
            oidx = int(orig_idxs[g])
            energy = float(energies[g])
            fmax_val = float(fmax_list[g])
            step = self._step_counts.get(oidx, 0)

            if oidx not in self._file_handles:
                out_dir = self._output_dirs[oidx]
                os.makedirs(out_dir, exist_ok=True)
                fh = open(os.path.join(out_dir, "relax.log"), "w")
                fh.write("           Step          Energy         fmax\n")
                self._file_handles[oidx] = fh

            self._file_handles[oidx].write(
                f"FIRE:  {step:9d}  {energy:14.6f}  {fmax_val:12.6f}\n"
            )
            self._file_handles[oidx].flush()
            self._step_counts[oidx] = step + 1

    def __enter__(self) -> "RelaxLogHook":
        return self

    def __exit__(self, *args: Any) -> None:
        for fh in self._file_handles.values():
            fh.close()
        self._file_handles.clear()


class PositionWrappingHook:
    """Hook to wrap batch positions back into the unit cell at BEFORE_COMPUTE.

    Prevents graph/neighbor list generation OOMs when coordinates explode.
    """

    frequency: int = 1

    def __init__(self, stage: Any) -> None:
        self.stage = stage

    def __call__(self, ctx: Any, stage: Any) -> None:
        if not NVALCHEMI_AVAILABLE:
            return
        batch = ctx.batch
        if batch is None or batch.num_graphs == 0:
            return

        if not hasattr(batch, "cell") or batch.cell is None:
            return

        if batch.cell.abs().sum() == 0:
            return

        batch_idx = batch.batch_idx.long()
        cells_inv = torch.linalg.inv(batch.cell)
        cells_inv_per_atom = cells_inv[batch_idx]

        frac = torch.bmm(batch.positions.unsqueeze(1), cells_inv_per_atom).squeeze(1)
        frac = frac - torch.floor(frac)

        cells_per_atom = batch.cell[batch_idx]
        wrapped_pos = torch.bmm(frac.unsqueeze(1), cells_per_atom).squeeze(1)
        batch.positions.copy_(wrapped_pos)


class ForceStressClippingHook:
    """Hook to cap huge forces and stresses at AFTER_COMPUTE.

    Prevents dynamics from exploding in variable-cell/atomic relaxation
    when starting configurations are highly unstable.
    """

    frequency: int = 1

    def __init__(
        self, stage: Any, max_force: float = 5.0, max_stress: float = 5.0
    ) -> None:
        self.stage = stage
        self.max_force = max_force
        self.max_stress = max_stress

    def __call__(self, ctx: Any, stage: Any) -> None:
        if not NVALCHEMI_AVAILABLE:
            return
        batch = ctx.batch
        if batch is None or batch.num_graphs == 0:
            return

        if hasattr(batch, "forces") and batch.forces is not None:
            forces = batch.forces
            force_norms = torch.norm(forces, dim=-1, keepdim=True)
            scale = torch.clamp(self.max_force / (force_norms + 1e-8), max=1.0)
            batch.forces.copy_(forces * scale)

        if hasattr(batch, "stress") and batch.stress is not None:
            batch.stress.copy_(
                torch.clamp(batch.stress, min=-self.max_stress, max=self.max_stress)
            )


class CellForceCriterion(_ConvergenceCriterion):
    """Variable-cell convergence test equal to ASE's ``FrechetCellFilter``.

    ``ConvergenceHook.from_fmax`` tests atomic forces only, so a variable-cell
    relaxation of a crystal whose forces vanish by symmetry "converges" at step
    0 with its cell untouched. ASE folds the cell into ``fmax`` as the rows of
    the virial divided by the number of atoms -- the scaling
    :class:`ScaledFIRE2VariableCell` applies to the cell force -- and this
    criterion applies the same test to every system in the batch.
    """

    key: str = "stress"

    def __call__(self, batch: Any) -> "torch.Tensor":
        stress = batch.stress.view(-1, 3, 3)
        volume = torch.linalg.det(batch.cell).abs()
        n_atoms = torch.clamp(batch.num_nodes_per_graph.to(stress.dtype), min=1.0)
        virial = stress * (volume / n_atoms).view(-1, 1, 1)
        row_norm = torch.linalg.vector_norm(virial, dim=-1)
        return row_norm.amax(dim=-1) <= self.threshold


def relax_convergence_hook(fmax: float, relax_cell: bool) -> "ConvergenceHook":
    """Return the convergence hook of a batched FIRE relaxation.

    Atomic forces must fall to ``fmax`` (eV/Å); with ``relax_cell`` the cell
    force must as well (see :class:`CellForceCriterion`). Converged systems move
    from status 0 to status 1.

    Args:
        fmax: Force threshold in eV/Å, applied to atoms and cell alike.
        relax_cell: Whether the cell is a degree of freedom.

    Returns:
        A ConvergenceHook to pass as ``convergence_hook`` to the optimizer.
    """
    hook = ConvergenceHook.from_fmax(threshold=fmax, source_status=0, target_status=1)
    if relax_cell:
        hook.criteria.append(CellForceCriterion(threshold=fmax))
    return hook


class ScaledFIRE2VariableCell(FIRE2VariableCell):
    """FIRE2 variable-cell optimizer with cell force and velocity scaling.

    Prevents cell explosion by scaling cell degrees of freedom by a factor
    proportional to the number of atoms in the system, matching the behavior
    of ASE's UnitCellFilter.
    """

    def pre_update(self, batch: Any) -> None:
        import torch
        from nvalchemi.dynamics.optimizers.fire2 import fire2_step_coord_cell
        from nvalchemi.dynamics._ops.npt_nph import stress_to_cell_force

        # 1. Compute cell force from stress
        volumes = torch.linalg.det(batch.cell).abs()
        stress_sigma = batch.stress
        cell_force = stress_to_cell_force(stress_sigma, batch.cell, volumes)

        # 2. Compute per-system cell factor f = number of atoms
        num_atoms_per_system = torch.zeros(
            batch.num_graphs, dtype=torch.int32, device=batch.device
        )
        num_atoms_per_system.scatter_add_(
            0,
            batch.batch_idx.long(),
            torch.ones_like(batch.batch_idx, dtype=torch.int32),
        )
        f = num_atoms_per_system.to(batch.positions.dtype)
        f = torch.clamp(f, min=1.0).view(-1, 1, 1)

        # 3. Scale cell, cell_velocities, and cell_force
        batch.cell.copy_(batch.cell * f)
        self._state.cell_velocities.copy_(self._state.cell_velocities * f)
        cell_force_scaled = cell_force / f

        # 4. Call the original step function
        fire2_step_coord_cell(
            batch.positions.detach(),
            batch.velocities,
            batch.forces,
            batch.cell.detach(),
            self._state.cell_velocities,
            cell_force_scaled,
            batch.batch_idx.int(),
            self._state.alpha,
            self._state.dt,
            self._state.nsteps_inc,
            vf=self._state.vf,
            v_sumsq=self._state.v_sumsq,
            f_sumsq=self._state.f_sumsq,
            delaystep=self.delaystep,
            dtgrow=self.dtgrow,
            dtshrink=self.dtshrink,
            alphashrink=self.alphashrink,
            alpha0=self.alpha0,
            tmax=self.tmax,
            tmin=self.tmin,
            maxstep=self.maxstep,
        )

        # 5. Unscale cell and cell_velocities
        batch.cell.copy_(batch.cell / f)
        self._state.cell_velocities.copy_(self._state.cell_velocities / f)
