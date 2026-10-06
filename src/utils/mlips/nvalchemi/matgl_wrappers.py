"""NValchemi BaseModelMixin wrappers for MatGL M3GNet and CHGNet models.

These follow the same structure as ``matgl.ext.alchmtk.TensorNetWrapper``
(for TensorNet) but target M3GNet and CHGNet, which both build their
line/bond graphs internally during the forward pass.

Usage
-----
    import matgl
    from src.utils.mlips.nvalchemi.matgl_wrappers import M3GNetWrapper, CHGNetWrapper

    potential = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    model = M3GNetWrapper.from_potential(potential)

Notes
-----
* Energy is the primitive output; forces and stresses are autograd.
* Requires :class:`~nvalchemi.hooks.NeighborListHook` with
  ``format=NeighborListFormat.COO`` to populate ``neighbor_list`` and
  ``neighbor_list_shifts`` before each model call.
* Only ``is_intensive=False`` (total-energy) models are supported.
* CHGNet additionally needs integer image shifts (``neighbor_list_shifts``)
  because ``create_directed_line_graph`` uses them to disambiguate periodic
  images.  These are passed as ``g.pbc_offset`` into CHGNet's forward.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import torch
from pymatgen.core.periodic_table import Element
from torch import nn

try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.models._utils import (
        autograd_forces,
        autograd_stresses,
        prepare_strain,
    )
    from nvalchemi.models.base import (
        BaseModelMixin,
        ModelConfig,
        NeighborConfig,
        NeighborListFormat,
    )

    _NVALCHEMI_AVAILABLE = True
except ImportError:
    _NVALCHEMI_AVAILABLE = False
    BaseModelMixin = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from nvalchemi._typing import ModelOutputs

    from matgl.apps._pes import Potential

__all__ = ["M3GNetWrapper", "CHGNetWrapper"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_z_to_type_table(element_types: tuple[str, ...]) -> torch.Tensor:
    """Return a ``[max_z + 1]`` lookup: atomic number -> type index."""
    symbol_to_idx = {sym: i for i, sym in enumerate(element_types)}
    max_z = max(Element(sym).Z for sym in element_types)
    table = torch.full((max_z + 1,), -1, dtype=torch.long)
    for sym, idx in symbol_to_idx.items():
        table[Element(sym).Z] = idx
    return table


def _adapt_input_core(
    data: Any,
    z_to_type: torch.Tensor,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    """Shared adapt_input logic for all PyG-based MatGL models.

    Mirrors ``TensorNetWrapper.adapt_input()`` exactly so that M3GNet and
    CHGNet receive the same edge and cell representations.
    """
    if isinstance(data, AtomicData):
        data = Batch.from_data_list([data])

    device = data.positions.device
    B: int = data.num_graphs

    node_type = z_to_type[data.atomic_numbers]

    edge_index = data.neighbor_list.T  # [2, E]
    E: int = edge_index.shape[1]

    shifts_raw = getattr(data, "neighbor_list_shifts", None)
    if shifts_raw is None:
        shifts = torch.zeros(E, 3, dtype=model_dtype, device=device)
    else:
        shifts = shifts_raw.to(dtype=model_dtype, device=device)

    cell_raw = getattr(data, "cell", None)
    if cell_raw is None:
        cell = (
            torch.eye(3, dtype=model_dtype, device=device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
    else:
        cell = cell_raw.to(dtype=model_dtype, device=device)

    positions = data.positions.to(dtype=model_dtype)
    edge_batch = data.batch_idx[edge_index[0]]
    pbc_offshift = torch.einsum("eb,ebc->ec", shifts, cell[edge_batch])

    return {
        "node_type": node_type,
        "pos": positions,
        "edge_index": edge_index,
        "pbc_offshift": pbc_offshift,
        "pbc_offset": shifts,  # raw integer image shifts (needed by CHGNet)
        "batch": data.batch_idx,
        "num_graphs": B,
    }


def _forward_core(
    self: Any,
    data: Any,
    extra_g_fields: dict[str, Any] | None = None,
) -> ModelOutputs:
    """Shared forward pass for M3GNet and CHGNet wrappers.

    Both models accept a SimpleNamespace ``g`` and compute bond vectors /
    line graphs internally, so the forward logic is identical apart from
    optional extra fields on ``g`` (e.g. CHGNet's ``pbc_offset``).
    """
    if isinstance(data, AtomicData):
        data = Batch.from_data_list([data])

    compute_forces = "forces" in (
        self.model_config.active_outputs & self.model_config.outputs
    )
    compute_stresses = "stress" in (
        self.model_config.active_outputs & self.model_config.outputs
    )

    displacement = None
    orig_positions = None
    orig_cell = None
    if compute_stresses and hasattr(data, "cell") and data.cell is not None:
        scaled_pos, scaled_cell, displacement = prepare_strain(
            data.positions, data.cell, data.batch_idx
        )
        orig_positions = data.positions
        orig_cell = data.cell
        data["positions"] = scaled_pos
        data["cell"] = scaled_cell

    if compute_forces or compute_stresses:
        pos = data.positions.clone()
        pos.requires_grad_(True)
        data["positions"] = pos

    inputs = self.adapt_input(data)

    g = types.SimpleNamespace(
        node_type=inputs["node_type"],
        pos=inputs["pos"],
        edge_index=inputs["edge_index"],
        pbc_offshift=inputs["pbc_offshift"],
        batch=inputs["batch"],
        num_graphs=inputs["num_graphs"],
    )
    if extra_g_fields:
        for k, v in extra_g_fields(inputs).items():
            setattr(g, k, v)

    raw_energy = self.model(g=g)
    total_energy = self.data_std * raw_energy + self.data_mean

    if self._element_ref_offset is not None:
        atomic_offset = self._element_ref_offset[inputs["node_type"]]
        graph_offset = torch.zeros(
            inputs["num_graphs"], device=atomic_offset.device, dtype=atomic_offset.dtype
        )
        graph_offset.scatter_add_(0, inputs["batch"], atomic_offset)
        total_energy = total_energy + graph_offset

    if total_energy.dim() == 0:
        total_energy = total_energy.unsqueeze(0)
    if total_energy.dim() == 1:
        total_energy = total_energy.unsqueeze(-1)

    result: dict[str, torch.Tensor | None] = {"energy": total_energy}

    if compute_forces:
        result["forces"] = autograd_forces(
            total_energy,
            data.positions,
            training=False,
            retain_graph=compute_stresses,
        )

    if compute_stresses and displacement is not None:
        result["stress"] = autograd_stresses(
            total_energy,
            displacement,
            orig_cell,
            data.num_graphs,
        )

    if orig_positions is not None:
        data["positions"] = orig_positions
        data["cell"] = orig_cell

    return self.adapt_output(result, data)


# ---------------------------------------------------------------------------
# M3GNetWrapper
# ---------------------------------------------------------------------------


class M3GNetWrapper(nn.Module, BaseModelMixin):  # type: ignore[misc]
    """NValchemi wrapper for M3GNet (PyG backend).

    Parameters
    ----------
    model : M3GNet
        An instantiated M3GNet model.  Must have ``is_intensive=False``.
    data_mean : float or torch.Tensor, optional
        Training-target mean for energy un-normalization.  Default: 0.0.
    data_std : float or torch.Tensor, optional
        Training-target standard deviation.  Default: 1.0.
    element_refs : torch.Tensor or None, optional
        Per-element-type energy offsets, shape ``[num_element_types]``.
        Added to the total energy after un-normalization.  Default: None.
    """

    def __init__(
        self,
        model: Any,
        data_mean: float | torch.Tensor = 0.0,
        data_std: float | torch.Tensor = 1.0,
        element_refs: torch.Tensor | None = None,
    ) -> None:
        if not _NVALCHEMI_AVAILABLE:
            raise ImportError(
                "nvalchemi-toolkit is required to use M3GNetWrapper. "
                "Install with: pip install nvalchemi-toolkit"
            )
        super().__init__()

        if model.is_intensive:
            raise ValueError(
                "M3GNetWrapper requires is_intensive=False (total-energy prediction)."
            )

        self.model = model

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces", "stress"},
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"neighbor_list_shifts", "cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=float(self.model.cutoff),
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, dtype=torch.float32)
        if not isinstance(data_std, torch.Tensor):
            data_std = torch.tensor(data_std, dtype=torch.float32)
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=torch.float32)
            self.register_buffer("_element_ref_offset", element_refs)
        else:
            self._element_ref_offset: torch.Tensor | None = None

        self.register_buffer(
            "_z_to_type",
            _build_z_to_type_table(self.model.element_types),
            persistent=False,
        )

        # Inflight batching gives wrong energies with this wrapper: structures
        # that enter the batch after a graduation relax to energies up to eV
        # away from the fixed-batch and sequential results (matgl 4.1,
        # nvalchemi 0.2; MACE inflight agrees to meV). Use fixed batches.
        self._nvalchemi_supports_inflight: bool = False

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        units: int = self.model.units
        return {
            "node_embeddings": (units,),
            "graph_embeddings": (units,),
        }

    @classmethod
    def from_potential(cls, potential: Potential) -> M3GNetWrapper:
        """Construct from a matgl :class:`~matgl.apps._pes.Potential`."""
        from matgl.models._m3gnet import M3GNet

        if not isinstance(potential.model, M3GNet):
            raise TypeError(
                f"Expected potential.model to be M3GNet, got {type(potential.model).__name__}."
            )

        element_refs = None
        if potential.element_refs is not None:
            element_refs = potential.element_refs.property_offset.clone()

        data_mean = potential.data_mean
        data_std = potential.data_std
        assert isinstance(data_mean, torch.Tensor)
        assert isinstance(data_std, torch.Tensor)
        return cls(
            model=potential.model,
            data_mean=data_mean.clone(),
            data_std=data_std.clone(),
            element_refs=element_refs,
        )

    def adapt_input(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        return _adapt_input_core(data, self._z_to_type, self._model_dtype())

    def adapt_output(self, model_output: dict[str, Any], data: Any) -> ModelOutputs:
        output = super().adapt_output(model_output, data)
        output["energy"] = model_output["energy"]
        if model_output.get("forces") is not None:
            output["forces"] = model_output["forces"]
        if model_output.get("stress") is not None:
            output["stress"] = model_output["stress"]
        return output

    def forward(self, data: Any, **kwargs: Any) -> ModelOutputs:
        """Run M3GNet (builds line graph internally) and return energy/forces/stress."""
        return _forward_core(self, data, extra_g_fields=None)

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:
        """Compute node and graph embeddings (no forces/stress)."""
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        inputs = self.adapt_input(data)
        g = types.SimpleNamespace(
            node_type=inputs["node_type"],
            pos=inputs["pos"],
            edge_index=inputs["edge_index"],
            pbc_offshift=inputs["pbc_offshift"],
            batch=inputs["batch"],
            num_graphs=inputs["num_graphs"],
        )

        self.model(g=g)
        # Last M3GNet block stores per-atom node features
        last_block_key = f"gc_{self.model.n_blocks}"
        node_embeddings = self.model.feature_dict[last_block_key]["node_feat"]

        units = node_embeddings.shape[-1]
        graph_embeddings = torch.zeros(
            inputs["num_graphs"],
            units,
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        graph_embeddings.scatter_add_(
            0,
            inputs["batch"].unsqueeze(-1).expand(-1, units),
            node_embeddings,
        )

        data.node_embeddings = node_embeddings
        data.graph_embeddings = graph_embeddings
        return data


# ---------------------------------------------------------------------------
# CHGNetWrapper
# ---------------------------------------------------------------------------


class CHGNetWrapper(nn.Module, BaseModelMixin):  # type: ignore[misc]
    """NValchemi wrapper for CHGNet (PyG backend).

    CHGNet builds its directed line graph internally from ``g.pbc_offset``
    (integer image shifts).  These are passed from ``neighbor_list_shifts``
    populated by :class:`~nvalchemi.hooks.NeighborListHook`.

    Parameters
    ----------
    model : CHGNet
        An instantiated CHGNet model.  Must have ``is_intensive=False``.
    data_mean : float or torch.Tensor, optional
        Energy un-normalization mean.  Default: 0.0.
    data_std : float or torch.Tensor, optional
        Energy un-normalization std.  Default: 1.0.
    element_refs : torch.Tensor or None, optional
        Per-element-type energy offsets.  Default: None.
    """

    def __init__(
        self,
        model: Any,
        data_mean: float | torch.Tensor = 0.0,
        data_std: float | torch.Tensor = 1.0,
        element_refs: torch.Tensor | None = None,
    ) -> None:
        if not _NVALCHEMI_AVAILABLE:
            raise ImportError(
                "nvalchemi-toolkit is required to use CHGNetWrapper. "
                "Install with: pip install nvalchemi-toolkit"
            )
        super().__init__()

        if model.is_intensive:
            raise ValueError(
                "CHGNetWrapper requires is_intensive=False (total-energy prediction)."
            )

        self.model = model

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces", "stress"},
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"neighbor_list_shifts", "cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=float(self.model.cutoff),
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, dtype=torch.float32)
        if not isinstance(data_std, torch.Tensor):
            data_std = torch.tensor(data_std, dtype=torch.float32)
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=torch.float32)
            self.register_buffer("_element_ref_offset", element_refs)
        else:
            self._element_ref_offset: torch.Tensor | None = None

        self.register_buffer(
            "_z_to_type",
            _build_z_to_type_table(self.model.element_types),
            persistent=False,
        )

        # Inflight batching gives wrong energies with this wrapper: structures
        # that enter the batch after a graduation relax to energies up to eV
        # away from the fixed-batch and sequential results (matgl 4.1,
        # nvalchemi 0.2; MACE inflight agrees to meV). Use fixed batches.
        self._nvalchemi_supports_inflight: bool = False

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        dim = self.model.atom_embedding.embedding_dim
        return {
            "node_embeddings": (dim,),
            "graph_embeddings": (dim,),
        }

    @classmethod
    def from_potential(cls, potential: Potential) -> CHGNetWrapper:
        """Construct from a matgl :class:`~matgl.apps._pes.Potential`."""
        from matgl.models._chgnet import CHGNet

        if not isinstance(potential.model, CHGNet):
            raise TypeError(
                f"Expected potential.model to be CHGNet, got {type(potential.model).__name__}."
            )

        element_refs = None
        if potential.element_refs is not None:
            element_refs = potential.element_refs.property_offset.clone()

        data_mean = potential.data_mean
        data_std = potential.data_std
        assert isinstance(data_mean, torch.Tensor)
        assert isinstance(data_std, torch.Tensor)
        return cls(
            model=potential.model,
            data_mean=data_mean.clone(),
            data_std=data_std.clone(),
            element_refs=element_refs,
        )

    def adapt_input(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        return _adapt_input_core(data, self._z_to_type, self._model_dtype())

    def adapt_output(self, model_output: dict[str, Any], data: Any) -> ModelOutputs:
        output = super().adapt_output(model_output, data)
        output["energy"] = model_output["energy"]
        if model_output.get("forces") is not None:
            output["forces"] = model_output["forces"]
        if model_output.get("stress") is not None:
            output["stress"] = model_output["stress"]
        return output

    def forward(self, data: Any, **kwargs: Any) -> ModelOutputs:
        """Run CHGNet (builds directed line graph internally) and return energy/forces/stress.

        Passes ``g.pbc_offset`` = integer image shifts so that
        ``create_directed_line_graph`` can correctly identify periodic images.
        """
        return _forward_core(
            self,
            data,
            extra_g_fields=lambda inputs: {"pbc_offset": inputs["pbc_offset"]},
        )

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:
        """Compute node and graph embeddings (no forces/stress)."""
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        inputs = self.adapt_input(data)
        g = types.SimpleNamespace(
            node_type=inputs["node_type"],
            pos=inputs["pos"],
            edge_index=inputs["edge_index"],
            pbc_offshift=inputs["pbc_offshift"],
            pbc_offset=inputs["pbc_offset"],
            batch=inputs["batch"],
            num_graphs=inputs["num_graphs"],
        )

        self.model(g=g)
        last_block_key = f"gc_{self.model.n_blocks}"
        node_embeddings = self.model.feature_dict[last_block_key]["atom_feat"]

        dim = node_embeddings.shape[-1]
        graph_embeddings = torch.zeros(
            inputs["num_graphs"],
            dim,
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        graph_embeddings.scatter_add_(
            0,
            inputs["batch"].unsqueeze(-1).expand(-1, dim),
            node_embeddings,
        )

        data.node_embeddings = node_embeddings
        data.graph_embeddings = graph_embeddings
        return data
