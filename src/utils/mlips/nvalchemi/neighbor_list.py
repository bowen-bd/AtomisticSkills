"""Geometry-cache compatibility for the locked NValchemi 0.2 runtime."""

from enum import Enum
from importlib.metadata import version
from typing import Any

import torch
from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.base import NeighborConfig
from packaging.version import Version


class _GeometryAwareNeighborListHook(NeighborListHook):
    """Invalidate 0.2.x allocations when geometry changes at constant N/B.

    Upstream caches periodic-image ranges, cell-list sizes and neighbor capacity
    by total atoms and graph count. Value comparisons also detect in-place Warp
    cell updates, which do not increment PyTorch tensor version counters.
    """

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Keep ordinary fixed-cell buffers; rebuild changed geometry fully."""
        batch = ctx.batch
        if self._alloc_N is not None:
            for cached, current in (
                (self._buf_cell, getattr(batch, "cell", None)),
                (self._buf_pbc, getattr(batch, "pbc", None)),
                (self._buf_batch_ptr, batch.batch_ptr),
            ):
                changed = (
                    (cached is None) != (current is None)
                    if cached is None or current is None
                    else not torch.equal(cached, current)
                )
                if changed:
                    # Explicit 0.2.x compatibility guard; no installed-package
                    # mutation. The upstream issue is not filed yet; see the
                    # reproduction in docs/verification/nvalchemi-neighbor-cache.md.
                    # These sentinels request upstream's complete first-build
                    # allocation path, including adaptive K and Verlet state.
                    self._alloc_N = None
                    self._neighbor_matrix = None
                    self._neighbor_matrix_shifts = None
                    break
        super().__call__(ctx, stage)


def make_neighbor_list_hook(config: NeighborConfig, **kwargs: Any) -> NeighborListHook:
    """Apply the geometry guard only to the affected toolkit 0.2.x family.

    Revalidate and retire this guard when locking a corrected upstream release.
    Other versions use the upstream hook unchanged; no monkeypatch is installed.
    """
    hook_type = (
        _GeometryAwareNeighborListHook
        if Version(version("nvalchemi-toolkit")).release[:2] == (0, 2)
        else NeighborListHook
    )
    return hook_type(config, **kwargs)
