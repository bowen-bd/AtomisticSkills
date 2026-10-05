"""NValchemi model factory for MatGL (TensorNet, M3GNet, CHGNet)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

if TYPE_CHECKING:
    from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

logger = logging.getLogger(__name__)


def get_nvalchemi_matgl_model(wrapper: "MatGLWrapper") -> Optional[Any]:
    """Return a NValchemi-compatible wrapper for the loaded MatGL model.

    Dispatches to the appropriate NValchemi wrapper class based on the
    inner model type:
      - TensorNet  → matgl.ext.alchmtk.TensorNetWrapper
      - M3GNet     → matgl.ext.alchmtk.M3GNetWrapper
      - CHGNet     → matgl.ext.alchmtk.CHGNetWrapper

    Returns None if nvalchemi is unavailable, the wrapper is not loaded,
    or the inner model type is not supported.
    """
    if not NVALCHEMI_AVAILABLE:
        return None
    if not wrapper.is_loaded or wrapper.model is None:
        return None

    # Check cached instance first
    if getattr(wrapper, "_nv_model", None) is not None:
        return wrapper._nv_model

    try:
        from matgl.models._tensornet import TensorNet
        from matgl.models._m3gnet import M3GNet
        from matgl.models._chgnet import CHGNet
        from matgl.ext.alchmtk import TensorNetWrapper
        from src.utils.mlips.nvalchemi.matgl_wrappers import (
            M3GNetWrapper,
            CHGNetWrapper,
        )

        inner = wrapper.model.model  # Potential.model

        if isinstance(inner, TensorNet):
            nv_model = TensorNetWrapper.from_potential(wrapper.model)
            # Inflight batching gives wrong energies (see M3GNetWrapper).
            nv_model._nvalchemi_supports_inflight = False
        elif isinstance(inner, M3GNet):
            nv_model = M3GNetWrapper.from_potential(wrapper.model)
        elif isinstance(inner, CHGNet):
            nv_model = CHGNetWrapper.from_potential(wrapper.model)
        else:
            logger.warning(
                f"MatGL inner model type {type(inner).__name__} not supported by NValchemi."
            )
            return None

        device = getattr(wrapper, "device", "cpu")
        nv_model = nv_model.to(device)
        wrapper._nv_model = nv_model  # cache
        return nv_model

    except Exception as e:
        logger.warning(f"Failed to build NValchemi MatGL model: {e}")
        return None
