import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_wrapper(
    model_type: str, model_name: Optional[str] = None, device: str = "auto", **kwargs
) -> Any:
    """
    Load the appropriate MLIP wrapper.

    Args:
        model_type: Type of model ('mace', 'fairchem', 'matgl')
        model_name: Specific model name (optional, uses default if None)
        device: Device to use ('auto', 'cpu', 'cuda')
        **kwargs: Additional model-specific arguments like `task_name` or `model_head`.
                  These are passed appropriately to the underlying wrappers.

    Returns:
        Loaded MLIP wrapper instance
    """
    model_type = model_type.lower()

    # Resolve aliases for task/head for multi-head models
    task = (
        kwargs.get("task_name")
        or kwargs.get("model_head")
        or kwargs.get("task")
        or kwargs.get("head")
    )

    # Without a model name each wrapper uses its own default; passing None on
    # would override it (and crash the MACE wrapper).
    named = {"model_name": model_name} if model_name else {}

    if model_type == "mace":
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(device=device, head=task, **named)
    elif model_type == "fairchem":
        from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper

        wrapper = FAIRCHEMWrapper(device=device, task_name=task, **named)
    elif model_type == "matgl":
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(device=device, **named)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported: mace, fairchem, matgl"
        )

    wrapper.load()
    logger.info(f"Loaded {model_type} model: {wrapper.model_name}")
    return wrapper
