"""Environment-specific conftest for MatGL tests."""

import pytest


def _resolve(name: str) -> str:
    """Apply the wrapper's alias table, the way MatGLWrapper.load does."""
    from src.utils.mlips.matgl.matgl_wrapper import AVAILABLE_MATGL_MODELS

    for key, value in AVAILABLE_MATGL_MODELS.items():
        if name.upper() == key.upper():
            return value
    return name


@pytest.fixture(scope="session")
def matgl_pretrained_models() -> set[str]:
    """
    The bare model names matgl can download right now.

    matgl 4.1 resolves a bare name only through the ``materialyze`` Hugging
    Face org and keeps no legacy mirror, so a model dropped from that org fails
    at load time with "Bad serialized model or bad model name". That is an
    upstream catalogue change rather than a defect here, and it should not turn
    the suite red: ask the Hub once per session and let the tests that need a
    missing model skip with the reason.

    Returns an empty set rather than raising when matgl has no Hub lookup
    (an older matgl elsewhere on the path), so a mis-targeted run skips
    instead of erroring.
    """
    try:
        from matgl.utils.io import get_available_pretrained_models
    except (ImportError, AttributeError):
        return set()

    # get_available_pretrained_models swallows Hub errors and returns [], which
    # degrades to "skip everything" when the Hub is unreachable.
    return set(get_available_pretrained_models())


@pytest.fixture(scope="session")
def require_matgl_model(matgl_pretrained_models):
    """
    Return a callable that skips the calling test if a model is unavailable.

    Usage:
        def test_bandgap(require_matgl_model):
            require_matgl_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            ...
    """

    def _require(name: str) -> None:
        resolved = _resolve(name)
        if resolved in matgl_pretrained_models:
            return
        pytest.skip(
            f"matgl cannot download {resolved} (requested as {name}): it is no "
            "longer published in the materialyze Hugging Face org, which is the "
            "only source matgl 4.1 resolves bare names against. Missing upstream "
            "artefact, not a regression in this repository."
        )

    return _require
