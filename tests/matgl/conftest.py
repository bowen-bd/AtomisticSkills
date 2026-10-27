"""Environment-specific conftest for MatGL tests."""

import pytest


def _resolve(name: str) -> str:
    """
    Apply the wrapper's alias table, the way ``MatGLWrapper.load`` does.

    The import stays inside the function because ``matgl_wrapper`` pulls in
    torch and matplotlib at module scope; a module-level import here would
    break collection of any suite that reaches ``tests/matgl`` without the
    mlip environment.
    """
    from src.utils.mlips.matgl.matgl_wrapper import resolve_model_name

    return resolve_model_name(name)


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

    An empty answer is a different case and is *not* reported as "everything
    was withdrawn": ``get_available_pretrained_models`` swallows Hub errors and
    returns ``[]``, so an empty catalogue means this runner cannot reach the
    Hub. Failing there is the point — skipping would name the wrong cause, and
    the property tests would quietly stop running instead of the suite saying
    it has no network.
    """
    from matgl.utils.io import get_available_pretrained_models

    models = set(get_available_pretrained_models())
    if not models:
        pytest.fail(
            "matgl reported an empty pretrained catalogue: "
            "get_available_pretrained_models() returns [] when the Hugging Face "
            "Hub is unreachable, so this runner most likely has no network. "
            "That is not the same as every artefact being withdrawn upstream, "
            "and it must not silently skip the model tests."
        )
    return models


@pytest.fixture(scope="session")
def require_matgl_model(matgl_pretrained_models):
    """
    Return a callable that skips the calling test if a model is unavailable.

    Usage:
        def test_formation_energy(require_matgl_model):
            require_matgl_model("M3GNet-Eform-MP-2019.4.1")
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
