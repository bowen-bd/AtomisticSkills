"""Most ORCA tests drive ORCA through SCINE (scine_utilities, scine_readuct),
which publishes wheels for x86_64 only; there they are skipped rather than
failed. test_orca_advanced.py drives ORCA directly and always runs."""

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
NEEDS_SCINE = {
    "test_orca_utils.py",
    "test_orca_singlepoint.py",
    "test_orca_optimization.py",
}


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("scine_utilities") is not None:
        return
    skip = pytest.mark.skip(reason="SCINE publishes x86_64 wheels only")
    for item in items:
        path = Path(str(item.fspath)).resolve()
        if path.parent == HERE and path.name in NEEDS_SCINE:
            item.add_marker(skip)
