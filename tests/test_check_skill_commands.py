"""Unit tests for tools/check_skill_commands.py without running subprocesses."""

from __future__ import annotations

import pytest

from tools.check_skill_commands import classify, format_summary


@pytest.mark.parametrize("arch", ["aarch64", "arm64", "AARCH64", "ARM64"])
def test_classify_pymol_on_arm(arch: str) -> None:
    output = "ModuleNotFoundError: No module named 'pymol'"
    reason = classify(output, arch)
    assert reason is not None
    assert "pymol" in reason.lower()
    assert "x86_64" in reason.lower()


def test_classify_pymol_on_x86() -> None:
    output = "ModuleNotFoundError: No module named 'pymol'"
    assert classify(output, "x86_64") is None
    assert classify(output, "amd64") is None


@pytest.mark.parametrize("arch", ["aarch64", "arm64"])
def test_classify_scine_on_arm(arch: str) -> None:
    for mod in ["scine_utilities", "scine_readuct"]:
        output = f"ModuleNotFoundError: No module named '{mod}'"
        reason = classify(output, arch)
        assert reason is not None
        assert "scine" in reason.lower()


def test_classify_scine_on_x86() -> None:
    output = "ModuleNotFoundError: No module named 'scine_utilities'"
    assert classify(output, "x86_64") is None


@pytest.mark.parametrize("arch", ["aarch64", "arm64"])
def test_classify_vina_on_arm(arch: str) -> None:
    # Case 1: Missing docking extra
    output_no_vina = "ModuleNotFoundError: No module named 'vina'"
    reason1 = classify(output_no_vina, arch)
    assert reason1 is not None
    assert "vina" in reason1.lower()

    # Case 2: Missing Boost on system
    output_boost = "ImportError: libboost_python312.so.1.83.0: cannot open shared object file (vina)"
    reason2 = classify(output_boost, arch)
    assert reason2 is not None
    assert "vina" in reason2.lower()
    assert "boost" in reason2.lower()


def test_classify_vina_on_x86() -> None:
    output = "ModuleNotFoundError: No module named 'vina'"
    assert classify(output, "x86_64") is None


@pytest.mark.parametrize("arch", ["aarch64", "arm64", "x86_64"])
def test_classify_unrelated_errors(arch: str) -> None:
    assert classify("ModuleNotFoundError: No module named 'foo'", arch) is None
    assert classify("SyntaxError: invalid syntax", arch) is None
    assert classify("ValueError: unexpected parameter", arch) is None


def test_format_summary() -> None:
    assert format_summary(10, 0, 2) == "10 passed, 0 failed, 2 skipped"
    assert format_summary(ok=5, fail=1, skip=0) == "5 passed, 1 failed, 0 skipped"
    assert (
        format_summary(passed=8, failed=3, skipped=1) == "8 passed, 3 failed, 1 skipped"
    )
