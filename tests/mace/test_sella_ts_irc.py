"""Sella runs a saddle-point search and an IRC check, not just --help.

The chem-ts-optimization / chem-irc-verification example (acetonitrile to
methyl isocyanide) with MACE on the CPU. Run in:
    venv/run mlip python -m pytest tests/mace/test_sella_ts_irc.py
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.mace

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / "skills" / "chem-ts-optimization"
IRC = ROOT / "skills" / "chem-irc-verification"


def run(script, *args):
    result = subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_ts_then_irc_with_mace(tmp_path):
    run(
        TS / "scripts" / "optimize_ts_sella.py",
        "--ts_guess",
        str(TS / "examples" / "acetonitrile" / "ts_guess.xyz"),
        "--model_type",
        "mace",
        "--device",
        "cpu",
        "--fmax",
        "0.05",
        "--steps",
        "300",
        "--output_dir",
        str(tmp_path / "ts"),
    )
    ts = json.loads((tmp_path / "ts" / "ts_optimization_results.json").read_text())
    assert ts["sella_converged"] and ts["is_first_order_saddle"], ts

    run(
        IRC / "scripts" / "verify_irc_sella.py",
        "--reactant",
        str(IRC / "examples" / "acetonitrile" / "reactant_optimized.xyz"),
        "--product",
        str(IRC / "examples" / "acetonitrile" / "product_optimized.xyz"),
        "--ts",
        str(tmp_path / "ts" / "ts_optimized.xyz"),
        "--model_type",
        "mace",
        "--device",
        "cpu",
        "--steps",
        "300",
        "--output_dir",
        str(tmp_path / "irc"),
    )
    irc = json.loads((tmp_path / "irc" / "irc_verification_results.json").read_text())
    assert irc["verification_passed"], irc
