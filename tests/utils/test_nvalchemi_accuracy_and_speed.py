"""Accuracy and speed comparison: NValchemi batch inference vs sequential.

Calls the *actual* MCP-exposed wrapper methods (``static_calculation``,
``relax_structure``, ``run_md``) — the same surface used by the MCP tools.

For each MLIP:
  1. Runs ``wrapper.static_calculation(list_of_structures)`` with NValchemi
     enabled explicitly with use_nvalchemi=True (GPU-parallel batch path).
  2. Runs ``wrapper.static_calculation(list_of_structures)`` with NValchemi
     disabled by default (sequential native-calculator path).
  3. Asserts energy and forces match within tolerance; reports speedup.

Note on stress
--------------
The sequential path returns Voigt-6 stress (from ``atoms.get_stress()``) while
the NValchemi path returns a full 3×3 Cauchy tensor.  We convert both to a
flat 6-component Voigt vector ``[xx, yy, zz, yz, xz, xy]`` before comparing.

Run commands
------------
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_accuracy_and_speed.py -k mace -v -s
    venv/run mlip python -m pytest tests/utils/test_nvalchemi_accuracy_and_speed.py -k matgl -v -s
    venv/run fairchem python -m pytest tests/utils/test_nvalchemi_accuracy_and_speed.py -k fairchem -v -s
    # standalone timing table:
    venv/run mlip python tests/utils/test_nvalchemi_accuracy_and_speed.py
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

ENERGY_TOL = 1e-3  # eV
FORCE_TOL = 1e-3  # eV/Å
STRESS_TOL = 1e-3  # eV/Å³ (Voigt 6-vector comparison)
N_REPEAT = 3  # timing repetitions (best-of-N)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def _cu_structures(n: int = 5):
    """N strained Cu FCC unit cells — same element, only volume changes."""
    from ase.build import bulk

    scales = np.linspace(0.96, 1.04, n)
    return [bulk("Cu", "fcc", a=3.6 * s) for s in scales]


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------


def _extract_static(
    batch_result: dict, idx: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (energy, forces [N,3], stress_voigt [6]) for structure *idx*.

    Handles both NValchemi (3×3 stress) and sequential (Voigt-6 stress).
    """
    r = batch_result["results"][idx]
    energy = float(r["energy"])
    forces = np.array(r["forces"], dtype=float)  # [[fx, fy, fz], ...]

    stress_raw = r.get("stress")
    if stress_raw is None:
        stress_voigt = np.zeros(6)
    else:
        s = np.array(stress_raw, dtype=float).squeeze()
        if s.ndim == 2 and s.shape == (3, 3):  # 3×3 Cauchy tensor (nested list)
            stress_voigt = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
            )
        elif s.ndim == 1 and s.size == 9:  # flat 3×3 (some backends flatten)
            s = s.reshape(3, 3)
            stress_voigt = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
            )
        else:  # 6-component Voigt from ASE get_stress()
            stress_voigt = s.flatten()[:6]
    return energy, forces, stress_voigt


def _time_call(fn, n_repeat: int = N_REPEAT):
    """Return (result, best_wall_time_s) over *n_repeat* calls."""
    result, best = None, float("inf")
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def _compare_static(label: str, wrapper: Any, structures: list) -> dict:
    """Run batch static_calculation with and without NValchemi; assert & report."""
    # NValchemi batch
    result_nv, t_nv = _time_call(
        lambda: wrapper.static_calculation(structures, use_nvalchemi=True)
    )
    assert "error" not in result_nv, f"NValchemi batch failed: {result_nv.get('error')}"
    assert result_nv.get("backend") == "nvalchemi", (
        f"Expected NValchemi backend, got: {result_nv.get('backend', 'sequential')}. "
        "Is nvalchemi installed and _get_nvalchemi_model() returning non-None?"
    )

    # Sequential fallback (same code path, NValchemi disabled)
    result_seq, t_seq = _time_call(lambda: wrapper.static_calculation(structures))
    assert "error" not in result_seq, (
        f"Sequential batch failed: {result_seq.get('error')}"
    )

    # Per-structure comparison
    n = len(structures)
    assert result_nv["total_structures"] == n
    assert result_seq["total_structures"] == n

    e_diffs, f_diffs, s_diffs = [], [], []
    for i in range(n):
        e_nv, f_nv, s_nv = _extract_static(result_nv, i)
        e_seq, f_seq, s_seq = _extract_static(result_seq, i)
        e_diffs.append(abs(e_nv - e_seq))
        f_diffs.append(np.abs(f_nv - f_seq).max())
        s_diffs.append(np.abs(s_nv - s_seq).max())

    speedup = t_seq / t_nv if t_nv > 0 else float("inf")

    print(
        f"\n{'=' * 62}\n{label}\n{'=' * 62}\n"
        f"  Structures  : {n}\n"
        f"  Sequential  : {t_seq * 1000:.1f} ms   (NValchemi disabled)\n"
        f"  NValchemi   : {t_nv * 1000:.1f} ms   (batch, single forward)\n"
        f"  Speedup     : {speedup:.2f}x\n"
        f"  ΔE max      : {max(e_diffs):.2e} eV  (tol {ENERGY_TOL:.0e})\n"
        f"  ΔF max      : {max(f_diffs):.2e} eV/Å  (tol {FORCE_TOL:.0e})\n"
        f"  ΔS max      : {max(s_diffs):.2e} eV/Å³  (tol {STRESS_TOL:.0e})\n"
    )

    assert max(e_diffs) < ENERGY_TOL, (
        f"[{label}] max ΔE={max(e_diffs):.3e} eV > {ENERGY_TOL}"
    )
    assert max(f_diffs) < FORCE_TOL, (
        f"[{label}] max ΔF={max(f_diffs):.3e} eV/Å > {FORCE_TOL}"
    )
    assert max(s_diffs) < STRESS_TOL, (
        f"[{label}] max ΔS={max(s_diffs):.3e} eV/Å³ > {STRESS_TOL}"
    )

    return {
        "label": label,
        "n": n,
        "t_seq_ms": t_seq * 1000,
        "t_nv_ms": t_nv * 1000,
        "speedup": speedup,
        "de_max": max(e_diffs),
        "df_max": max(f_diffs),
        "ds_max": max(s_diffs),
    }


# ---------------------------------------------------------------------------
# Fixtures: load once per class
# ---------------------------------------------------------------------------


def _load_wrapper(cls_path: str, model_name: str):
    """Import and load a wrapper; return wrapper or pytest.skip."""
    import importlib
    import importlib.util

    mod, cls = cls_path.rsplit(".", 1)

    if "fairchem" in mod and importlib.util.find_spec("fairchem") is None:
        pytest.skip("fairchem-core not installed")
    if "mace" in mod and importlib.util.find_spec("mace") is None:
        pytest.skip("mace-torch not installed")
    if "matgl" in mod and importlib.util.find_spec("matgl") is None:
        pytest.skip("matgl not installed")

    WrapperCls = getattr(importlib.import_module(mod), cls)
    w = WrapperCls(model_name=model_name, device="cpu")
    w.load()
    nv = w._get_nvalchemi_model()
    if nv is None:
        pytest.skip(f"_get_nvalchemi_model() returned None for {model_name}")
    return w


def _require_nvalchemi():
    from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

    if not NVALCHEMI_AVAILABLE:
        pytest.skip("nvalchemi not installed")


# ---------------------------------------------------------------------------
# MACE
# ---------------------------------------------------------------------------


@pytest.mark.mace
class TestStaticMACE:
    @pytest.fixture(scope="class")
    def wrapper(self):
        _require_nvalchemi()
        return _load_wrapper(
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            "MACE-OMAT-0-small",
        )

    def test_energy_forces_stress_5_structures(self, wrapper):
        _compare_static("MACE-OMAT-0-small | 5×Cu FCC", wrapper, _cu_structures(5))

    def test_single_vs_batch_consistent(self, wrapper):
        """Single-structure static_calculation must equal first element of batch."""
        structs = _cu_structures(2)
        single = wrapper.static_calculation(structs[0])
        batch = wrapper.static_calculation(structs)

        e_single = single["energy"]
        e_batch = batch["results"][0]["energy"]
        assert abs(e_single - e_batch) < ENERGY_TOL, (
            f"Single result {e_single:.4f} != batch[0] result {e_batch:.4f}"
        )


# ---------------------------------------------------------------------------
# MatGL TensorNet
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestStaticTensorNet:
    @pytest.fixture(scope="class")
    def wrapper(self):
        _require_nvalchemi()
        return _load_wrapper(
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            "TensorNet-PES-MatPES-PBE-2025.2",
        )

    def test_energy_forces_stress_5_structures(self, wrapper):
        _compare_static("TensorNet-MatPES-PBE | 5×Cu FCC", wrapper, _cu_structures(5))


# ---------------------------------------------------------------------------
# MatGL M3GNet
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestStaticM3GNet:
    @pytest.fixture(scope="class")
    def wrapper(self):
        _require_nvalchemi()
        return _load_wrapper(
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            "M3GNet-PES-MatPES-PBE-2025.2",
        )

    def test_energy_forces_stress_5_structures(self, wrapper):
        _compare_static(
            "M3GNet-PES-MatPES-PBE-2025.2 | 5×Cu FCC", wrapper, _cu_structures(5)
        )


# ---------------------------------------------------------------------------
# MatGL CHGNet
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestStaticCHGNet:
    @pytest.fixture(scope="class")
    def wrapper(self):
        _require_nvalchemi()
        return _load_wrapper(
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            "CHGNet-PES-MatPES-PBE-2025.2.10",
        )

    def test_energy_forces_stress_5_structures(self, wrapper):
        _compare_static(
            "CHGNet-PES-MatPES-PBE-2025.2.10 | 5×Cu FCC", wrapper, _cu_structures(5)
        )


# ---------------------------------------------------------------------------
# FairChem
# ---------------------------------------------------------------------------


@pytest.mark.fairchem
class TestStaticFairChem:
    @pytest.fixture(scope="class")
    def wrapper(self):
        _require_nvalchemi()
        return _load_wrapper(
            "src.utils.mlips.fairchem.fairchem_wrapper.FAIRCHEMWrapper",
            "uma-s-1p2",
        )

    def test_energy_forces_stress_5_structures(self, wrapper):
        _compare_static("FairChem uma-s-1p2 | 5×Cu FCC", wrapper, _cu_structures(5))

    def test_neighbor_config_is_none(self, wrapper):
        nv = wrapper._get_nvalchemi_model()
        assert nv.model_config.neighbor_config is None


# ---------------------------------------------------------------------------
# Standalone entry point — prints a summary table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

    if not NVALCHEMI_AVAILABLE:
        print("nvalchemi not installed — cannot run benchmarks.")
        raise SystemExit(1)

    MODELS = [
        ("MACE-OMAT-0-small", "src.utils.mlips.mace.mace_wrapper.MACEWrapper"),
        (
            "TensorNet-PES-MatPES-PBE-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
        ),
        (
            "M3GNet-PES-MatPES-PBE-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
        ),
        (
            "CHGNet-PES-MatPES-PBE-2025.2.10",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
        ),
        ("uma-s-1p2", "src.utils.mlips.fairchem.fairchem_wrapper.FAIRCHEMWrapper"),
    ]

    structs = _cu_structures(5)
    rows = []
    for model_name, cls_path in MODELS:
        import importlib

        mod, cls = cls_path.rsplit(".", 1)
        WrapperCls = getattr(importlib.import_module(mod), cls)
        w = WrapperCls(model_name=model_name, device="cpu")
        print(f"\nLoading {model_name} ...", end=" ", flush=True)
        try:
            w.load()
        except Exception as exc:
            print(f"SKIP ({exc})")
            continue
        nv = w._get_nvalchemi_model()
        if nv is None:
            print("SKIP (_get_nvalchemi_model returned None)")
            continue
        print("OK")
        try:
            r = _compare_static(model_name, w, structs)
            rows.append(r)
        except AssertionError as ae:
            print(f"  FAIL: {ae}")

    if rows:
        W = 44
        print(f"\n{'=' * (W + 46)}")
        print(
            f"{'Model':<{W}} {'Speedup':>8}  {'ΔE max':>10}  {'ΔF max':>10}  {'ΔS max':>10}"
        )
        print(f"{'-' * (W + 46)}")
        for r in rows:
            print(
                f"{r['label']:<{W}} {r['speedup']:>7.2f}x"
                f"  {r['de_max']:>9.2e}  {r['df_max']:>9.2e}  {r['ds_max']:>9.2e}"
            )
        print(f"{'=' * (W + 46)}")
