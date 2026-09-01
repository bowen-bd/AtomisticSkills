"""
Benchmark NValchemi batch inference speedup and accuracy for MLIP wrappers.

Runs static_calculation on N strained Cu FCC unit cells (N=2,5,10,20) for
each model in the specified environment, comparing NValchemi batch vs
sequential (NValchemi disabled) throughput and max energy/force/stress errors.

Usage:
    python skills/ml-mlip-nvalchemi/scripts/run_nvalchemi_benchmark.py \
        --env mace --device cuda --n-repeat 3 --output results.json

Requirements:
    - Conda environment: mace-agent, matgl-agent, or fairchem-agent
    - nvalchemi-toolkit installed in the environment
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Model registry per environment
# Each entry: (model_name, wrapper_cls_path, task_name_or_None, label)
# ---------------------------------------------------------------------------

ENV_MODELS: dict[str, list[tuple[str, str, str | None, str]]] = {
    "mace": [
        # OMAT models
        (
            "MACE-OMAT-0-small",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            None,
            "MACE-OMAT-0-small",
        ),
        (
            "MACE-OMAT-0-medium",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            None,
            "MACE-OMAT-0-medium",
        ),
        # MACE-MH multi-head: solid-state heads
        (
            "MACE-MH-1",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            "omat_pbe",
            "MACE-MH-1/omat_pbe",
        ),
        (
            "MACE-MH-1",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            "matpes_r2scan",
            "MACE-MH-1/matpes_r2scan",
        ),
        # MACE-MP second generation
        (
            "MACE-MP-medium-0b3",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            None,
            "MACE-MP-medium-0b3",
        ),
        # MACE-MatPES dedicated models
        (
            "MACE-MATPES-PBE-0",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            None,
            "MACE-MATPES-PBE-0",
        ),
        (
            "MACE-MATPES-R2SCAN-0",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            None,
            "MACE-MATPES-R2SCAN-0",
        ),
    ],
    "matgl": [
        # TensorNet PES (NValchemi supported)
        (
            "TensorNet-PES-MatPES-PBE-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "TensorNet-PES-PBE-2025.2",
        ),
        (
            "TensorNet-PES-MatPES-r2SCAN-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "TensorNet-PES-r2SCAN-2025.2",
        ),
        (
            "TensorNet-PES-ANI-1x-Subset",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "TensorNet-PES-ANI-1x",
        ),
        # M3GNet PES (NValchemi supported)
        (
            "M3GNet-PES-MatPES-PBE-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "M3GNet-PES-PBE-2025.2",
        ),
        (
            "M3GNet-PES-MatPES-r2SCAN-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "M3GNet-PES-r2SCAN-2025.2",
        ),
        (
            "M3GNet-PES-ANI-1x-Subset",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "M3GNet-PES-ANI-1x",
        ),
        # CHGNet PES (NValchemi supported)
        (
            "CHGNet-PES-MatPES-PBE-2025.2.10",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "CHGNet-PES-PBE-2025.2.10",
        ),
        (
            "CHGNet-PES-MatPES-r2SCAN-2025.2.10",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "CHGNet-PES-r2SCAN-2025.2.10",
        ),
        # QET PES (NValchemi NOT supported — will skip gracefully)
        (
            "QET-PES-MatPES-PBE-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "QET-PES-PBE-2025.2",
        ),
        (
            "QET-PES-MatPES-r2SCAN-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "QET-PES-r2SCAN-2025.2",
        ),
        # SO3Net PES (NValchemi NOT supported — will skip gracefully)
        (
            "SO3Net-PES-ANI-1x-Subset",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "SO3Net-PES-ANI-1x",
        ),
    ],
    "fairchem": [
        # UMA models — omat head (solid-state, matches Cu FCC test)
        (
            "uma-s-1p2",
            "src.utils.mlips.fairchem.fairchem_wrapper.FAIRCHEMWrapper",
            "omat",
            "uma-s-1p2/omat",
        ),
        (
            "uma-m-1p1",
            "src.utils.mlips.fairchem.fairchem_wrapper.FAIRCHEMWrapper",
            "omat",
            "uma-m-1p1/omat",
        ),
        (
            "uma-s-1p1",
            "src.utils.mlips.fairchem.fairchem_wrapper.FAIRCHEMWrapper",
            "omat",
            "uma-s-1p1/omat",
        ),
    ],
}

BATCH_SIZES = [2, 5, 10, 20]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cu_structures(n: int):
    from ase.build import bulk

    scales = np.linspace(0.96, 1.04, n)
    return [bulk("Cu", "fcc", a=3.6 * s) for s in scales]


@contextmanager
def _nvalchemi_disabled():
    import src.utils.mlips.nvalchemi.nvalchemi_utils as _m

    orig = _m.check_nvalchemi_available
    _m.check_nvalchemi_available = lambda: False
    try:
        yield
    finally:
        _m.check_nvalchemi_available = orig


def _extract_static(r: dict) -> tuple[float, np.ndarray, np.ndarray]:
    energy = float(r["energy"])
    forces = np.array(r["forces"], dtype=float)
    stress_raw = r.get("stress")
    if stress_raw is None:
        stress_v = np.zeros(6)
    else:
        s = np.array(stress_raw, dtype=float).squeeze()
        if s.ndim == 2 and s.shape == (3, 3):
            stress_v = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        elif s.ndim == 1 and s.size == 9:
            s = s.reshape(3, 3)
            stress_v = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        else:
            stress_v = s.flatten()[:6]
    return energy, forces, stress_v


def _time_call(fn, n_repeat: int):
    result, best = None, float("inf")
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def _benchmark_model(wrapper: Any, label: str, n_repeat: int) -> dict:
    print(f"\n  Benchmarking {label} ...", flush=True)

    from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

    if not NVALCHEMI_AVAILABLE:
        print("    SKIP: nvalchemi not available", flush=True)
        return {"label": label, "skipped": True, "reason": "nvalchemi unavailable"}

    nv_model = wrapper._get_nvalchemi_model()
    if nv_model is None:
        print(
            "    SKIP: _get_nvalchemi_model() returned None (model type not supported)",
            flush=True,
        )
        return {
            "label": label,
            "skipped": True,
            "reason": "model type not supported by NValchemi",
        }

    rows = []
    for n in BATCH_SIZES:
        structures = _make_cu_structures(n)

        result_nv, t_nv = _time_call(
            lambda: wrapper.static_calculation(structures), n_repeat
        )
        if "error" in result_nv:
            print(f"    N={n}: NValchemi ERROR: {result_nv['error']}", flush=True)
            rows.append({"n": n, "error": str(result_nv["error"])})
            continue

        with _nvalchemi_disabled():
            result_seq, t_seq = _time_call(
                lambda: wrapper.static_calculation(structures), n_repeat
            )
        if "error" in result_seq:
            print(f"    N={n}: Sequential ERROR: {result_seq['error']}", flush=True)
            rows.append({"n": n, "error": str(result_seq["error"])})
            continue

        e_diffs, f_diffs, s_diffs = [], [], []
        for i in range(n):
            e_nv, f_nv, s_nv = _extract_static(result_nv["results"][i])
            e_seq, f_seq, s_seq = _extract_static(result_seq["results"][i])
            e_diffs.append(abs(e_nv - e_seq))
            f_diffs.append(float(np.abs(f_nv - f_seq).max()))
            s_diffs.append(float(np.abs(s_nv - s_seq).max()))

        speedup = t_seq / t_nv if t_nv > 0 else float("inf")
        row = {
            "n": n,
            "t_nv_ms": round(t_nv * 1000, 1),
            "t_seq_ms": round(t_seq * 1000, 1),
            "speedup": round(speedup, 3),
            "de_max": float(max(e_diffs)),
            "df_max": float(max(f_diffs)),
            "ds_max": float(max(s_diffs)),
        }
        print(
            f"    N={n:2d}: NV={t_nv*1000:.0f}ms  Seq={t_seq*1000:.0f}ms"
            f"  Speedup={speedup:.2f}x  ΔE={max(e_diffs):.1e}  ΔF={max(f_diffs):.1e}",
            flush=True,
        )
        rows.append(row)

    return {"label": label, "skipped": False, "rows": rows}


def _load_wrapper(
    cls_path: str, model_name: str, task_name: str | None, device: str
) -> Any | None:
    import importlib

    mod, cls = cls_path.rsplit(".", 1)
    WrapperCls = getattr(importlib.import_module(mod), cls)
    kwargs: dict = {"model_name": model_name, "device": device}
    if task_name is not None:
        # MACE uses `head`; FairChem/others use `task_name`
        if "mace" in cls_path.lower():
            kwargs["head"] = task_name
        else:
            kwargs["task_name"] = task_name
    w = WrapperCls(**kwargs)
    try:
        w.load()
        return w
    except Exception as exc:
        print(f"    LOAD FAILED: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark NValchemi batch inference")
    parser.add_argument("--env", choices=list(ENV_MODELS), required=True)
    parser.add_argument(
        "--device", default="cuda", help="Device: cuda (default), cpu, auto"
    )
    parser.add_argument("--n-repeat", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Filter by label substring (e.g. --models TensorNet MACE-OMAT)",
    )
    args = parser.parse_args()

    entries = ENV_MODELS[args.env]
    if args.models:
        entries = [e for e in entries if any(m in e[3] for m in args.models)]

    print(f"\n{'='*62}")
    print(f"NValchemi Benchmark: {args.env.upper()} | device={args.device}")
    print(f"Models: {[e[3] for e in entries]}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"{'='*62}")

    results = []
    loaded_cache: dict[
        tuple[str, str | None], Any
    ] = {}  # (model_name, task_name) → wrapper

    for model_name, cls_path, task_name, label in entries:
        cache_key = (model_name, task_name)
        if cache_key not in loaded_cache:
            print(f"\nLoading {label} ...", flush=True)
            w = _load_wrapper(cls_path, model_name, task_name, args.device)
            loaded_cache[cache_key] = w
        wrapper = loaded_cache[cache_key]

        if wrapper is None:
            results.append({"label": label, "skipped": True, "reason": "load failed"})
            continue
        r = _benchmark_model(wrapper, label, args.n_repeat)
        results.append(r)

    # Summary table
    print(f"\n{'='*95}")
    print(
        f"{'Label':<45} {'N':>4} {'NV ms':>8} {'Seq ms':>8} {'Speedup':>9}  {'ΔE max':>10}  {'ΔF max':>10}"
    )
    print(f"{'-'*95}")
    for r in results:
        if r.get("skipped"):
            print(f"  {r['label']:<43}  SKIPPED: {r.get('reason','')}")
            continue
        for row in r.get("rows", []):
            if "error" in row:
                print(f"  {r['label']:<43}  N={row['n']:2d}  ERROR: {row['error']}")
                continue
            print(
                f"  {r['label']:<43}  {row['n']:>4}"
                f"  {row['t_nv_ms']:>7.0f}  {row['t_seq_ms']:>7.0f}"
                f"  {row['speedup']:>8.2f}x  {row['de_max']:>10.2e}  {row['df_max']:>10.2e}"
            )
    print(f"{'='*95}")

    if args.output:
        out = {"env": args.env, "device": args.device, "models": results}
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
