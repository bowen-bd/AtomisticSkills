"""
Benchmark NValchemi batched MD speedup for MLIP wrappers.

Runs `run_md` on 20 strained Cu FCC supercells (≥10 Å) for 100 steps at 300 K
under nvt_nose_hoover, comparing NValchemi batch vs sequential (NValchemi disabled)
wall time. Best-of-N timing is used.

Usage:
    # mace-agent
    python skills/ml-mlip-nvalchemi/scripts/run_md_benchmark.py --env mace

    # matgl-agent
    python skills/ml-mlip-nvalchemi/scripts/run_md_benchmark.py --env matgl

    # fairchem-agent
    python skills/ml-mlip-nvalchemi/scripts/run_md_benchmark.py --env fairchem

Requirements:
    - Correct conda environment per env flag (mace-agent / matgl-agent / fairchem-agent)
    - nvalchemi-toolkit installed in the environment
"""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Model entries: (model_name, wrapper_cls_path, task_name_or_None, label)
# ---------------------------------------------------------------------------

ENV_MODELS: dict[str, list[tuple[str, str, str | None, str]]] = {
    "mace": [
        (
            "MACE-OMAT-0-small",
            "src.utils.mlips.mace.mace_wrapper.MACEWrapper",
            None,
            "MACE-OMAT-0-small",
        ),
    ],
    "matgl": [
        (
            "TensorNet-PES-MatPES-PBE-2025.2",
            "src.utils.mlips.matgl.matgl_wrapper.MatGLWrapper",
            None,
            "TensorNet-PES-MatPES-PBE-2025.2",
        ),
    ],
    "fairchem": [
        (
            "uma-s-1p2",
            "src.utils.mlips.fairchem.fairchem_wrapper.FAIRCHEMWrapper",
            "omat",
            "FairChem uma-s-1p2",
        ),
    ],
}

# Benchmark parameters
N_STRUCTURES = 20
MD_STEPS = 100
MD_TEMPERATURE = 300.0
MD_TIMESTEP = 2.0
MD_ENSEMBLE = "nvt_nose_hoover"
SUPERCELL_MIN_LENGTH = 10.0  # Å — unit cells are expanded until all sides ≥ this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cu_supercells(n: int, min_length: float = SUPERCELL_MIN_LENGTH):
    """Make n strained Cu FCC supercells where each side ≥ min_length Å.

    Uses the conventional cubic FCC cell (4 atoms, side = a) with a fixed
    repeat count determined from the equilibrium a=3.6 Å.  All structures in
    the batch therefore share the same atom count, which is required for
    NValchemi's fixed-batch GPU kernels.
    """
    from ase.build import bulk

    a0 = 3.6  # equilibrium lattice parameter (Å)
    reps = max(1, int(np.ceil(min_length / a0)))  # e.g. ceil(10/3.6) = 3
    scales = np.linspace(0.96, 1.04, n)
    structures = []
    for s in scales:
        a = a0 * s
        unit_cell = bulk("Cu", "fcc", a=a, cubic=True)  # 4-atom cubic cell
        sc = unit_cell.repeat(reps)
        structures.append(sc)
    return structures


@contextmanager
def _nvalchemi_disabled():
    import src.utils.mlips.nvalchemi.nvalchemi_utils as _m

    orig = _m.check_nvalchemi_available
    _m.check_nvalchemi_available = lambda: False
    try:
        yield
    finally:
        _m.check_nvalchemi_available = orig


def _load_wrapper(
    cls_path: str, model_name: str, task_name: str | None, device: str
) -> Any | None:
    import importlib

    mod, cls = cls_path.rsplit(".", 1)
    WrapperCls = getattr(importlib.import_module(mod), cls)
    kwargs: dict = {"model_name": model_name, "device": device}
    if task_name is not None:
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


def _benchmark_md(
    wrapper: Any, label: str, structures: list, n_repeat: int, output_dir: str
) -> dict:
    print(f"\n  Benchmarking MD: {label} ...", flush=True)
    print(
        f"  Structures: {N_STRUCTURES} x Cu FCC supercell, ~{len(structures[0])} atoms each",
        flush=True,
    )
    print(
        f"  Steps: {MD_STEPS}, Ensemble: {MD_ENSEMBLE}, T: {MD_TEMPERATURE} K",
        flush=True,
    )

    from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

    if not NVALCHEMI_AVAILABLE:
        print("    SKIP: nvalchemi not available", flush=True)
        return {"label": label, "skipped": True, "reason": "nvalchemi unavailable"}

    nv_model = wrapper._get_nvalchemi_model()
    if nv_model is None:
        print("    SKIP: _get_nvalchemi_model() returned None", flush=True)
        return {
            "label": label,
            "skipped": True,
            "reason": "model not supported by NValchemi",
        }

    # --- Sequential timing: NValchemi disabled, run one-by-one ---
    print(
        "  Running sequential MD (NValchemi disabled, one structure at a time)...",
        flush=True,
    )
    best_seq = float("inf")
    for rep in range(n_repeat):
        t0 = time.perf_counter()
        with _nvalchemi_disabled():
            for struct in structures:
                wrapper.run_md(
                    structure_data=struct,
                    temperature=MD_TEMPERATURE,
                    steps=MD_STEPS,
                    timestep=MD_TIMESTEP,
                    ensemble=MD_ENSEMBLE,
                    log_interval=MD_STEPS + 1,  # suppress per-step I/O
                    output_dir=output_dir,
                )
        elapsed = time.perf_counter() - t0
        best_seq = min(best_seq, elapsed)
        print(f"    Sequential repeat {rep+1} wall time: {elapsed:.2f} s", flush=True)
    print(f"  Best sequential time: {best_seq:.2f} s", flush=True)

    # --- Batched timing: NValchemi active, all structures at once ---
    # Run each repeat independently and only count runs that actually used NValchemi
    # (not the sequential fallback). Some models (e.g. TensorNet) corrupt the CUDA
    # context after a successful batch run, so we catch exceptions from second+ repeats.
    print("  Running batched MD (NValchemi)...", flush=True)
    batch_times: list[float] = []
    for rep in range(n_repeat):
        t0 = time.perf_counter()
        try:
            r = wrapper.run_md(
                structure_data=structures,
                temperature=MD_TEMPERATURE,
                steps=MD_STEPS,
                timestep=MD_TIMESTEP,
                ensemble=MD_ENSEMBLE,
                log_interval=MD_STEPS + 1,
                output_dir=output_dir,
                extract_batch_results=False,
            )
        except Exception as exc:
            print(f"    Batched repeat {rep+1} EXCEPTION: {exc}", flush=True)
            break  # CUDA context likely corrupted; stop repeating
        elapsed = time.perf_counter() - t0
        backend = r.get("backend", "unknown")
        print(
            f"    Batched repeat {rep+1} wall time: {elapsed:.2f} s (backend={backend})",
            flush=True,
        )
        if backend == "nvalchemi":
            batch_times.append(elapsed)
        else:
            print(
                f"    WARNING: backend={backend}, not counting as NValchemi batch time",
                flush=True,
            )
            break  # NValchemi failed and fell back; stop

    if not batch_times:
        print("  No successful NValchemi batch runs — reporting N/A", flush=True)
        return {
            "label": label,
            "skipped": True,
            "reason": "NValchemi batch MD unavailable",
        }

    best_batch = min(batch_times)
    print(
        f"  Best batched time: {best_batch:.2f} s ({len(batch_times)} successful NValchemi run(s))",
        flush=True,
    )
    speedup = best_seq / best_batch if best_batch > 0 else float("inf")
    print(f"  Speedup: {speedup:.2f}x", flush=True)

    return {
        "label": label,
        "skipped": False,
        "n_structures": N_STRUCTURES,
        "atoms_per_structure": len(structures[0]),
        "steps": MD_STEPS,
        "t_seq_s": round(best_seq, 2),
        "t_batch_s": round(best_batch, 2),
        "speedup": round(speedup, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NValchemi batched MD speedup"
    )
    parser.add_argument("--env", choices=list(ENV_MODELS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n-repeat", type=int, default=2, help="Timing repetitions (best-of-N)"
    )
    args = parser.parse_args()

    structures = _make_cu_supercells(N_STRUCTURES)
    atoms_count = len(structures[0])

    print(f"\n{'='*65}")
    print(f"MD Benchmark: {args.env.upper()} | device={args.device}")
    print(
        f"N={N_STRUCTURES} Cu FCC supercells ({atoms_count} atoms each, ≥{SUPERCELL_MIN_LENGTH} Å sides)"
    )
    print(
        f"Steps={MD_STEPS}, Ensemble={MD_ENSEMBLE}, T={MD_TEMPERATURE} K, dt={MD_TIMESTEP} fs"
    )
    print(f"Best-of-{args.n_repeat} timing")
    print(f"{'='*65}")

    results = []
    for model_name, cls_path, task_name, label in ENV_MODELS[args.env]:
        print(f"\nLoading {label} ...", flush=True)
        wrapper = _load_wrapper(cls_path, model_name, task_name, args.device)
        if wrapper is None:
            results.append({"label": label, "skipped": True, "reason": "load failed"})
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            r = _benchmark_md(wrapper, label, structures, args.n_repeat, tmpdir)
        results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Model':<35} {'Seq (s)':>8} {'Batch (s)':>10} {'Speedup':>9}")
    print(f"{'-'*70}")
    for r in results:
        if r.get("skipped"):
            print(f"  {r['label']:<33}  SKIPPED: {r.get('reason','')}")
            continue
        print(
            f"  {r['label']:<33}  {r['t_seq_s']:>7.2f}  {r['t_batch_s']:>9.2f}  {r['speedup']:>8.2f}x"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
