#!/usr/bin/env python3
"""Write matching full uniform QE and Wannier90 k-point blocks.

Usage: venv/run cpu python generate_kmesh.py --grid 4 4 4 --output-dir mesh
Requirements: cpu environment, NumPy and PyYAML.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from wannier_tb.io import save_config


def uniform_mesh(
    grid: tuple[int, int, int], shift: tuple[float, float, float] = (0, 0, 0)
) -> np.ndarray:
    """Return full fractional points; shift is measured in mesh spacings."""
    if (
        len(grid) != 3
        or min(grid) < 1
        or not np.isfinite(shift).all()
        or not all(0 <= v < 1 for v in shift)
    ):
        raise ValueError("Use three positive divisions and three shifts in [0,1)")
    return (
        np.asarray(list(product(*(range(n) for n in grid)))) + np.asarray(shift)
    ) / grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, nargs=3, required=True)
    parser.add_argument(
        "--shift",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Shifts in mesh spacings; 0 0 0 includes Gamma",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("mesh"))
    args = parser.parse_args()
    points = uniform_mesh(tuple(args.grid), tuple(args.shift))
    out = save_config(args.output_dir, "generate_kmesh", vars(args))
    coordinates = [" ".join(f"{x:.12f}" for x in point) for point in points]
    (out / "kmesh_qe.in").write_text(
        "K_POINTS crystal\n"
        + str(len(points))
        + "\n"
        + "\n".join(f"{row} {1 / len(points):.12f}" for row in coordinates)
        + "\n"
    )
    (out / "kmesh_w90.win").write_text(
        "mp_grid = "
        + " ".join(map(str, args.grid))
        + "\nbegin kpoints\n"
        + "\n".join(coordinates)
        + "\nend kpoints\n"
    )
    print(f"Wrote {len(points)} matching k-points to {out}")


if __name__ == "__main__":
    main()
