#!/usr/bin/env python3
"""Generate an explicit uniform k-point list for a QE NSCF ``K_POINTS crystal`` card.

EPW reads back the full uniform mesh, so the NSCF must use an explicit k-list
(``K_POINTS crystal``), never ``automatic`` (step 2). This prints the header
line and the ``n1 * n2 * n3`` fractional points with unit weights, ready to
paste into ``nscf.in``.

Usage:
    python gen_kpoints.py 12 12 1

Requirements:
    - Conda environment: base-agent
    - Required packages: none (Python standard library only)
"""

from __future__ import annotations

import argparse


def uniform_kpoints(n1: int, n2: int, n3: int) -> list[tuple[float, float, float]]:
    """Build the fractional coordinates of a uniform Gamma-centred mesh.

    Args:
        n1: Number of divisions along b1.
        n2: Number of divisions along b2.
        n3: Number of divisions along b3 (use 1 for a 2D slab).

    Returns:
        List of (k1, k2, k3) fractional coordinates.
    """
    return [
        (i / n1, j / n2, k / n3)
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]


def format_card(points: list[tuple[float, float, float]]) -> str:
    """Format a k-point list as a QE ``K_POINTS crystal`` card.

    Args:
        points: Fractional coordinates from :func:`uniform_kpoints`.

    Returns:
        The card text, including the ``K_POINTS crystal`` header and count line.
    """
    lines = ["K_POINTS crystal", str(len(points))]
    lines += [f"{a:.10f}  {b:.10f}  {c:.10f}  1.0" for a, b, c in points]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate an explicit uniform K_POINTS crystal card for QE NSCF."
    )
    parser.add_argument("n1", type=int, help="Divisions along b1")
    parser.add_argument("n2", type=int, help="Divisions along b2")
    parser.add_argument("n3", type=int, help="Divisions along b3 (1 for a 2D slab)")
    args = parser.parse_args()

    print(format_card(uniform_kpoints(args.n1, args.n2, args.n3)))


if __name__ == "__main__":
    main()
