#!/usr/bin/env python3
"""Print a reference-vs-measured comparison table for two JSON files.

Loads a reference JSON (expected values) and a measured JSON (parser output),
and prints a ``field | reference | measured | delta`` table. It makes no
assertions: the reviewer or agent judges whether each delta is acceptable, as
tolerances here are physical (gauge rotations, mesh convergence), not exact.

Usage:
    python compare_reference.py reference.json measured.json

Requirements:
    - Conda environment: base-agent
    - Required packages: none (Python standard library only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def delta(reference: object, measured: object) -> str:
    """Compute ``measured - reference`` for numeric fields, else ``-``.

    Args:
        reference: Reference value for a field.
        measured: Measured value for a field.

    Returns:
        Signed difference formatted to 6 significant figures, or ``-`` when
        either value is non-numeric.
    """
    try:
        return f"{float(measured) - float(reference):+.6g}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Print a reference-vs-measured comparison table for two JSON files."
    )
    parser.add_argument("reference", help="Path to reference.json (expected values)")
    parser.add_argument("measured", help="Path to measured.json (parser output)")
    args = parser.parse_args()

    ref = json.loads(Path(args.reference).read_text() or "{}")
    measured = json.loads(Path(args.measured).read_text() or "{}")

    print(f"{'field':<35} {'reference':>18} {'measured':>18} {'delta':>12}")
    for key in sorted(set(ref) | set(measured)):
        r, mv = ref.get(key, "-"), measured.get(key, "-")
        print(f"{key:<35} {str(r):>18} {str(mv):>18} {delta(r, mv):>12}")


if __name__ == "__main__":
    main()
