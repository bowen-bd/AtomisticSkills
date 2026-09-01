#!/usr/bin/env python3
"""Parse a Wannier90 ``.wout`` Final State block into JSON.

Extracts the maximally-localized Wannier function (MLWF) spread decomposition
and centres from the last ``Final State`` block. This is the primary quality
gate for the Wannier stage (step 7): a total spread that is large relative to
the gauge-invariant lower bound Omega_I signals image folding in a long-vacuum
2D cell.

Usage:
    python parse_wout.py <prefix.wout>

Requirements:
    - Conda environment: base-agent
    - Required packages: none (Python standard library only)
"""

from __future__ import annotations

import argparse
import json
import re


def parse_wout(text: str) -> dict:
    """Parse the final ``Final State`` block of a Wannier90 ``.wout`` file.

    Args:
        text: Full contents of the ``.wout`` file.

    Returns:
        Dictionary with the spread decomposition (omega_I_A2, omega_D_A2,
        omega_OD_A2, omega_total_A2), the per-WF centres (wf_centres_xyz_A) and
        spreads (wf_spreads_A2), and the z-range of the centres
        (wf_centres_z_min, wf_centres_z_max) used as an image-folding check.
    """
    block_starts = [
        m.start() for m in re.finditer(r"^\s*Final State\s*$", text, re.MULTILINE)
    ]
    tail = text[block_starts[-1] :] if block_starts else text

    centres: list[list[float]] = []
    spreads: list[float] = []
    for m in re.finditer(
        r"WF centre and spread\s+\d+\s*\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)\s+([-\d.]+)",
        tail,
    ):
        centres.append([float(m.group(i)) for i in (1, 2, 3)])
        spreads.append(float(m.group(4)))

    def grab(label: str) -> float | None:
        m = re.search(rf"Omega\s+{label}\s*=\s*([-\d.Ee+]+)", tail)
        return float(m.group(1)) if m else None

    result: dict = {
        "omega_I_A2": grab("I"),
        "omega_D_A2": grab("D"),
        "omega_OD_A2": grab("OD"),
        "omega_total_A2": grab("Total"),
        "wf_centres_xyz_A": centres,
        "wf_spreads_A2": spreads,
    }
    if centres:
        zs = [c[2] for c in centres]
        result["wf_centres_z_min"] = min(zs)
        result["wf_centres_z_max"] = max(zs)
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Parse a Wannier90 .wout Final State block into JSON."
    )
    parser.add_argument("wout", help="Path to the Wannier90 .wout file")
    args = parser.parse_args()

    with open(args.wout) as fh:
        text = fh.read()
    print(json.dumps(parse_wout(text), indent=2))


if __name__ == "__main__":
    main()
