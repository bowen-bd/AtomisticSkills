#!/usr/bin/env python3
"""Parse ph.x ``electron_phonon='prt'`` output into JSON.

Reads the reference electron-phonon matrix elements |g(k, q, nu)| printed by a
single-q DFPT run (step 5). For the diagonal (ibnd == jbnd) band whose energy
is closest to the Fermi level (treated as the conduction-band minimum), it
reports the rank-1 |g| and the gauge-invariant sum over the four highest-omega
(LO/TO) modes. The gauge-invariant sum is the quantity to benchmark against,
not the raw per-mode value.

Usage:
    python parse_prt.py <ph.out> [--fermi <eV>]

Requirements:
    - Conda environment: base-agent
    - Required packages: none (Python standard library only)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*$",
    re.MULTILINE,
)


def parse_rows(text: str) -> list[dict]:
    """Extract the |g| table rows from a ph.x prt output.

    Args:
        text: Full contents of the ph.x output file.

    Returns:
        List of row dictionaries with keys ibnd, jbnd, imode, enk, enkq,
        omega_meV, g_meV.
    """
    rows: list[dict] = []
    for m in ROW_RE.finditer(text):
        rows.append(
            {
                "ibnd": int(m.group(1)),
                "jbnd": int(m.group(2)),
                "imode": int(m.group(3)),
                "enk": float(m.group(4)),
                "enkq": float(m.group(5)),
                "omega_meV": float(m.group(6)),
                "g_meV": float(m.group(7)),
            }
        )
    return rows


def summarize(text: str, fermi: float) -> dict:
    """Summarize the CBM electron-phonon coupling from a ph.x prt output.

    Args:
        text: Full contents of the ph.x output file.
        fermi: Fermi energy in eV used to select the CBM band.

    Returns:
        Dictionary with the q/k coordinates, the rank-1 |g| and mode, and the
        gauge-invariant sum of |g|^2 over the four highest-omega modes.
    """
    q = re.search(r"q coord\.:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
    k = re.search(r"k coord\.:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
    rows = parse_rows(text)

    result: dict = {
        "q_cartesian": [float(q.group(i)) for i in (1, 2, 3)] if q else None,
        "k_cartesian": [float(k.group(i)) for i in (1, 2, 3)] if k else None,
        "fermi_eV_used": fermi,
        "n_rows": len(rows),
    }

    diag: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["ibnd"] == r["jbnd"]:
            diag[r["ibnd"]].append(r)
    if diag:
        cbm = min(diag, key=lambda b: abs(diag[b][0]["enk"] - fermi))
        block = sorted(diag[cbm], key=lambda r: r["imode"])
        gs = [r["g_meV"] for r in block]
        om = [r["omega_meV"] for r in block]
        imax = max(range(len(gs)), key=lambda i: gs[i])
        top4 = sorted(range(len(om)), key=lambda i: om[i], reverse=True)[:4]
        result["cbm_ibnd"] = cbm
        result["cbm_enk_eV"] = block[0]["enk"]
        result["rank1_mode_index"] = block[imax]["imode"]
        result["rank1_omega_meV"] = om[imax]
        result["rank1_g_meV"] = gs[imax]
        result["sum_g2_LO_TO_quartet_meV2"] = sum(gs[i] ** 2 for i in top4)
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Parse ph.x electron_phonon='prt' output into JSON."
    )
    parser.add_argument("ph_out", help="Path to the ph.x prt output file")
    parser.add_argument(
        "--fermi",
        type=float,
        default=-5.655843,
        help="Fermi energy in eV used to select the CBM band (default: ZrS2 value)",
    )
    args = parser.parse_args()

    with open(args.ph_out) as fh:
        text = fh.read()
    print(json.dumps(summarize(text, args.fermi), indent=2))


if __name__ == "__main__":
    main()
