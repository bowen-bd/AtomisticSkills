#!/usr/bin/env python3
"""Inspect folded Wannier90 matrix elements and optional physical distances.

Usage: venv/run cpu python parse_hr.py seed_hr.dat --win seed.win --wout seed.wout --output-dir results
Requirements: cpu environment, NumPy and PyYAML.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from parse_wout import parse_wout
from wannier_tb.io import (
    failure_report,
    BOHR_ANGSTROM,
    read_cell,
    read_hr,
    resolve_wsvec,
    save_config,
    write_json,
)


def parse_hamiltonian(
    filepath: str | Path,
    threshold: float = 1e-4,
    cell_matrix: np.ndarray | None = None,
    centres: np.ndarray | None = None,
    wsvec: str | Path | None = None,
    legacy: bool = False,
) -> tuple[dict, list[dict]]:
    """Summarize folded H(R), keeping degeneracies separate from raw hoppings.

    With cell and centres, report the shortest orbital-pair distance among the
    sidecar shifts. With cell alone, report lattice-translation distances only.
    No cell means distances and the decay profile are not computed.
    """
    nw, nr, deg, r, h = read_hr(filepath)
    shifts, convention = resolve_wsvec(filepath, r, nw, wsvec, legacy)
    if centres is not None and (
        cell_matrix is None
        or centres.shape != (nw, 3)
        or not np.isfinite(centres).all()
    ):
        raise ValueError(
            "Centres require a matching cell and one finite xyz row per Wannier function"
        )
    onsite = np.real(h[np.all(r == 0, axis=1)][0].diagonal()).tolist()
    hops = []
    shells: dict[float, float] = {}
    max_intercell = 0.0
    for ir, vector in enumerate(r):
        for m in range(nw):
            for n in range(nw):
                value = h[ir, m, n]
                magnitude = float(abs(value))
                if np.any(vector):
                    max_intercell = max(max_intercell, magnitude)
                translations = (
                    shifts[(*vector, m, n)]
                    if shifts is not None
                    else np.zeros((1, 3), dtype=int)
                )
                distance = None
                if cell_matrix is not None:
                    if centres is None:
                        distance = float(np.linalg.norm(vector @ cell_matrix))
                    else:
                        displacements = (
                            (vector + translations) @ cell_matrix
                            + centres[n]
                            - centres[m]
                        )
                        distance = float(np.min(np.linalg.norm(displacements, axis=1)))
                    if m != n or np.any(vector):
                        shell = round(distance, 3)
                        shells[shell] = max(shells.get(shell, 0.0), magnitude)
                if magnitude >= threshold:
                    hops.append(
                        {
                            "Rx": int(vector[0]),
                            "Ry": int(vector[1]),
                            "Rz": int(vector[2]),
                            "orb_m": m + 1,
                            "orb_n": n + 1,
                            "degeneracy": int(deg[ir]),
                            "ws_multiplicity": len(translations),
                            "distance_angstrom": distance,
                            "folded_re_eV": float(value.real),
                            "folded_im_eV": float(value.imag),
                            "folded_magnitude_eV": magnitude,
                        }
                    )
    hops.sort(
        key=lambda row: (row["distance_angstrom"] or 0, -row["folded_magnitude_eV"])
    )
    summary = {
        "num_wann": nw,
        "nrpts": nr,
        **convention,
        "onsite_energies_eV": onsite,
        "mean_onsite_eV": float(np.mean(onsite)),
        "threshold_eV": threshold,
        "num_hoppings_above_threshold": len(hops),
        "max_hopping_intercell_eV": max_intercell,
        "matrix_convention": "folded H(R); Fourier weights and ws shifts are not applied to amplitudes in this table",
        "distance_kind": "orbital_pair"
        if centres is not None
        else ("lattice_translation" if cell_matrix is not None else "not_computed"),
        "decay_profile": [
            {"distance_angstrom": d, "max_folded_hopping_eV": v}
            for d, v in sorted(shells.items())
        ],
    }
    return summary, hops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hr_file", type=Path)
    parser.add_argument("--win", type=Path, help="Unit cell for distances in Angstrom")
    parser.add_argument(
        "--wout",
        type=Path,
        help="Final Wannier centres for orbital-pair distances; requires --win",
    )
    parser.add_argument("--wsvec", type=Path)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()
    if args.threshold < 0 or not np.isfinite(args.threshold):
        parser.error("threshold must be finite and nonnegative")
    if args.wout and not args.win:
        parser.error("--wout requires --win")
    out = save_config(args.output_dir, "parse_hr", vars(args))
    with failure_report(out / "tb_hamiltonian.json", nested_validation=False):
        cell = read_cell(args.win) if args.win else None
        centres = None
        if args.wout:
            result = parse_wout(args.wout.read_text())
            centres = np.asarray(result["wf_centres"])
            if result["length_unit"] == "Bohr":
                centres *= BOHR_ANGSTROM
        summary, hops = parse_hamiltonian(
            args.hr_file, args.threshold, cell, centres, args.wsvec, args.legacy
        )
        write_json(out / "tb_hamiltonian.json", summary)
        columns = [
            "Rx",
            "Ry",
            "Rz",
            "orb_m",
            "orb_n",
            "degeneracy",
            "ws_multiplicity",
            "distance_angstrom",
            "folded_re_eV",
            "folded_im_eV",
            "folded_magnitude_eV",
        ]
        with (out / "tb_hoppings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(hops)
        print(
            f"{summary['num_wann']} WFs, {summary['nrpts']} R vectors; distances: {summary['distance_kind']}"
        )


if __name__ == "__main__":
    main()
