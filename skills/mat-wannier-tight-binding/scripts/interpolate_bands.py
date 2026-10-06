#!/usr/bin/env python3
"""Interpolate standard folded Wannier90 Hamiltonians with Wigner-Seitz shifts.

Usage: venv/run cpu python interpolate_bands.py seed_hr.dat --kpoints seed_band.kpt --win seed.win --ref-bands seed_band.dat --output-dir results
Requirements: cpu environment, NumPy and PyYAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wannier_tb.hamiltonian import interpolate
from wannier_tb.io import (
    failure_report,
    read_bands,
    read_cell,
    read_hr,
    read_kpoints,
    resolve_wsvec,
    save_config,
    sha256,
    write_bands,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hr_file", type=Path)
    parser.add_argument("--kpoints", required=True, type=Path)
    parser.add_argument(
        "--win",
        type=Path,
        help="Cell for reciprocal-distance axis when no reference path is supplied",
    )
    parser.add_argument("--wsvec", type=Path, help="Defaults to sidecar next to hr.dat")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Allow absent sidecar only for known use_ws_distance=false output",
    )
    parser.add_argument(
        "--ref-bands", type=Path, help="Band-separated distance/energy pairs"
    )
    parser.add_argument(
        "--ref-kpoints",
        type=Path,
        help="Reference coordinates; defaults to the .kpt next to reference .dat",
    )
    parser.add_argument(
        "--reference-kind", choices=["wannier90", "dft"], default="wannier90"
    )
    parser.add_argument(
        "--ref-band-indices",
        type=int,
        nargs="+",
        help="One-based reference columns defining the target manifold",
    )
    parser.add_argument(
        "--ref-energy-shift",
        type=float,
        default=0.0,
        help="Add this explicit energy alignment to reference energies (eV)",
    )
    parser.add_argument(
        "--energy-window",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Compare only states whose aligned reference energies lie in this range",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=1e-3,
        help="Maximum allowed comparison error (eV)",
    )
    parser.add_argument(
        "--hermiticity-tol",
        type=float,
        default=1e-5,
        help="Maximum |H-H†| before roundoff cleanup (eV)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if (
        not all(
            np.isfinite(v)
            for v in (args.max_error, args.hermiticity_tol, args.ref_energy_shift)
        )
        or min(args.max_error, args.hermiticity_tol) <= 0
    ):
        parser.error(
            "Tolerances must be finite and positive; energy shift must be finite"
        )
    if args.energy_window and (
        not np.isfinite(args.energy_window).all()
        or args.energy_window[0] > args.energy_window[1]
    ):
        parser.error("Invalid energy window")
    if not args.ref_bands and (
        args.ref_kpoints
        or args.ref_band_indices
        or args.energy_window
        or args.ref_energy_shift
    ):
        parser.error("Reference comparison options require --ref-bands")
    out = save_config(args.output_dir, "interpolate_bands", vars(args))
    with failure_report(out / "band_comparison.json", nested_validation=False):
        nw, nr, deg, r, h = read_hr(args.hr_file)
        shifts, convention = resolve_wsvec(args.hr_file, r, nw, args.wsvec, args.legacy)
        kpoints = read_kpoints(args.kpoints)
        bands, residual = interpolate(r, h, deg, kpoints, shifts, args.hermiticity_tol)
        summary = {
            "num_kpoints": len(kpoints),
            "num_bands": nw,
            "nrpts": nr,
            **convention,
            "hr_sha256": sha256(args.hr_file),
            "max_hermiticity_residual_eV": residual,
            "has_reference": args.ref_bands is not None,
            "status": "NOT_CHECKED",
        }
        if args.ref_bands:
            ref_path = args.ref_kpoints or args.ref_bands.with_suffix(".kpt")
            ref_kpoints = read_kpoints(ref_path)
            if ref_kpoints.shape != kpoints.shape or not np.allclose(
                ref_kpoints, kpoints, atol=1e-7, rtol=0
            ):
                raise ValueError(
                    "Reference k-point coordinates/order differ from interpolation points"
                )
            distances, reference = read_bands(args.ref_bands)
            if args.ref_band_indices:
                indices = np.asarray(args.ref_band_indices) - 1
                if (
                    len(np.unique(indices)) != len(indices)
                    or np.any(indices < 0)
                    or np.any(indices >= reference.shape[1])
                ):
                    raise ValueError(
                        "Reference band indices are duplicated or out of range"
                    )
                reference = reference[:, indices]
            if reference.shape != bands.shape:
                raise ValueError(
                    f"Reference shape {reference.shape} differs from model {bands.shape}; select the target reference bands explicitly"
                )
            reference = np.sort(reference + args.ref_energy_shift, axis=1)
            mask = np.ones(reference.shape, dtype=bool)
            if args.energy_window:
                mask = (reference >= args.energy_window[0]) & (
                    reference <= args.energy_window[1]
                )
            if not mask.any():
                raise ValueError(
                    "No reference states lie in the requested energy window"
                )
            error = (bands - reference)[mask]
            maximum = float(np.max(np.abs(error)))
            summary.update(
                {
                    "reference_kind": args.reference_kind,
                    "reference_kpoints": str(ref_path.resolve()),
                    "reference_sha256": sha256(args.ref_bands),
                    "compared_eigenvalues": int(mask.sum()),
                    "mae_eV": float(np.mean(np.abs(error))),
                    "rmse_eV": float(np.sqrt(np.mean(error**2))),
                    "max_error_eV": maximum,
                    "status": "PASSED" if maximum <= args.max_error else "FAILED",
                    "axis_kind": "reference_path_distance",
                    "validation_scope": "same-model numerical regression"
                    if args.reference_kind == "wannier90"
                    else "user-supplied independent DFT comparison",
                }
            )
        elif args.win:
            reciprocal = 2 * np.pi * np.linalg.inv(read_cell(args.win)).T
            distances = np.r_[
                0.0,
                np.cumsum(
                    np.linalg.norm(np.diff(kpoints, axis=0) @ reciprocal, axis=1)
                ),
            ]
            summary["axis_kind"] = "reciprocal_distance_inverse_angstrom"
        else:
            distances = np.arange(len(kpoints), dtype=float)
            summary["axis_kind"] = "kpoint_index"
        write_bands(out / "tb_interpolated_bands.dat", distances, bands)
        np.savetxt(
            out / "tb_interpolated_bands.kpt",
            kpoints,
            header=str(len(kpoints)),
            comments="",
            fmt="%.10f",
        )
        write_json(out / "band_comparison.json", summary)
        print(
            f"{nw} bands, {len(kpoints)} k-points, use_ws_distance={convention['use_ws_distance']}; {summary['status']}"
        )
        if args.ref_bands:
            print(
                f"Maximum {args.reference_kind} comparison error: {summary['max_error_eV']:.8g} eV"
            )
        if summary["status"] == "FAILED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
