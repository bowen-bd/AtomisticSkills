#!/usr/bin/env python3
"""Parse final Wannier90 spreads and distinguish completion from convergence.

Usage: venv/run cpu python parse_wout.py seed.wout --output-dir results
Requirements: cpu environment, NumPy and PyYAML through wannier_tb.io.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

from wannier_tb.io import failure_report, save_config, write_json

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def parse_wout(text: str) -> dict[str, Any]:
    """Require a complete final state and return spreads and convergence evidence."""
    blocks = list(re.finditer(r"^\s*Final State\s*$", text, re.M))
    if not blocks or "All done: wannier90 exiting" not in text[blocks[-1].end() :]:
        raise ValueError("Missing final state or normal Wannier90 termination")
    tail = text[blocks[-1].end() :]
    matches = list(
        re.finditer(
            rf"WF centre and spread\s+(\d+)\s*\(\s*({NUMBER}),\s*({NUMBER}),\s*({NUMBER})\s*\)\s+({NUMBER})",
            tail,
        )
    )
    expected = re.findall(r"Number of Wannier Functions\s*:\s*(\d+)", text)
    if not expected or [int(m[1]) for m in matches] != list(
        range(1, int(expected[-1]) + 1)
    ):
        raise ValueError("Missing, duplicate or inconsistent final Wannier functions")
    centres = [
        [float(m[i].replace("D", "E").replace("d", "e")) for i in (2, 3, 4)]
        for m in matches
    ]
    spreads = [float(m[5].replace("D", "E").replace("d", "e")) for m in matches]
    result: dict[str, Any] = {}
    for label, key in [
        ("I", "omega_I"),
        ("D", "omega_D"),
        ("OD", "omega_OD"),
        ("Total", "omega_total"),
    ]:
        match = re.search(rf"Omega\s+{label}\s*=\s*({NUMBER})", tail)
        if match is None:
            raise ValueError(f"Missing final Omega {label}")
        result[key] = float(match[1].replace("D", "E").replace("d", "e"))
    if not np.isfinite([*result.values(), *spreads, *np.ravel(centres)]).all():
        raise ValueError("Nonfinite spread or centre in final state")
    if min(*spreads, *result.values()) < -1e-7:
        raise ValueError("Negative spread in final state")
    total = result["omega_total"]
    if not np.isclose(sum(spreads), total, atol=1e-6, rtol=1e-7) or not np.isclose(
        result["omega_I"] + result["omega_D"] + result["omega_OD"],
        total,
        atol=1e-6,
        rtol=1e-7,
    ):
        raise ValueError("Final spreads do not sum to Omega Total")
    settings = re.search(r"\*-+ WANNIERISE -+\*(.*?)\*-+", text, re.S)
    window = (
        re.search(r"Convergence window\s*:\s*(-?\d+)", settings[1])
        if settings
        else None
    )
    localization_converged = "Wannierisation convergence criteria satisfied" in text
    localization_status = (
        "converged"
        if localization_converged
        else ("not_requested" if window and int(window[1]) < 0 else "not_established")
    )
    dis_match = re.search(rf"Final\s+Omega_I\s+({NUMBER})", text)
    has_dis = dis_match is not None or bool(
        re.search(r"Using band disentanglement\s*:\s*T", text)
    )
    dis_converged = "Disentanglement convergence criteria satisfied" in text
    iterations = re.findall(
        rf"^\s*(\d+)\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+<-- CONV",
        text,
        re.M,
    )
    result.update(
        {
            "num_wann": len(spreads),
            "length_unit": "Bohr" if "Spreads (Bohr^2)" in tail else "Ang",
            "wf_centres": centres,
            "wf_spreads": spreads,
            "normal_termination": True,
            "has_disentanglement": has_dis,
            "final_disentangle_omega_I": float(dis_match[1].replace("D", "E"))
            if dis_match
            else None,
            "localization_status": localization_status,
            "disentanglement_status": (
                "converged" if dis_converged else "not_established"
            )
            if has_dis
            else "not_applicable",
            "last_printed_localization_iteration": int(iterations[-1][0])
            if iterations
            else None,
            "last_printed_spread_change": float(iterations[-1][1].replace("D", "E"))
            if iterations
            else None,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--max-omega-tot",
        type=float,
        default=None,
        help="Optional system-specific localization bound",
    )
    parser.add_argument("--max-omega-d", type=float, default=None)
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Fail unless native convergence criteria were satisfied",
    )
    args = parser.parse_args()
    for value in (args.max_omega_tot, args.max_omega_d):
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error("Spread bounds must be finite and nonnegative")
    out = save_config(args.output_dir, "parse_wout", vars(args))
    with failure_report(out / "wout_summary.json", nested_validation=True):
        # Input errors deliberately propagate with a nonzero exit; never fabricate a passing summary.
        summary = parse_wout(args.wout.read_text())
        failures = []
        for field, bound in [
            ("omega_total", args.max_omega_tot),
            ("omega_D", args.max_omega_d),
        ]:
            if bound is not None and summary[field] > bound:
                failures.append(f'{field} exceeds {bound} {summary["length_unit"]}^2')
        converged = summary["localization_status"] == "converged" and summary[
            "disentanglement_status"
        ] in ("converged", "not_applicable")
        if args.require_converged and not converged:
            failures.append("Native iterative convergence is not established")
        summary["validation"] = {
            "status": "FAILED"
            if failures
            else ("PASSED" if converged else "CONVERGENCE_NOT_ESTABLISHED"),
            "failures": failures,
        }
        write_json(out / "wout_summary.json", summary)
        print(
            f"{args.wout.name}: {summary['num_wann']} WFs; Omega={summary['omega_total']:.9f} {summary['length_unit']}^2; {summary['validation']['status']}"
        )
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
