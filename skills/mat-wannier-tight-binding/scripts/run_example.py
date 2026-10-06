#!/usr/bin/env python3
"""Run a bundled Wannier90 example in a fresh output directory.

Usage: venv/run cpu python run_example.py silicon-sp3 --wannier90 /path/to/wannier90.x --output-dir /path/to/new/run --plot
Requirements: cpu environment, NumPy, PyYAML, matplotlib for --plot; external Wannier90.
Uses precomputed tutorial DFT matrices; does not run Quantum ESPRESSO.
"""

from __future__ import annotations

import argparse
import lzma
from pathlib import Path
import shutil
import subprocess
import sys

from wannier_tb.io import save_config, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=["gaas-valence", "silicon-sp3"])
    parser.add_argument("--wannier90", default="wannier90.x")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    binary = shutil.which(args.wannier90)
    if binary is None:
        parser.error(
            "Wannier90 executable not found; build it with build_wannier90.sh or supply --wannier90"
        )
    binary = str(Path(binary).resolve())
    version = subprocess.run(
        [binary, "-v"], check=True, text=True, capture_output=True
    ).stdout.strip()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(
            "Use an empty output directory to avoid reusing stale calculation files"
        )
    scripts = Path(__file__).resolve().parent
    example = scripts.parent / "examples" / args.example
    seed = "silicon" if args.example == "silicon-sp3" else "gaas"
    save_config(
        out,
        "run_example",
        {**vars(args), "resolved_binary": binary, "version": version},
    )
    provenance = {}
    suffixes = (
        ("win", "amn", "mmn", "eig") if seed == "silicon" else ("win", "amn", "mmn")
    )
    for suffix in suffixes:
        source = example / f"{seed}.{suffix}"
        destination = out / source.name
        if source.exists():
            shutil.copy2(source, destination)
        else:
            with lzma.open(source.with_suffix(source.suffix + ".xz"), "rb") as packed:
                with destination.open("wb") as unpacked:
                    shutil.copyfileobj(packed, unpacked)
        provenance[source.name] = sha256(destination)
    with (out / "wannier90.stdout.log").open("w") as log:
        subprocess.run(
            [binary, seed], cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    analysis = out / "analysis"
    commands = [
        [
            "parse_wout.py",
            f"{seed}.wout",
            "--require-converged",
            "--output-dir",
            str(analysis),
        ],
    ]
    if seed == "silicon":
        commands.extend(
            [
                [
                    "parse_hr.py",
                    f"{seed}_hr.dat",
                    "--win",
                    f"{seed}.win",
                    "--wout",
                    f"{seed}.wout",
                    "--output-dir",
                    str(analysis),
                ],
                [
                    "interpolate_bands.py",
                    f"{seed}_hr.dat",
                    "--kpoints",
                    f"{seed}_band.kpt",
                    "--win",
                    f"{seed}.win",
                    "--ref-bands",
                    f"{seed}_band.dat",
                    "--reference-kind",
                    "wannier90",
                    "--max-error",
                    "5e-5",
                    "--output-dir",
                    str(analysis),
                ],
            ]
        )
        if args.plot:
            commands.append(
                [
                    "plot_tb_bands.py",
                    str(analysis / "tb_interpolated_bands.dat"),
                    "--ref-bands",
                    f"{seed}_band.dat",
                    "--labelinfo",
                    f"{seed}_band.labelinfo.dat",
                    "--prefix",
                    "silicon_tb_bands",
                    "--output-dir",
                    str(analysis),
                ]
            )
    for command in commands:
        subprocess.run(
            [sys.executable, str(scripts / command[0]), *command[1:]],
            cwd=out,
            check=True,
        )
    write_json(
        out / "execution.json",
        {
            "wannier90_version": version,
            "binary": binary,
            "input_sha256": provenance,
            "example": args.example,
            "dft_executed": False,
            "status": "PASSED",
        },
    )
    print(f"Completed {args.example}: {out}")


if __name__ == "__main__":
    main()
