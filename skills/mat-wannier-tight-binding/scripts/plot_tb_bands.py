#!/usr/bin/env python3
"""Plot band-separated distance/energy data with optional reference and labels.

Usage: venv/run cpu python plot_tb_bands.py results/tb_interpolated_bands.dat --ref-bands seed_band.dat --labelinfo seed_band.labelinfo.dat --output-dir results
Requirements: cpu environment, matplotlib, NumPy and PyYAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wannier_tb.io import read_bands, save_config

plt.rcParams.update({"font.size": 14})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tb_bands", type=Path)
    parser.add_argument("--ref-bands", type=Path)
    parser.add_argument("--reference-label", default="Wannier90")
    parser.add_argument("--ref-energy-shift", type=float, default=0.0)
    parser.add_argument("--labelinfo", type=Path, help="Wannier90 *_band.labelinfo.dat")
    parser.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Explicit energy zero to subtract (eV)",
    )
    parser.add_argument("--energy-range", type=float, nargs=2)
    parser.add_argument(
        "--x-label",
        default="Wave vector",
        help="Use k-point index for index-only interpolation",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="tb_bands")
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("prefix must be a filename stem")
    out = save_config(args.output_dir, "plot_tb_bands", vars(args))
    distances, bands = read_bands(args.tb_bands)
    zero = args.fermi if args.fermi is not None else 0.0
    bands -= zero
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, band in enumerate(bands.T):
        ax.plot(
            distances,
            band,
            color="#1f77b4",
            linewidth=2.5,
            label="Python TB" if i == 0 else None,
        )
    if args.ref_bands:
        ref_x, ref = read_bands(args.ref_bands)
        if ref_x.shape != distances.shape or not np.allclose(
            ref_x, distances, atol=1e-7, rtol=0
        ):
            raise ValueError("Reference path distances differ from the plotted bands")
        step = max(1, len(distances) // 45)
        for i, band in enumerate(ref.T):
            ax.plot(
                ref_x[::step],
                (band + args.ref_energy_shift - zero)[::step],
                "o",
                color="#d62728",
                markersize=3,
                markerfacecolor="none",
                label=args.reference_label if i == 0 else None,
            )
    if args.labelinfo:
        ticks: dict[float, str] = {}
        for line in args.labelinfo.read_text().splitlines():
            row = line.split()
            if not row:
                continue
            index, position = int(row[1]) - 1, float(row[2])
            if not 0 <= index < len(distances) or not np.isclose(
                distances[index], position, atol=1e-7, rtol=0
            ):
                raise ValueError("Label positions do not match the plotted path")
            position = float(
                distances[index]
            )  # Keep endpoint ticks inside the axis after text rounding.
            label = r"$\Gamma$" if row[0] in ("G", "GAMMA") else row[0]
            old = ticks.get(position)
            ticks[position] = (
                label if old is None or old == label else old + "|" + label
            )
        ax.set_xticks(list(ticks), list(ticks.values()))
        for position in ticks:
            ax.axvline(position, color="0.8", linewidth=0.6, zorder=0)
    if args.fermi is not None:
        ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel(args.x_label, fontweight="bold")
    ax.set_ylabel(
        "Energy (eV)" if args.fermi is None else r"Energy $- E_F$ (eV)",
        fontweight="bold",
    )
    if len(distances) > 1:
        ax.set_xlim(distances[0], distances[-1])
    if args.energy_range:
        ax.set_ylim(*args.energy_range)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(out / f"{args.prefix}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / args.prefix}.png and .svg")


if __name__ == "__main__":
    main()
