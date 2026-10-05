#!/usr/bin/env python3
"""
Predict LC-MS/MS spectra from SMILES via ICEBERG (two-stage DAG + intensity GNN).

Runs inference, saves fragment SMILES assignments, and plots the predicted spectrum.

Usage:
    venv/run msms python skills/chem-msms-predict/scripts/predict_msms.py \\
        --smiles "c1ccccc1C(=O)OCCN" \\
        --gen_ckpt downloads/iceberg_msg_all/gen/best.ckpt \\
        --inten_ckpt downloads/iceberg_msg_all/inten_contr/best.ckpt \\
        --collision_energies 20 40 \\
        --output_dir results/msms_prediction

Requirements:
    - Environment: msms (run with: venv/run msms python ...; x86_64 only)
    - ICEBERG 2.1 checkpoints, fetched by download_weights.py:
      downloads/iceberg_msg_all/gen/best.ckpt
      downloads/iceberg_msg_all/inten_contr/best.ckpt
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def run_iceberg(
    smiles: str,
    gen_ckpt: Path,
    inten_ckpt: Path,
    collision_energies: list,
    adduct: str,
    instrument: str,
    batch_size: int,
    num_workers: int,
    sparse_k: int,
    max_nodes: int,
    threshold: float,
) -> tuple:
    """Run ICEBERG two-stage inference on the CPU. Returns (save_dir, precursor_mass)."""
    from ms_pred.iceberg.iceberg_elucidation import iceberg_prediction

    # torch >= 2.6 loads checkpoints weights-only by default, which rejects the
    # pathlib objects in ICEBERG's saved hyperparameters. The weights come from
    # the ms-pred authors, so the prediction subprocess may unpickle them fully.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    save_dir, precursor_mass = iceberg_prediction(
        candidate_smiles=[smiles],
        collision_energies=collision_energies,
        nce=False,
        adduct=adduct,
        instrument=instrument,
        exp_name="skill_pred",
        python_path=sys.executable,
        gen_ckpt=str(gen_ckpt),
        inten_ckpt=str(inten_ckpt),
        cuda_devices=None,
        batch_size=batch_size,
        num_cpu_workers=num_workers,
        sparse_k=sparse_k,
        max_nodes=max_nodes,
        threshold=threshold,
        binned_out=False,
        force_recompute=True,
    )
    # iceberg_prediction reports a failed prediction run only by not writing this.
    if not (Path(save_dir) / "iceberg_run_successful").exists():
        sys.exit("ICEBERG prediction failed; see its output above.")
    return save_dir, precursor_mass


def load_predictions(save_dir: Path) -> tuple:
    """
    Load predicted spectra and fragment SMILES from ICEBERG HDF5 output.

    Returns:
        spec_dict: {collision_energy_str -> (K,2) ndarray of [mz, intensity]}
        frag_dict: {collision_energy_str -> list of fragment SMILES}
        canonical_smi: SMILES as stored in HDF5
    """
    from ms_pred.iceberg.iceberg_elucidation import load_pred_spec
    from rdkit import Chem

    smiles, pred_specs = load_pred_spec(save_dir)
    canonical_smi, composite = smiles[0], pred_specs[0]
    # Each fragment is a mask over the atoms of the root molecule as ICEBERG
    # parses it. Kekulized, so a fragment that cuts an aromatic ring is still
    # valid SMILES.
    mol = Chem.MolFromSmiles(canonical_smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    spec_dict, frag_dict = {}, {}
    for ce, ms in composite.items():
        spec_dict[ce] = ms.spec
        frag_dict[ce] = [
            Chem.MolFragmentToSmiles(
                mol, atomsToUse=np.flatnonzero(mask).tolist(), kekuleSmiles=True
            )
            for mask in (ms.frags if ms.has_frags else [])
        ]
    return spec_dict, frag_dict, canonical_smi


def plot_spectrum(
    spec_dict: dict,
    smiles: str,
    precursor_mass: float,
    adduct: str,
    output_path: Path,
) -> None:
    """Stem plot of predicted MS/MS spectrum, one panel per collision energy."""
    ces = sorted(spec_dict.keys(), key=lambda x: float(x))
    n = len(ces)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), squeeze=False)

    for ax, ce in zip(axes[:, 0], ces):
        spec = spec_dict[ce]
        mz = spec[:, 0]
        inten = spec[:, 1] / spec[:, 1].max()

        _, stemlines, _ = ax.stem(mz, inten, linefmt="C0-", markerfmt=" ", basefmt="k-")
        plt.setp(stemlines, linewidth=0.8)

        ax.axvline(
            precursor_mass,
            color="red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label=f"precursor {adduct} ({precursor_mass:.4f} Da)",
        )

        for i in np.argsort(inten)[::-1][:5]:
            ax.text(
                mz[i],
                inten[i] + 0.02,
                f"{mz[i]:.2f}",
                fontsize=7,
                ha="center",
                va="bottom",
                color="C0",
            )

        try:
            ce_label = f"{float(ce):.0f} eV"
        except ValueError:
            ce_label = str(ce)

        ax.set_title(f"{smiles}  |  CE: {ce_label}", fontsize=9)
        ax.set_xlabel("m/z", fontsize=9)
        ax.set_ylabel("Relative Intensity", fontsize=9)
        ax.set_ylim(-0.05, 1.3)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Spectrum plot saved: {output_path}")


def save_fragments(spec_dict: dict, frag_dict: dict, output_path: Path) -> None:
    """Save {ce -> [{mz, intensity, fragment_smiles}]} sorted by intensity descending."""
    result = {}
    for ce in spec_dict:
        spec = spec_dict[ce]
        frags = frag_dict.get(ce, [])
        max_inten = spec[:, 1].max()
        entries = [
            {
                "mz": float(spec[i, 0]),
                "intensity": float(spec[i, 1] / max_inten),
                "fragment_smiles": frags[i] if i < len(frags) else None,
            }
            for i in range(len(spec))
        ]
        entries.sort(key=lambda x: x["intensity"], reverse=True)
        result[ce] = entries
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Fragment assignments saved: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict LC-MS/MS spectra via ICEBERG")
    p.add_argument("--smiles", required=True, help="Input SMILES string")
    p.add_argument(
        "--gen_ckpt",
        required=True,
        type=Path,
        help="ICEBERG generator checkpoint (.ckpt)",
    )
    p.add_argument(
        "--inten_ckpt",
        required=True,
        type=Path,
        help="ICEBERG intensity checkpoint (.ckpt)",
    )
    p.add_argument(
        "--collision_energies",
        nargs="+",
        type=int,
        default=[20, 40],
        help="Collision energies in eV (default: 20 40)",
    )
    p.add_argument("--adduct", default="[M+H]+", help="Adduct type (default: [M+H]+)")
    p.add_argument(
        "--instrument",
        default="Orbitrap",
        choices=["Orbitrap", "QTOF"],
        help="Instrument the MSG checkpoints condition on (default: Orbitrap)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/msms_prediction"),
        help="Output directory",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--num_workers", type=int, default=0, help="Parallel CPU workers (0: serial)"
    )
    p.add_argument("--sparse_k", type=int, default=100, help="Top-K peaks to output")
    p.add_argument("--max_nodes", type=int, default=100, help="Max fragment DAG nodes")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Fragment generator confidence cutoff (default: 0.1)",
    )
    p.add_argument(
        "--no_fragments",
        action="store_true",
        help="Skip saving fragment SMILES assignments",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for ckpt in (args.gen_ckpt, args.inten_ckpt):
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Fetch the ICEBERG 2.1 weights with "
                "skills/chem-msms-predict/scripts/download_weights.py."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with open(args.output_dir / "input_configs.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Running ICEBERG for: {args.smiles}")
    save_dir, precursor_mass = run_iceberg(
        smiles=args.smiles,
        gen_ckpt=args.gen_ckpt,
        inten_ckpt=args.inten_ckpt,
        collision_energies=args.collision_energies,
        adduct=args.adduct,
        instrument=args.instrument,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sparse_k=args.sparse_k,
        max_nodes=args.max_nodes,
        threshold=args.threshold,
    )
    print(f"Precursor mass ({args.adduct}): {precursor_mass:.4f} Da")

    spec_dict, frag_dict, canonical_smi = load_predictions(save_dir)

    plot_spectrum(
        spec_dict=spec_dict,
        smiles=canonical_smi,
        precursor_mass=precursor_mass,
        adduct=args.adduct,
        output_path=args.output_dir / "spectrum.png",
    )

    if not args.no_fragments:
        save_fragments(
            spec_dict=spec_dict,
            frag_dict=frag_dict,
            output_path=args.output_dir / "fragments.json",
        )

    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
