#!/usr/bin/env python3
"""
Register predicted or retrieved spectra into the local spectrum catalog.

Reads a source directory (output of chem-nmr-predict, chem-db-spectra, or a custom
predictor) and upserts entries keyed by (canonical_smiles, modality) into catalog.json.

Supported modalities: nmr_1h, ir

Usage:
    # Env: nmr-agent
    python register_spectrum.py \\
        --source_dir research/nmr_predictions/ \\
        --modality nmr_1h \\
        --catalog_dir research/spectrum_catalog/

    # Env: base-agent  (for IR from chem-db-spectra)
    python register_spectrum.py \\
        --source_dir research/ir_references/ \\
        --modality ir \\
        --smiles "OC1CC2CCC1C2" \\
        --names "borneol" \\
        --catalog_dir research/spectrum_catalog/

Requirements:
    - Conda environment: nmr-agent (or base-agent for IR-only registration)
    - Required packages: rdkit
"""

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

try:
    from src.utils.config_utils import save_skill_inputs as _save_skill_inputs
except ImportError:
    _save_skill_inputs = None


def mol_identifiers(smiles: str) -> dict:
    """
    Compute canonical SMILES, InChIKey, and InChIKey-14 from a SMILES string.

    InChIKey-14 is the first 14 characters of the InChIKey (connectivity layer only),
    which ignores stereochemistry and allows diastereomer/enantiomer-insensitive lookup.

    Returns dict with keys: canonical_smiles, inchikey, inchikey14.
    Raises ValueError if SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol)

    inchikey = MolToInchiKey(mol) or ""
    inchikey14 = inchikey[:14] if inchikey else ""
    return {
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "inchikey14": inchikey14,
    }


def canonicalize(smiles: str) -> str:
    """Return RDKit canonical SMILES, raise ValueError if invalid."""
    return mol_identifiers(smiles)["canonical_smiles"]


def catalog_key(canonical_smiles: str, modality: str) -> str:
    return f"{canonical_smiles}|{modality}"


def load_catalog(catalog_path: pathlib.Path) -> dict:
    if catalog_path.exists():
        return json.loads(catalog_path.read_text())
    return {}


def save_catalog(catalog: dict, catalog_path: pathlib.Path) -> None:
    catalog_path.write_text(json.dumps(catalog, indent=2))


def register_from_nmr_predict(
    source_dir: pathlib.Path,
    modality: str,
    catalog: dict,
    overwrite: bool,
) -> tuple[int, int]:
    """
    Register entries from a chem-nmr-predict output directory.

    Reads predictions.json manifest; each 'found' entry contains smiles + file paths.
    Returns (registered_count, skipped_count).
    """
    manifest_path = source_dir / "predictions.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"predictions.json not found in {source_dir}")

    manifest = json.loads(manifest_path.read_text())
    found = manifest.get("found", [])
    registered = skipped = 0

    for entry in found:
        smiles = entry.get("smiles", "")
        name = entry.get("name", "")
        spectrum = entry.get("spectrum", "")

        try:
            ids = mol_identifiers(smiles)
        except ValueError as e:
            print(f"  SKIP {name}: {e}")
            skipped += 1
            continue

        key = catalog_key(ids["canonical_smiles"], modality)
        if key in catalog and not overwrite:
            print(f"  SKIP {name}: already in catalog (--overwrite to replace)")
            skipped += 1
            continue

        catalog[key] = {
            "name": name,
            "smiles_input": smiles,
            "canonical_smiles": ids["canonical_smiles"],
            "inchikey": ids["inchikey"],
            "inchikey14": ids["inchikey14"],
            "modality": modality,
            "spectrum": str(pathlib.Path(spectrum).resolve()),
            "signals": str(pathlib.Path(entry.get("signals", "")).resolve())
            if entry.get("signals")
            else "",
            "n_signals": entry.get("n_signals", 0),
            "n_atoms_h": entry.get("n_atoms_h", 0),
            "parameters": manifest.get("parameters", {}),
        }
        print(
            f"  Registered {name} ({ids['canonical_smiles']}) [{modality}]  InChIKey: {ids['inchikey']}"
        )
        registered += 1

    return registered, skipped


def register_from_files(
    source_dir: pathlib.Path,
    modality: str,
    smiles_list: list[str],
    names: list[str],
    catalog: dict,
    overwrite: bool,
) -> tuple[int, int]:
    """
    Register spectrum files by matching filenames to provided SMILES/names.

    Used for IR files from chem-db-spectra or other predictors that do not
    produce a predictions.json manifest.
    """
    registered = skipped = 0

    for smiles, name in zip(smiles_list, names):
        safe_name = re.sub(r"[^\w\-]", "_", name)[:40]

        try:
            ids = mol_identifiers(smiles)
        except ValueError as e:
            print(f"  SKIP {name}: {e}")
            skipped += 1
            continue

        # Find matching spectrum file (prefer .xy, then .jdx, then .csv)
        candidates = (
            list(source_dir.glob(f"{safe_name}*.xy"))
            + list(source_dir.glob(f"{safe_name}*.jdx"))
            + list(source_dir.glob(f"{safe_name}*.csv"))
        )
        if not candidates:
            print(f"  SKIP {name}: no spectrum file found in {source_dir}")
            skipped += 1
            continue

        spectrum_path = candidates[0]
        key = catalog_key(ids["canonical_smiles"], modality)

        if key in catalog and not overwrite:
            print(f"  SKIP {name}: already in catalog (--overwrite to replace)")
            skipped += 1
            continue

        catalog[key] = {
            "name": name,
            "smiles_input": smiles,
            "canonical_smiles": ids["canonical_smiles"],
            "inchikey": ids["inchikey"],
            "inchikey14": ids["inchikey14"],
            "modality": modality,
            "spectrum": str(spectrum_path.resolve()),
            "signals": "",
            "parameters": {},
        }
        print(
            f"  Registered {name} ({ids['canonical_smiles']}) [{modality}]  InChIKey: {ids['inchikey']}"
        )
        registered += 1

    return registered, skipped


def main():
    ap = argparse.ArgumentParser(
        description="Register spectra into the local spectrum catalog."
    )
    ap.add_argument(
        "--source_dir",
        required=True,
        help="Directory containing spectrum files (and optionally predictions.json)",
    )
    ap.add_argument(
        "--modality",
        choices=("nmr_1h", "ir"),
        required=True,
        help="Spectrum modality to register under",
    )
    ap.add_argument(
        "--catalog_dir",
        default="research/spectrum_catalog",
        help="Local spectrum catalog directory (default: research/spectrum_catalog/)",
    )
    ap.add_argument(
        "--smiles",
        nargs="+",
        help="SMILES for each spectrum file (required when no predictions.json present)",
    )
    ap.add_argument(
        "--names",
        nargs="+",
        help="Names matching --smiles order (required when no predictions.json present)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing catalog entries for the same (SMILES, modality)",
    )
    args = ap.parse_args()

    source_dir = pathlib.Path(args.source_dir)
    catalog_dir = pathlib.Path(args.catalog_dir)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "catalog.json"
    catalog = load_catalog(catalog_path)

    manifest_path = source_dir / "predictions.json"

    if manifest_path.exists():
        registered, skipped = register_from_nmr_predict(
            source_dir, args.modality, catalog, args.overwrite
        )
    else:
        if not args.smiles or not args.names:
            ap.error(
                "--smiles and --names required when predictions.json is absent in source_dir"
            )
        if len(args.smiles) != len(args.names):
            ap.error("--smiles and --names must have equal length")
        registered, skipped = register_from_files(
            source_dir, args.modality, args.smiles, args.names, catalog, args.overwrite
        )

    save_catalog(catalog, catalog_path)
    print(f"\nDone: {registered} registered, {skipped} skipped.")
    print(f"Catalog -> {catalog_path} ({len(catalog)} total entries)")

    if _save_skill_inputs is not None:
        _save_skill_inputs(args, args.catalog_dir)


if __name__ == "__main__":
    main()
