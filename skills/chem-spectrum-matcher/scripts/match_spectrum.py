#!/usr/bin/env python3
"""
Match an experimental query spectrum against predicted or database reference spectra.

For each candidate SMILES:
  1. Look up local catalog (catalog.json).
  2. Optionally fall back to NMRShiftDB2 (nmr_1h) or NIST WebBook (ir).
  3. Compute similarity between query and each reference.
  4. Output ranked match_results.json and optional overlay plot.

Supported modalities: nmr_1h, ir
Supported metrics:    l2, cosine, wasserstein

Usage:
    # Env: nmr-agent
    python match_spectrum.py \\
        --query experimental.xy \\
        --smiles "OC1CC2CCC1C2" "OC1CC2CCC1[C@@H]2C" \\
        --names "borneol" "isoborneol" \\
        --modality nmr_1h \\
        --catalog_dir research/spectrum_catalog/ \\
        --output_dir results/spectrum_match/ \\
        --fallback_public_db \\
        --plot

Requirements:
    - Conda environment: nmr-agent
    - Required packages: numpy, scipy, rdkit, requests, matplotlib
"""

import argparse
import json
import pathlib
import re
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

try:
    from src.utils.config_utils import save_skill_inputs as _save_skill_inputs
except ImportError:
    _save_skill_inputs = None

import numpy as np
import requests


SUPPORTED_MODALITIES = ("nmr_1h", "ir")
SUPPORTED_METRICS = ("l2", "cosine", "wasserstein")
SUPPORTED_LOOKUP = ("canonical_smiles", "inchikey", "inchikey14")

# ---------------------------------------------------------------------------
# Molecule identifier helpers
# ---------------------------------------------------------------------------


def mol_identifiers(smiles: str) -> dict:
    """
    Compute canonical SMILES, InChIKey, and InChIKey-14 from a SMILES string.

    InChIKey-14 uses only the first 14 characters (connectivity layer), making
    it insensitive to stereochemistry — enantiomers and diastereomers match the
    same InChIKey-14 entry.

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
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# Spectrum I/O
# ---------------------------------------------------------------------------


def load_spectrum(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a two-column spectrum file (.xy, .csv, .tsv) or JCAMP-DX (.jdx).

    Returns (x_axis, intensity) arrays sorted by ascending x_axis.
    """
    suffix = path.suffix.lower()

    if suffix == ".jdx":
        return _load_jdx(path)

    # Auto-detect delimiter
    text = path.read_text()
    delimiter = "," if "," in text.split("\n")[0] else None
    data = np.loadtxt(path, delimiter=delimiter)
    x, y = data[:, 0], data[:, 1]
    order = np.argsort(x)
    return x[order], y[order]


def _load_jdx(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse minimal JCAMP-DX file to extract x/y arrays."""
    try:
        import jcamp

        data = jcamp.jcamp_readfile(str(path))
        x = np.array(data.get("x", []))
        y = np.array(data.get("y", []))
    except ImportError:
        # Fallback: parse ##XYDATA= (X++(Y..Y)) blocks manually
        x, y = [], []
        lines = path.read_text().splitlines()
        in_data = False
        x_factor = y_factor = 1.0
        for line in lines:
            if line.startswith("##XFACTOR="):
                x_factor = float(line.split("=")[1])
            if line.startswith("##YFACTOR="):
                y_factor = float(line.split("=")[1])
            if line.startswith("##XYDATA="):
                in_data = True
                continue
            if in_data:
                if line.startswith("##"):
                    break
                tokens = line.split()
                if tokens:
                    x.append(float(tokens[0]) * x_factor)
                    for t in tokens[1:]:
                        y.append(float(t) * y_factor)
        x, y = np.array(x), np.array(y[: len(x)])
    order = np.argsort(x)
    return x[order], y[order]


def interpolate_to_grid(
    x: np.ndarray, y: np.ndarray, x_min: float, x_max: float, n_points: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate spectrum onto a uniform grid for comparison."""
    grid = np.linspace(x_min, x_max, n_points)
    y_grid = np.interp(grid, x, y, left=0.0, right=0.0)
    return grid, y_grid


def normalize(y: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if flat."""
    y = y - y.min()
    m = y.max()
    return y / m if m > 0 else y


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


def similarity_l2(y1: np.ndarray, y2: np.ndarray) -> float:
    """1 - normalized L2 distance. Range [0, 1], higher = more similar."""
    dist = np.sqrt(np.mean((y1 - y2) ** 2))
    return float(max(0.0, 1.0 - dist))


def similarity_cosine(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cosine similarity. Range [0, 1]."""
    n1, n2 = np.linalg.norm(y1), np.linalg.norm(y2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.clip(np.dot(y1, y2) / (n1 * n2), 0.0, 1.0))


def similarity_wasserstein(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    1 - normalized Wasserstein-1 distance between two normalized distributions.

    Treats spectra as probability distributions. Range [0, 1].
    """
    from scipy.stats import wasserstein_distance

    # Normalize to probability distributions
    s1, s2 = y1.sum(), y2.sum()
    if s1 == 0 or s2 == 0:
        return 0.0
    dist = wasserstein_distance(
        np.arange(len(y1)), np.arange(len(y2)), y1 / s1, y2 / s2
    )
    # Normalize by max possible distance (full width)
    max_dist = len(y1)
    return float(max(0.0, 1.0 - dist / max_dist))


METRIC_FNS = {
    "l2": similarity_l2,
    "cosine": similarity_cosine,
    "wasserstein": similarity_wasserstein,
}


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def catalog_key(canonical_smiles: str, modality: str) -> str:
    return f"{canonical_smiles}|{modality}"


def load_catalog(catalog_dir: pathlib.Path) -> tuple[dict, dict, dict]:
    """
    Load catalog.json and build secondary indexes for inchikey and inchikey14.

    Returns (catalog, inchikey_index, inchikey14_index).
    - catalog: primary dict keyed by "canonical_smiles|modality"
    - inchikey_index: maps "inchikey|modality" → primary key (exact stereo match)
    - inchikey14_index: maps "inchikey14|modality" → list of primary keys (stereo-insensitive)
    """
    catalog_file = catalog_dir / "catalog.json"
    catalog = json.loads(catalog_file.read_text()) if catalog_file.exists() else {}

    inchikey_idx: dict[str, str] = {}
    inchikey14_idx: dict[str, list[str]] = {}

    for primary_key, entry in catalog.items():
        modality = entry.get("modality", primary_key.split("|")[-1])
        ik = entry.get("inchikey", "")
        ik14 = entry.get("inchikey14", "")
        if ik:
            inchikey_idx[f"{ik}|{modality}"] = primary_key
        if ik14:
            k14 = f"{ik14}|{modality}"
            inchikey14_idx.setdefault(k14, []).append(primary_key)

    return catalog, inchikey_idx, inchikey14_idx


def lookup_local(
    ids: dict,
    modality: str,
    catalog: dict,
    inchikey_idx: dict,
    inchikey14_idx: dict,
    output_dir: pathlib.Path,
    safe_name: str,
    lookup_by: str = "canonical_smiles",
) -> pathlib.Path | None:
    """
    Copy cached spectrum to output_dir. Return path or None on miss.

    lookup_by controls which identifier is used:
    - "canonical_smiles": exact SMILES match (stereochemistry-aware, default)
    - "inchikey": exact InChIKey match (stereochemistry-aware)
    - "inchikey14": first-14-char InChIKey match (stereochemistry-insensitive;
      returns first catalog hit when multiple stereoisomers exist)
    """
    if lookup_by == "canonical_smiles":
        primary_key = catalog_key(ids["canonical_smiles"], modality)
        entry = catalog.get(primary_key)
    elif lookup_by == "inchikey":
        primary_key = inchikey_idx.get(f"{ids['inchikey']}|{modality}")
        entry = catalog.get(primary_key) if primary_key else None
    elif lookup_by == "inchikey14":
        candidates = inchikey14_idx.get(f"{ids['inchikey14']}|{modality}", [])
        entry = catalog.get(candidates[0]) if candidates else None
    else:
        entry = None

    if entry is None:
        return None

    src = pathlib.Path(entry.get("spectrum", ""))
    if not src.exists():
        return None

    dest = output_dir / f"{safe_name}_ref{src.suffix}"
    shutil.copy(src, dest)
    return dest


# ---------------------------------------------------------------------------
# Public DB fallback
# ---------------------------------------------------------------------------

_NMRSHIFTDB2_URL = "https://nmrshiftdb.nmr.uni-koeln.de/portal/nmrshiftdb/search/searchbystructure.json"


def fetch_nmrshiftdb2(smiles: str) -> list[tuple[float, float]]:
    """Return (shift_ppm, intensity=1.0) peak list from NMRShiftDB2, or []."""
    params = {"smiles": smiles, "spectrum_type": "1H", "searchtype": "substructure"}
    try:
        resp = requests.get(_NMRSHIFTDB2_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    spectra = data.get("spectra", []) or []
    if not spectra:
        return []

    peaks = []
    for token in (spectra[0].get("peakList", "") or "").split():
        try:
            peaks.append((float(token), 1.0))
        except ValueError:
            continue
    return peaks


def fetch_nist_ir(smiles: str) -> list[tuple[float, float]]:
    """Placeholder: NIST IR retrieval requires formula-based lookup via chem-db-spectra."""
    # Full implementation delegates to chem-db-spectra/scripts/query_spectra.py
    return []


def peaks_to_xy(
    peaks: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    n_points: int = 8192,
    linewidth: float = 1.0,
    field_mhz: float = 400.0,
    modality: str = "nmr_1h",
) -> tuple[np.ndarray, np.ndarray]:
    """Broaden a stick spectrum (peak list) to a continuous curve via Lorentzian."""
    x_grid = np.linspace(x_min, x_max, n_points)
    y = np.zeros(n_points)
    hwhm = linewidth / 2.0

    for pos, amp in peaks:
        if modality == "nmr_1h":
            freq = pos * field_mhz
            freq_axis = x_grid * field_mhz
        else:
            freq, freq_axis = pos, x_grid

        y += amp * (hwhm**2) / ((freq_axis - freq) ** 2 + hwhm**2)

    m = y.max()
    return x_grid, y / m if m > 0 else y


def fetch_public_db(smiles: str, modality: str) -> list[tuple[float, float]]:
    if modality == "nmr_1h":
        return fetch_nmrshiftdb2(smiles)
    elif modality == "ir":
        return fetch_nist_ir(smiles)
    return []


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_overlay(
    query_x: np.ndarray,
    query_y: np.ndarray,
    references: list[dict],
    output_path: pathlib.Path,
    modality: str,
) -> None:
    """Plot query vs top-3 references, stacked vertically."""
    import matplotlib.pyplot as plt

    x_label = "Chemical shift (ppm)" if "nmr" in modality else "Wavenumber (cm⁻¹)"
    top = references[:3]
    n = 1 + len(top)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    axes[0].plot(query_x, query_y, color="black", lw=1.2)
    axes[0].set_ylabel("Query", fontsize=9)
    axes[0].set_yticks([])

    if "nmr" in modality:
        axes[0].invert_xaxis()

    for ax, ref in zip(axes[1:], top):
        ref_path = pathlib.Path(ref["spectrum_path"])
        if ref_path.exists():
            rx, ry = load_spectrum(ref_path)
            ax.plot(rx, ry, color="steelblue", lw=1.0)
        score_str = f"{ref['score']:.3f}"
        ax.set_ylabel(f"{ref['name']}\n({score_str})", fontsize=8)
        ax.set_yticks([])

    axes[-1].set_xlabel(x_label, fontsize=9)
    fig.suptitle("Spectral Match: Query vs References", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Match an experimental spectrum against predicted/database references."
    )
    ap.add_argument(
        "--query", required=True, help="Experimental spectrum file (.xy, .csv, .jdx)"
    )
    ap.add_argument(
        "--smiles", nargs="+", required=True, help="Candidate SMILES strings"
    )
    ap.add_argument(
        "--names", nargs="+", help="Labels for candidates (order matches --smiles)"
    )
    ap.add_argument(
        "--modality",
        choices=SUPPORTED_MODALITIES,
        default="nmr_1h",
        help="Spectrum modality (default: nmr_1h)",
    )
    ap.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="l2",
        help="Similarity metric (default: l2)",
    )
    ap.add_argument(
        "--catalog_dir",
        default="research/spectrum_catalog",
        help="Local catalog directory",
    )
    ap.add_argument("--output_dir", default="spectrum_match", help="Output directory")
    ap.add_argument(
        "--fallback_public_db",
        action="store_true",
        help="Query public DB on catalog miss",
    )
    ap.add_argument(
        "--field_mhz",
        type=float,
        default=400.0,
        help="NMR field strength in MHz (default: 400)",
    )
    ap.add_argument(
        "--linewidth",
        type=float,
        default=1.0,
        help="Lorentzian linewidth Hz for broadening (default: 1.0)",
    )
    ap.add_argument(
        "--plot", action="store_true", help="Save overlay plot match_plot.png"
    )
    ap.add_argument(
        "--lookup_by",
        choices=SUPPORTED_LOOKUP,
        default="canonical_smiles",
        help=(
            "Identifier used for catalog lookup: "
            "'canonical_smiles' (exact stereo, default), "
            "'inchikey' (exact stereo via InChIKey), "
            "'inchikey14' (stereo-insensitive, first 14 chars of InChIKey)"
        ),
    )
    args = ap.parse_args()

    names = (
        args.names
        if args.names and len(args.names) == len(args.smiles)
        else [f"comp{i}" for i in range(len(args.smiles))]
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir = pathlib.Path(args.catalog_dir)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    catalog, inchikey_idx, inchikey14_idx = load_catalog(catalog_dir)
    metric_fn = METRIC_FNS[args.metric]

    # Load and normalize query
    query_path = pathlib.Path(args.query)
    q_x, q_y = load_spectrum(query_path)
    x_min, x_max = float(q_x.min()), float(q_x.max())
    _, q_interp = interpolate_to_grid(q_x, q_y, x_min, x_max)
    q_norm = normalize(q_interp)

    results = []

    for smiles, name in zip(args.smiles, names):
        safe_name = re.sub(r"[^\w\-]", "_", name)[:40]
        print(f"  {name}...", end="  ", flush=True)

        try:
            ids = mol_identifiers(smiles)
        except ValueError as e:
            print(f"INVALID SMILES: {e}")
            results.append(
                {
                    "name": name,
                    "smiles": smiles,
                    "score": None,
                    "source": "error",
                    "error": str(e),
                }
            )
            continue

        # 1. Local catalog
        ref_path = lookup_local(
            ids,
            args.modality,
            catalog,
            inchikey_idx,
            inchikey14_idx,
            output_dir,
            safe_name,
            lookup_by=args.lookup_by,
        )
        source = "local_catalog"

        # 2. Public DB fallback
        if ref_path is None and args.fallback_public_db:
            peaks = fetch_public_db(ids["canonical_smiles"], args.modality)
            if peaks:
                r_x, r_y = peaks_to_xy(
                    peaks,
                    x_min,
                    x_max,
                    linewidth=args.linewidth,
                    field_mhz=args.field_mhz,
                    modality=args.modality,
                )
                ref_path = output_dir / f"{safe_name}_ref.xy"
                arr = np.column_stack([r_x[::-1], r_y[::-1]])
                np.savetxt(ref_path, arr, delimiter="\t", fmt="%.6f")
                source = "public_db"

        if ref_path is None:
            print("missed")
            results.append(
                {"name": name, "smiles": smiles, "score": None, "source": "missed"}
            )
            continue

        # Compute similarity
        r_x, r_y = load_spectrum(ref_path)
        _, r_interp = interpolate_to_grid(r_x, r_y, x_min, x_max)
        r_norm = normalize(r_interp)

        score = metric_fn(q_norm, r_norm)
        print(f"score={score:.3f} ({source})")

        results.append(
            {
                "name": name,
                "smiles": smiles,
                "canonical_smiles": ids["canonical_smiles"],
                "inchikey": ids["inchikey"],
                "inchikey14": ids["inchikey14"],
                "score": round(score, 4),
                "source": source,
                "spectrum_path": str(ref_path),
            }
        )

    # Rank by score (missed entries last)
    results.sort(key=lambda r: r.get("score") or -1.0, reverse=True)

    summary = {
        "query": str(query_path),
        "modality": args.modality,
        "metric": args.metric,
        "lookup_by": args.lookup_by,
        "candidates": results,
    }
    (output_dir / "match_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults -> {output_dir}/match_results.json")

    if args.plot:
        q_x_full, q_y_full = load_spectrum(query_path)
        scored = [r for r in results if r.get("score") is not None]
        plot_overlay(
            q_x_full, q_y_full, scored, output_dir / "match_plot.png", args.modality
        )

    if _save_skill_inputs is not None:
        _save_skill_inputs(args, args.output_dir)


if __name__ == "__main__":
    main()
