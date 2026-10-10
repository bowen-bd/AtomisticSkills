"""
Fetch a small bulk-modulus dataset (K_VRH, GPa) from the Materials Project.

Writes a JSON list of {"material_id", "structure", "bulk_modulus"} records for
non-deprecated elasticity documents with at most --max_sites sites and
0 < K_VRH < --max_k GPa, sorted by MP ID and truncated to --limit entries, so
the selection is reproducible for a given MP database version.

Usage:
    python fetch_bulk_modulus.py --output .agents/test/mp_bulk_modulus.json

Requirements:
    - Environment: cpu (MP_API_KEY in the environment)
    - Required packages: mp_api, pymatgen
"""

import argparse
import json
from pathlib import Path

from mp_api.client import MPRester

from src.utils.config_utils import save_skill_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch MP bulk moduli (K_VRH) and structures into a JSON dataset."
    )
    parser.add_argument("--output", default=".agents/test/mp_bulk_modulus.json")
    parser.add_argument("--limit", type=int, default=500, help="Entries to keep")
    parser.add_argument("--max_sites", type=int, default=12, help="Max sites per cell")
    parser.add_argument("--max_k", type=float, default=600.0, help="Max K_VRH (GPa)")
    args = parser.parse_args()

    with MPRester() as mpr:
        docs = mpr.materials.elasticity.search(
            k_vrh=(1e-3, args.max_k),
            fields=["material_id", "structure", "bulk_modulus", "deprecated", "nsites"],
        )
    records = []
    for d in docs:
        k = d.bulk_modulus.vrh if d.bulk_modulus is not None else None
        if (
            d.deprecated
            or d.nsites > args.max_sites
            or k is None
            or not 0.0 < k < args.max_k
        ):
            continue
        records.append(
            {
                "material_id": str(d.material_id),
                "structure": d.structure.as_dict(),
                "bulk_modulus": float(k),
            }
        )
    records.sort(key=lambda r: int(r["material_id"].split("-")[1]))
    records = records[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records))
    save_skill_inputs(args, str(out.parent))
    print(f"Saved {len(records)} entries to {out}")


if __name__ == "__main__":
    main()
