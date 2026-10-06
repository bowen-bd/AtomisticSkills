"""Download the public ICEBERG 2.1 weights (trained on MassSpecGym, `msg_all`).

The ms-pred authors publish them on Dropbox (see the ms-pred README); weights
trained on NIST are available from them on proof of a NIST license. Both stages
land in OUTPUT_DIR:

    OUTPUT_DIR/gen/best.ckpt          # stage 1: fragment DAG generator
    OUTPUT_DIR/inten_contr/best.ckpt  # stage 2: intensity predictor

Usage:
    venv/run msms python skills/chem-msms-predict/scripts/download_weights.py
    venv/run msms python skills/chem-msms-predict/scripts/download_weights.py --output_dir downloads/iceberg_msg_all

Requirements:
    - Environment: msms (run with: venv/run msms python ...; x86_64 only)
    - Network access (~80 MB)
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://www.dropbox.com/scl/fo/kwm35ih8tlfnshfrcq8ot/"
    "AOeS4M0_v9MhqeEZys9sCRQ?rlkey=f1n6pbzx94g1k2el2wcbmee61&dl=1"
)
CHECKPOINTS = ("gen/best.ckpt", "inten_contr/best.ckpt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("downloads/iceberg_msg_all"),
        help="where to put the weights (default: downloads/iceberg_msg_all)",
    )
    args = parser.parse_args()
    out = args.output_dir
    if all((out / c).is_file() for c in CHECKPOINTS):
        print(f"ICEBERG weights already in {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "msg_all.zip"
        print("Downloading ICEBERG 2.1 weights (msg_all, ~80 MB)...")
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    missing = [c for c in CHECKPOINTS if not (out / c).is_file()]
    if missing:
        raise SystemExit(f"Download incomplete, missing: {', '.join(missing)}")
    for c in CHECKPOINTS:
        print(f"  {out / c}")


if __name__ == "__main__":
    main()
