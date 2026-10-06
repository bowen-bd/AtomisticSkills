"""Fetch React-OT's source and apply the fixes it needs, for generate_ts.py.

React-OT is not published as a package, and its repository needs two fixes to
import cleanly: a missing ``reactot/trainer/__init__.py`` and a renamed ASE
import (``ase.neb`` became ``ase.mep``). This clones a pinned commit into a
cache directory and patches it there; ``generate_ts.py`` imports React-OT from
that directory. Its dependencies come from the ``reactot`` uv environment.

Usage:
    venv/run reactot python skills/chem-react-ot/scripts/setup_react_ot.py
    venv/run reactot python skills/chem-react-ot/scripts/setup_react_ot.py --dir /path/to/react-ot

Requirements:
    - Environment: reactot (run with: venv/run reactot python ...)
    - git
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO = "https://github.com/deepprinciple/react-ot.git"
COMMIT = "6dfccd0112cf6b504da80ad2516ff8e5ab8bade1"
DEFAULT_DIR = Path.home() / ".cache" / "atomisticskills" / "react-ot"


def react_ot_dir() -> Path:
    """Where React-OT's source lives: $REACT_OT_DIR, else the cache directory."""
    return Path(os.environ.get("REACT_OT_DIR", DEFAULT_DIR)).expanduser()


def patch(src: Path) -> list[str]:
    """Apply the import fixes in place; return what was changed."""
    changed = []
    init = src / "reactot" / "trainer" / "__init__.py"
    if not init.exists():
        init.touch()
        changed.append(str(init.relative_to(src)))
    utils = src / "reactot" / "diffusion" / "_utils.py"
    text = utils.read_text()
    if "from ase.neb import NEB" in text:
        utils.write_text(
            text.replace("from ase.neb import NEB", "from ase.mep import NEB")
        )
        changed.append(str(utils.relative_to(src)))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dir",
        type=Path,
        default=react_ot_dir(),
        help=f"where to put the source (default: $REACT_OT_DIR or {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    src = args.dir.expanduser()
    if not (src / ".git").is_dir():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", REPO, str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "--quiet", COMMIT], check=True)
    changed = patch(src)
    print(
        f"React-OT {COMMIT[:7]} ready at {src}"
        + (f" (patched: {', '.join(changed)})" if changed else "")
    )
    if src != DEFAULT_DIR:
        print(f"Set REACT_OT_DIR={src} so generate_ts.py finds it.")


if __name__ == "__main__":
    main()
