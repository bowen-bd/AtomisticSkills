"""Fetch MatterGen's source and apply the fix it needs, for the mattergen server.

MatterGen runs from a source checkout: its PyPI distribution omits the data
files it needs (sampling configs, GemNet scale factors, training configs). Its
GemNet basis code also calls ``np.math.factorial``, an alias NumPy 2 removed,
while the verified environment runs NumPy 2; the standard library's
``math.factorial`` is the same function. This clones a pinned commit (without
the Git LFS checkpoints: the weights come from Hugging Face) and patches it.

Usage:
    venv/run mattergen python skills/ml-generative-mattergen/scripts/setup_mattergen.py
    venv/run mattergen python skills/ml-generative-mattergen/scripts/setup_mattergen.py --dir /path/to/mattergen

Requirements:
    - Environment: mattergen (run with: venv/run mattergen python ...), or any Python 3
    - git
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO = "https://github.com/microsoft/mattergen"
# v1.0.3 plus seven fixes (package data, batched guidance, the progress log)
COMMIT = "94441ee60efc01e8a5651fdfddf4a147f01db057"
# The default MATTERGEN_REPO of the wrapper: a `mattergen` checkout next to this project
DEFAULT_DIR = Path(__file__).resolve().parents[3].parent / "mattergen"
BASIS_UTILS = Path("mattergen/common/gemnet/layers/basis_utils.py")


def mattergen_dir() -> Path:
    """Where MatterGen's source lives: $MATTERGEN_REPO, else next to this project."""
    return Path(os.environ.get("MATTERGEN_REPO", DEFAULT_DIR)).expanduser()


def patch(src: Path) -> list[str]:
    """Replace np.math (removed in NumPy 2) with math; return the files changed."""
    path = src / BASIS_UTILS
    text = path.read_text()
    if "np.math." not in text:
        return []
    text = text.replace("np.math.factorial", "math.factorial")
    if "\nimport math\n" not in text:
        text = text.replace(
            "\nimport numpy as np\n", "\nimport math\nimport numpy as np\n", 1
        )
    path.write_text(text)
    return [str(BASIS_UTILS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dir",
        type=Path,
        default=mattergen_dir(),
        help=f"where to put the source (default: $MATTERGEN_REPO or {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    src = args.dir.expanduser()
    # The repository keeps its checkpoints in Git LFS; skip them.
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not (src / ".git").is_dir():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", REPO, str(src)], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(src), "checkout", "--quiet", COMMIT], check=True, env=env
    )
    changed = patch(src)
    print(
        f"MatterGen {COMMIT[:7]} ready at {src}"
        + (f" (patched: {', '.join(changed)})" if changed else "")
    )
    if src != DEFAULT_DIR:
        print(f"Set MATTERGEN_REPO={src} so the mattergen server finds it.")


if __name__ == "__main__":
    main()
