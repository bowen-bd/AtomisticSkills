"""Read standard folded Wannier90 output and preserve per-stage provenance.

Requirements: cpu environment, NumPy and PyYAML. All lengths are in Angstrom
unless explicitly identified as lattice indices; energies are in eV.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import yaml

BOHR_ANGSTROM = 0.529177210903


def fields(stream: TextIO) -> list[str]:
    """Read the next nonempty record, rejecting unexpected end of file."""
    for line in stream:
        if line.strip():
            return line.split()
    raise ValueError(f"Unexpected end of file: {stream.name}")


def read_hr(
    filepath: str | Path,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate an hr.dat stream; return N, NR, degeneracies, R and H."""
    with Path(filepath).open() as stream:
        stream.readline()  # Free-form creation comment.
        nw, nr = int(fields(stream)[0]), int(fields(stream)[0])
        if nw <= 0 or nr <= 0:
            raise ValueError("num_wann and nrpts must be positive")
        deg: list[int] = []
        while len(deg) < nr:
            deg.extend(map(int, fields(stream)))
        if len(deg) != nr or min(deg) <= 0:
            raise ValueError("Invalid degeneracy count or nonpositive degeneracy")
        vectors = np.empty((nr, 3), dtype=int)
        matrices = np.empty((nr, nw, nw), dtype=complex)
        seen_r: set[tuple[int, ...]] = set()
        for ir in range(nr):
            seen_pairs: set[tuple[int, int]] = set()
            for j in range(nw * nw):
                row = fields(stream)
                if len(row) != 7:
                    raise ValueError("Each hr.dat matrix record must have seven fields")
                r = tuple(map(int, row[:3]))
                m, n = int(row[3]) - 1, int(row[4]) - 1
                if j == 0:
                    if r in seen_r:
                        raise ValueError(f"Duplicate lattice vector {r}")
                    seen_r.add(r)
                    vectors[ir] = r
                if tuple(vectors[ir]) != r or not (0 <= m < nw and 0 <= n < nw):
                    raise ValueError(
                        "Inconsistent R block or out-of-range orbital index"
                    )
                if (m, n) in seen_pairs:
                    raise ValueError(
                        f"Duplicate matrix element at {r}, {m + 1}, {n + 1}"
                    )
                seen_pairs.add((m, n))
                matrices[ir, m, n] = complex(float(row[5]), float(row[6]))
            if len(seen_pairs) != nw * nw:
                raise ValueError("Incomplete Hamiltonian block")
        if any(line.strip() for line in stream):
            raise ValueError("Unexpected records after the Hamiltonian")
    if not np.isfinite(matrices).all() or (0, 0, 0) not in seen_r:
        raise ValueError("Hamiltonian must be finite and contain R=(0,0,0)")
    return nw, nr, np.asarray(deg, dtype=float), vectors, matrices


def read_wsvec(filepath: str | Path, vectors: np.ndarray, nw: int) -> tuple[bool, dict]:
    """Read all orbital-pair superlattice shifts for a folded hr.dat file.

    Reject incomplete mappings and expanded/weight-applied hr.dat files whose
    R list does not match this sidecar. Never apply weights twice.
    """
    expected = {tuple(r) for r in vectors}
    shifts: dict[tuple, np.ndarray] = {}
    with Path(filepath).open() as stream:
        header = stream.readline().lower()
        mode = re.search(r"use_ws_distance\s*=\s*\.?\s*(true|false|t|f)\b", header)
        if mode is None:
            raise ValueError("wsvec.dat header must declare use_ws_distance")
        enabled = mode.group(1) in {"true", "t"}
        for line in stream:
            if not line.strip():
                continue
            values = list(map(int, line.split()))
            if len(values) != 5:
                raise ValueError("Invalid wsvec.dat orbital-pair record")
            r, m, n = tuple(values[:3]), values[3] - 1, values[4] - 1
            key = (*r, m, n)
            if r not in expected or not (0 <= m < nw and 0 <= n < nw) or key in shifts:
                raise ValueError("wsvec.dat is duplicated or incompatible with hr.dat")
            count = int(fields(stream)[0])
            if count <= 0:
                raise ValueError("Wigner-Seitz multiplicities must be positive")
            translations = [list(map(int, fields(stream))) for _ in range(count)]
            if any(len(t) != 3 for t in translations):
                raise ValueError("Each Wigner-Seitz shift must have three components")
            array = np.asarray(translations, dtype=int)
            if len(np.unique(array, axis=0)) != count:
                raise ValueError("Duplicate Wigner-Seitz shift")
            if not enabled and (count != 1 or np.any(array)):
                raise ValueError("use_ws_distance=false must have one zero shift")
            shifts[key] = array
    if len(shifts) != len(vectors) * nw * nw:
        raise ValueError("wsvec.dat is missing orbital-pair mappings")
    return enabled, shifts


def resolve_wsvec(
    hr_file: str | Path,
    vectors: np.ndarray,
    nw: int,
    wsvec: str | Path | None = None,
    legacy: bool = False,
) -> tuple[dict | None, dict]:
    """Require a sidecar unless the caller explicitly selects legacy output."""
    hr_file = Path(hr_file)
    sidecar = (
        Path(wsvec)
        if wsvec
        else hr_file.with_name(hr_file.name.removesuffix("_hr.dat") + "_wsvec.dat")
    )
    if legacy and wsvec:
        raise ValueError("--legacy and --wsvec cannot be combined")
    if sidecar.exists():
        enabled, shifts = read_wsvec(sidecar, vectors, nw)
        if legacy and enabled:
            raise ValueError(
                "--legacy conflicts with use_ws_distance=true in the sidecar"
            )
        return shifts, {
            "use_ws_distance": enabled,
            "wsvec_file": str(sidecar.resolve()),
        }
    if wsvec or not legacy:
        raise FileNotFoundError(
            f"Required Wigner-Seitz sidecar missing: {sidecar}; use --legacy only for known use_ws_distance=false output"
        )
    return None, {"use_ws_distance": False, "wsvec_file": None}


def read_kpoints(filepath: str | Path) -> np.ndarray:
    """Read a counted Wannier90 .kpt file or a plain three/four-column list."""
    rows = [
        line.split()
        for line in Path(filepath).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    if not rows:
        raise ValueError("Empty k-point file")
    count = int(rows.pop(0)[0]) if len(rows[0]) == 1 else len(rows)
    if len(rows) != count or count < 1 or any(len(row) not in (3, 4) for row in rows):
        raise ValueError("K-point count or column count is inconsistent")
    data = np.asarray([[float(x) for x in row[:3]] for row in rows])
    if not np.isfinite(data).all():
        raise ValueError("K-points must be finite")
    return data


def read_bands(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read band-separated distance/energy blocks; validate a shared path."""
    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in Path(filepath).read_text().splitlines() + [""]:
        if line.lstrip().startswith(("#", "!")):
            continue
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        row = line.split()
        if len(row) != 2:
            raise ValueError(
                "Band files must contain distance/energy pairs separated into bands by blank lines"
            )
        current.append([float(x) for x in row])
    if not blocks or len({len(block) for block in blocks}) != 1:
        raise ValueError("Empty or inconsistent band lengths")
    array = np.asarray(blocks)
    if not np.isfinite(array).all() or not np.allclose(
        array[:, :, 0], array[0, :, 0], atol=1e-8, rtol=0
    ):
        raise ValueError("Bands must be finite and share the same path coordinates")
    if np.any(np.diff(array[0, :, 0]) < -1e-8):
        raise ValueError("Band path distances must be nondecreasing")
    return array[0, :, 0], array[:, :, 1].T


def read_cell(filepath: str | Path) -> np.ndarray:
    """Read unit_cell_cart from a .win file, converting bohr to Angstrom."""
    text = Path(filepath).read_text()
    match = re.search(
        r"begin\s+unit_cell_cart\s*\n(.*?)end\s+unit_cell_cart", text, re.I | re.S
    )
    if match is None:
        raise ValueError("The .win file has no unit_cell_cart block")
    rows = [re.split(r"[!#]", line)[0].strip() for line in match.group(1).splitlines()]
    rows = [row for row in rows if row]
    factor = 1.0
    if rows[0].lower() in {"ang", "angstrom", "bohr"}:
        factor = BOHR_ANGSTROM if rows.pop(0).lower() == "bohr" else 1.0
    cell = np.asarray([list(map(float, row.split())) for row in rows]) * factor
    if (
        cell.shape != (3, 3)
        or not np.isfinite(cell).all()
        or abs(np.linalg.det(cell)) < 1e-12
    ):
        raise ValueError("Invalid or singular unit cell")
    return cell


def write_bands(
    filepath: str | Path, distances: np.ndarray, energies: np.ndarray
) -> None:
    """Write the standard two-column, band-separated format."""
    with Path(filepath).open("w") as stream:
        for band in energies.T:
            np.savetxt(stream, np.column_stack((distances, band)), fmt="%.10f")
            stream.write("\n")


def save_config(output_dir: str | Path, stage: str, parameters: dict[str, Any]) -> Path:
    """Save all CLI values under a stage key without erasing other stages."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "input_configs.yaml"
    document = yaml.safe_load(path.read_text()) if path.exists() else {}
    if not isinstance(document, dict):
        raise ValueError("Existing input_configs.yaml must be a mapping")
    serializable = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in parameters.items()
    }
    document.setdefault("stages", {})[stage] = serializable
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return directory


def write_json(path: str | Path, data: dict) -> None:
    """Write JSON with finite values only."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@contextmanager
def failure_report(path: Path, nested_validation: bool = False) -> Iterator[None]:
    """Replace stale success reports with an explicit failure on invalid input."""
    try:
        yield
    except (OSError, ValueError) as error:
        result = {"status": "FAILED", "error": str(error)}
        write_json(path, {"validation": result} if nested_validation else result)
        raise SystemExit(f"{path.name}: {error}") from error


def sha256(path: str | Path) -> str:
    """Hash an input artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
