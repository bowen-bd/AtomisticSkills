"""Fourier interpolation with orbital-pair Wigner-Seitz translations."""

from __future__ import annotations

import numpy as np


def expand_hamiltonian(
    vectors: np.ndarray,
    matrices: np.ndarray,
    degeneracies: np.ndarray,
    shifts: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine H(R)/(N_R*N_T) at each R+T, ready for a plain Fourier sum."""
    if shifts is None:
        return vectors, matrices / degeneracies[:, None, None]
    nw = matrices.shape[1]
    expanded: dict[tuple, np.ndarray] = {}
    for ir, r in enumerate(vectors):
        for m in range(nw):
            for n in range(nw):
                translations = shifts[(*r, m, n)]
                value = matrices[ir, m, n] / degeneracies[ir] / len(translations)
                for translation in translations:
                    key = tuple(r + translation)
                    if key not in expanded:
                        expanded[key] = np.zeros((nw, nw), dtype=complex)
                    expanded[key][m, n] += value
    keys = sorted(expanded)
    return np.asarray(keys), np.asarray([expanded[key] for key in keys])


def interpolate(
    vectors: np.ndarray,
    matrices: np.ndarray,
    degeneracies: np.ndarray,
    kpoints: np.ndarray,
    shifts: dict | None = None,
    hermiticity_tol: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Diagonalize H(k), rejecting non-Hermitian input before roundoff cleanup."""
    r, h = expand_hamiltonian(vectors, matrices, degeneracies, shifts)
    eigenvalues = []
    max_residual = 0.0
    for k in kpoints:
        hk = np.tensordot(np.exp(2j * np.pi * (r @ k)), h, axes=(0, 0))
        residual = float(np.max(np.abs(hk - hk.conj().T)))
        max_residual = max(max_residual, residual)
        if not np.isfinite(hk).all() or residual > hermiticity_tol:
            raise ValueError(
                f"H(k) is not Hermitian: residual={residual:.6g} eV at k={k.tolist()}"
            )
        eigenvalues.append(np.linalg.eigvalsh((hk + hk.conj().T) / 2))
    return np.asarray(eigenvalues), max_residual
