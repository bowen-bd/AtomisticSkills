"""
NMR spectrum I/O and preprocessing utilities: load single spectra or stacks of
time-series spectra, subtract baselines, and restrict the ppm range.

Usage:
    from spectra import load_spectrum, load_time_series, preprocess_spectrum

Requirements:
    - Environment: cpu (run with: venv/run cpu python ...)
    - Required packages: numpy
"""

import pathlib
import numpy as np
from typing import Tuple, List, Optional


def detect_delimiter(path: str) -> str:
    """Heuristic delimiter detection: .tsv -> tab, else sniff first line."""
    p = pathlib.Path(path)
    if p.suffix.lower() == ".tsv":
        return "\t"
    try:
        with open(path, "r", errors="ignore") as f:
            line = f.readline()
        return "\t" if "\t" in line else ","
    except Exception:
        return ","


def load_spectrum(
    path: str, delimiter: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a two-column spectrum file (ppm, intensity).

    Args:
        path: Path to .csv or .xy file (two numeric columns: ppm, intensity).
        delimiter: Column delimiter. If None, auto-detected from file extension/content.

    Returns:
        Tuple of (ppm_array, intensity_array), both 1-D float64, sorted by ppm ascending.
    """
    if delimiter is None:
        delimiter = detect_delimiter(path)
    try:
        arr = np.loadtxt(path, delimiter=delimiter, usecols=[0, 1])
    except ValueError:
        arr = np.loadtxt(path, delimiter=delimiter, usecols=[0, 1], skiprows=1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: expected two numeric columns (ppm, intensity)")
    ppm, intensity = arr[:, 0], arr[:, 1]
    order = np.argsort(ppm)
    return ppm[order], intensity[order]


def interpolate_to_grid(
    ppm: np.ndarray,
    intensity: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a spectrum to a common ppm grid (linear interpolation, zero outside range).

    Args:
        ppm: Source ppm values (sorted ascending).
        intensity: Source intensity values.
        grid: Target ppm grid (sorted ascending).

    Returns:
        Interpolated intensity array aligned to grid.
    """
    return np.interp(grid, ppm, intensity, left=0.0, right=0.0)


def load_time_series(
    paths: List[str],
    n_points: int = 2000,
    ppm_min: Optional[float] = None,
    ppm_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple spectra and stack into a matrix on a common ppm grid.

    Args:
        paths: List of .csv/.xy file paths (time-ordered).
        n_points: Number of interpolation grid points.
        ppm_min: Minimum ppm for the common grid (default: min across all spectra).
        ppm_max: Maximum ppm for the common grid (default: max across all spectra).

    Returns:
        Tuple of:
            grid: 1-D array of shape (n_points,) — common ppm axis.
            matrix: 2-D array of shape (n_spectra, n_points) — interpolated intensities.
    """
    spectra = [load_spectrum(p) for p in paths]
    if ppm_min is None:
        ppm_min = float(min(s[0].min() for s in spectra))
    if ppm_max is None:
        ppm_max = float(max(s[0].max() for s in spectra))
    grid = np.linspace(ppm_min, ppm_max, n_points)
    matrix = np.stack(
        [interpolate_to_grid(ppm, intens, grid) for ppm, intens in spectra]
    )
    return grid, matrix


# ---------------------------------------------------------------------------
# Baseline correction and ppm restriction (shared by deconvolve.py / kinetics.py)
# ---------------------------------------------------------------------------

BASELINE_STATS = ("median", "max")


def min_baseline(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Legacy baseline: shift a spectrum so its minimum intensity becomes 0.

    This removes a constant offset only for noise-free spectra. For a noisy
    spectrum the minimum is a noise excursion several sigma below the true
    baseline, so the shift leaves a positive pedestal under every point and
    biases deconvolved fractions toward an even mixture.

    Args:
        arr: (N, 2) array of (ppm, intensity).

    Returns:
        Tuple of (corrected copy of arr, subtracted value).
    """
    value = float(arr[:, 1].min())
    corrected = arr.copy()
    corrected[:, 1] -= value
    return corrected, value


def window_baseline(
    arr: np.ndarray,
    window: Tuple[float, float],
    stat: str = "median",
) -> Tuple[np.ndarray, float]:
    """
    Subtract a baseline estimated from a signal-free ppm window; clip negatives to 0.

    ``stat="median"`` estimates the offset of a baseline carrying random noise
    (robust to stray points); after subtraction and clipping, only the positive
    half of the noise remains. ``stat="max"`` subtracts the upper envelope of the
    window and suits one-sided, non-random floors (e.g. the residue of a traced
    flat line in a digitized spectrum), which a median leaves half in place. On
    noisy spectra ``max`` removes roughly 3 sigma from every peak, so prefer
    ``median`` there.

    If the window contains no samples (e.g. zero-intensity runs were dropped
    from the file), the spectrum is linearly interpolated across the window,
    so the estimate comes from the samples bounding the gap.

    Args:
        arr: (N, 2) array of (ppm, intensity).
        window: (lo, hi) ppm limits of a region without signals.
        stat: "median" or "max".

    Returns:
        Tuple of (corrected copy of arr, subtracted baseline value).
    """
    if stat not in BASELINE_STATS:
        raise ValueError(f"baseline stat must be one of {BASELINE_STATS}, got {stat!r}")
    lo, hi = sorted(window)
    order = np.argsort(arr[:, 0])
    ppm, intensity = arr[order, 0], arr[order, 1]
    inside = (ppm >= lo) & (ppm <= hi)
    values = intensity[inside]
    if values.size == 0:
        values = np.interp(np.linspace(lo, hi, 50), ppm, intensity)
    value = float(np.median(values) if stat == "median" else np.max(values))
    corrected = arr.copy()
    corrected[:, 1] = np.clip(corrected[:, 1] - value, 0.0, None)
    return corrected, value


def restrict_ppm(arr: np.ndarray, ppm_range: Tuple[float, float]) -> np.ndarray:
    """
    Keep only the points of a spectrum inside a ppm range (inclusive).

    Args:
        arr: (N, 2) array of (ppm, intensity).
        ppm_range: (lo, hi) ppm limits.

    Returns:
        (M, 2) array with the points inside the range.
    """
    lo, hi = sorted(ppm_range)
    kept = arr[(arr[:, 0] >= lo) & (arr[:, 0] <= hi)]
    if kept.shape[0] < 2:
        raise ValueError(f"fewer than two points inside ppm range {lo}-{hi}")
    return kept


def preprocess_spectrum(
    arr: np.ndarray,
    baseline_correct: bool = False,
    baseline_window: Optional[Tuple[float, float]] = None,
    baseline_stat: str = "median",
    ppm_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply the baseline treatment, then the ppm restriction, to one spectrum.

    The baseline is estimated on the full spectrum, so the window may lie
    outside ``ppm_range``. ``baseline_correct`` and ``baseline_window`` are
    mutually exclusive.

    Args:
        arr: (N, 2) array of (ppm, intensity).
        baseline_correct: Legacy minimum subtraction.
        baseline_window: (lo, hi) signal-free window for ``window_baseline``.
        baseline_stat: Statistic for ``window_baseline`` ("median" or "max").
        ppm_range: Optional (lo, hi) range to keep after baseline correction.

    Returns:
        Tuple of (processed array, subtracted baseline value; 0.0 if none).
    """
    if baseline_correct and baseline_window is not None:
        raise ValueError("use either baseline_correct or baseline_window, not both")
    value = 0.0
    if baseline_window is not None:
        arr, value = window_baseline(arr, baseline_window, baseline_stat)
    elif baseline_correct:
        arr, value = min_baseline(arr)
    if ppm_range is not None:
        arr = restrict_ppm(arr, ppm_range)
    return arr, value


def range_area_fraction(arr: np.ndarray, ppm_range: Tuple[float, float]) -> float:
    """
    Fraction of a spectrum's (non-negative) area that lies inside a ppm range.

    For a reference spectrum covering all of a molecule's protons, this is the
    fraction of those protons whose signals fall inside the range.

    Args:
        arr: (N, 2) array of (ppm, intensity), already baseline-corrected.
        ppm_range: (lo, hi) ppm limits.

    Returns:
        Area inside the range divided by the total area (0.0 if the total is 0).
    """
    order = np.argsort(arr[:, 0])
    ppm = arr[order, 0]
    intensity = np.clip(arr[order, 1], 0.0, None)
    total = float(np.trapezoid(intensity, ppm))
    if total <= 0:
        return 0.0
    lo, hi = sorted(ppm_range)
    inside = (ppm >= lo) & (ppm <= hi)
    return float(np.trapezoid(intensity[inside], ppm[inside])) / total


def effective_protons(
    full_refs: List[np.ndarray],
    protons: List[float],
    ppm_range: Optional[Tuple[float, float]] = None,
    range_protons: Optional[List[float]] = None,
) -> List[float]:
    """
    Proton counts to use for the proton correction of a (possibly cropped) fit.

    Deconvolution normalizes each reference over the fitted range, so its signal
    fraction counts only the protons whose signals lie in that range. Without a
    range, this is the per-molecule count. With a range, ``range_protons`` (the
    protons of each reference inside the range) is used when given; otherwise
    each per-molecule count is scaled by the fraction of the reference's area
    inside the range, computed from the full baseline-corrected reference.

    Args:
        full_refs: Baseline-corrected reference spectra over their full range.
        protons: Protons per molecule for each reference.
        ppm_range: Fitted ppm range, or None for the full spectra.
        range_protons: Optional protons of each reference inside ``ppm_range``.

    Returns:
        One proton count per reference.
    """
    if range_protons is not None:
        if ppm_range is None:
            raise ValueError("range protons require a ppm range")
        if len(range_protons) != len(full_refs):
            raise ValueError("give one range proton count per reference")
        return [float(n) for n in range_protons]
    if ppm_range is None:
        return [float(n) for n in protons]
    return [
        float(n) * range_area_fraction(ref, ppm_range)
        for n, ref in zip(protons, full_refs)
    ]
