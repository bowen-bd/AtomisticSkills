"""Tests for the chem-nmr-analysis baseline and ppm-range helpers.

Synthetic two-component 1H spectra with a known composition check that the
window baseline recovers the true fractions where the legacy minimum
subtraction (``--baseline-correct``) does not.
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../skills/chem-nmr-analysis/scripts")
)
sys.path.insert(0, SCRIPTS)

from deconvolve import deconvolve_spectra
from spectra import (
    effective_protons,
    min_baseline,
    preprocess_spectrum,
    restrict_ppm,
    window_baseline,
)

GRID = np.linspace(0.5, 4.5, 1600)
SIGNAL_FREE = (2.2, 3.2)
TRUE_FRACTION_A = 0.3


def _peaks(
    centres_heights: list[tuple[float, float]], width: float = 0.01
) -> np.ndarray:
    """Sum of Gaussian lines on GRID."""
    return sum(h * np.exp(-0.5 * ((GRID - c) / width) ** 2) for c, h in centres_heights)


def _unit_area(intensity: np.ndarray) -> np.ndarray:
    """Normalize a spectrum to unit area on GRID."""
    return intensity / np.trapezoid(intensity, GRID)


# Two components with the same proton count: A (4.00 ppm diagnostic), B (3.60 ppm)
COMP_A = _unit_area(_peaks([(4.0, 1.0), (1.0, 3.0), (1.6, 2.0)]))
COMP_B = _unit_area(_peaks([(3.6, 1.0), (1.2, 3.0), (1.8, 2.0)]))


def _spectrum(intensity: np.ndarray) -> np.ndarray:
    return np.column_stack([GRID, intensity])


def _noisy_mixture(
    offset: float = -0.5, sigma: float = 0.1, seed: int = 0
) -> np.ndarray:
    """Mixture with a known composition, a constant offset and white noise."""
    rng = np.random.default_rng(seed)
    clean = TRUE_FRACTION_A * COMP_A + (1.0 - TRUE_FRACTION_A) * COMP_B
    return _spectrum(clean + offset + rng.normal(0.0, sigma, GRID.size))


def _fraction_a(mixture: np.ndarray, refs: list[np.ndarray]) -> float:
    return deconvolve_spectra(mixture, refs, [1, 1])["proportions"][0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_median_recovers_fraction_on_noisy_offset_mixture(seed: int) -> None:
    """Median of a signal-free window removes the offset; the fraction is recovered."""
    refs = [_spectrum(COMP_A), _spectrum(COMP_B)]
    mixture, value = window_baseline(_noisy_mixture(seed=seed), SIGNAL_FREE, "median")

    assert value == pytest.approx(-0.5, abs=0.02)
    assert _fraction_a(mixture, refs) == pytest.approx(TRUE_FRACTION_A, abs=0.025)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_min_subtraction_is_biased_on_noisy_offset_mixture(seed: int) -> None:
    """The minimum sits several sigma below the baseline and biases the fraction."""
    refs = [_spectrum(COMP_A), _spectrum(COMP_B)]
    mixture, value = min_baseline(_noisy_mixture(seed=seed))

    assert value < -0.5 - 2 * 0.1
    assert _fraction_a(mixture, refs) - TRUE_FRACTION_A > 0.05


def test_window_max_removes_one_sided_reference_floor() -> None:
    """A positive digitization-like floor on the references needs the max statistic."""
    floor = 0.2 + 0.2 * ((GRID * 30.0) % 1.0)  # sawtooth between 0.2 and 0.4
    refs_raw = [_spectrum(COMP_A + floor), _spectrum(COMP_B + floor)]
    mixture = _spectrum(TRUE_FRACTION_A * COMP_A + (1.0 - TRUE_FRACTION_A) * COMP_B)

    refs_max = [window_baseline(r, SIGNAL_FREE, "max")[0] for r in refs_raw]
    refs_median = [window_baseline(r, SIGNAL_FREE, "median")[0] for r in refs_raw]

    err_max = abs(_fraction_a(mixture, refs_max) - TRUE_FRACTION_A)
    err_median = abs(_fraction_a(mixture, refs_median) - TRUE_FRACTION_A)
    assert err_max < 0.005
    assert err_median > 0.02


def test_window_without_samples_uses_bounding_points() -> None:
    """A window inside a gap of dropped zero-intensity points gives a zero baseline."""
    keep = (GRID < 2.0) | (GRID > 3.4)
    gapped = _spectrum(COMP_A)[keep]
    corrected, value = window_baseline(gapped, SIGNAL_FREE, "median")

    assert value == pytest.approx(0.0, abs=1e-6)
    assert np.array_equal(corrected[:, 0], gapped[:, 0])


def test_negatives_are_clipped_and_ppm_range_applied_after_baseline() -> None:
    """Window baseline clips negatives; the ppm range is cut after the baseline step."""
    processed, value = preprocess_spectrum(
        _noisy_mixture(),
        baseline_window=SIGNAL_FREE,
        ppm_range=(3.5, 4.1),
    )

    assert value == pytest.approx(-0.5, abs=0.02)
    assert processed[:, 1].min() >= 0.0
    assert processed[:, 0].min() >= 3.5 and processed[:, 0].max() <= 4.1


def test_restrict_ppm_rejects_empty_range() -> None:
    with pytest.raises(ValueError):
        restrict_ppm(_spectrum(COMP_A), (10.0, 11.0))


def test_baseline_options_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        preprocess_spectrum(
            _spectrum(COMP_A), baseline_correct=True, baseline_window=SIGNAL_FREE
        )
    with pytest.raises(ValueError):
        window_baseline(_spectrum(COMP_A), SIGNAL_FREE, "mean")


# ---------------------------------------------------------------------------
# Proton correction of a fit restricted to a ppm range
# ---------------------------------------------------------------------------

# Two six-proton molecules in a 50:50 mixture with unequal protons in 0-3 ppm:
# A has 3 H at 1 ppm and 3 H at 7 ppm; B has 1 H at 2 ppm and 5 H at 8 ppm.
WIDE = np.linspace(0.0, 9.0, 3600)
WINDOW = (0.0, 3.0)


def _lines(centres_protons: list[tuple[float, float]]) -> np.ndarray:
    """Spectrum on WIDE whose line areas are proportional to the proton counts."""
    width = 0.01
    return np.column_stack(
        [
            WIDE,
            sum(
                n
                * np.exp(-0.5 * ((WIDE - c) / width) ** 2)
                / (width * np.sqrt(2 * np.pi))
                for c, n in centres_protons
            ),
        ]
    )


WIN_A = _lines([(1.0, 3.0), (7.0, 3.0)])
WIN_B = _lines([(2.0, 1.0), (8.0, 5.0)])
WIN_MIX = np.column_stack([WIDE, WIN_A[:, 1] + WIN_B[:, 1]])  # equal moles


def test_effective_protons_scale_by_reference_area_in_range() -> None:
    """Without explicit counts, each count is scaled by the area share in range."""
    counts = effective_protons([WIN_A, WIN_B], [6, 6], WINDOW)
    assert counts == pytest.approx([3.0, 1.0], rel=1e-3)
    assert effective_protons([WIN_A, WIN_B], [6, 6]) == [6.0, 6.0]
    assert effective_protons([WIN_A, WIN_B], [6, 6], WINDOW, [3, 1]) == [3.0, 1.0]


def test_cropped_fit_with_per_molecule_counts_is_biased() -> None:
    """Dividing cropped signal fractions by whole-molecule counts gives 75:25."""
    crop = [restrict_ppm(a, WINDOW) for a in (WIN_MIX, WIN_A, WIN_B)]
    res = deconvolve_spectra(crop[0], crop[1:], [6, 6])
    assert res["proportions"][0] == pytest.approx(0.75, abs=0.01)


def test_cropped_fit_with_range_protons_recovers_composition() -> None:
    """Range proton counts recover 50:50 from the 0-3 ppm window."""
    crop = [restrict_ppm(a, WINDOW) for a in (WIN_MIX, WIN_A, WIN_B)]
    counts = effective_protons([WIN_A, WIN_B], [6, 6], WINDOW)
    res = deconvolve_spectra(crop[0], crop[1:], counts)
    assert res["proportions"][0] == pytest.approx(0.5, abs=0.01)


def _write(path: str, arr: np.ndarray) -> str:
    np.savetxt(path, arr, delimiter=",")
    return path


@pytest.mark.parametrize("extra", [[], ["--range-protons", "3", "1"]])
def test_deconvolve_cli_ppm_range_recovers_composition(tmp_path, extra) -> None:
    """deconvolve.py --ppm-range returns 50:50 with automatic or explicit counts."""
    files = [
        _write(str(tmp_path / f"{n}.csv"), a)
        for n, a in (("mix", WIN_MIX), ("a", WIN_A), ("b", WIN_B))
    ]
    out = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPTS, "deconvolve.py"),
            *files,
            "--protons",
            "6",
            "6",
            "--names",
            "A",
            "B",
            "--ppm-range",
            "0",
            "3",
            "--json",
            "--quiet",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout
    result = json.loads(out.split("JSON:", 1)[1])
    assert result["proportions"]["A"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("extra", [[], ["--range_protons", "3", "1"]])
def test_kinetics_cli_ppm_range_recovers_composition(tmp_path, extra) -> None:
    """kinetics.py --ppm_range returns 50:50 at every time point."""
    refs = [
        _write(str(tmp_path / f"{n}.csv"), a) for n, a in (("a", WIN_A), ("b", WIN_B))
    ]
    tps = [_write(str(tmp_path / f"t{i}.csv"), WIN_MIX) for i in range(2)]
    subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPTS, "kinetics.py"),
            "--refs",
            *refs,
            "--timepoints",
            *tps,
            "--times",
            "0",
            "10",
            "--protons",
            "6",
            "6",
            "--names",
            "A",
            "B",
            "--ppm_range",
            "0",
            "3",
            "--quiet",
            "--output_dir",
            str(tmp_path / "out"),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    with open(tmp_path / "out" / "kinetics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all(float(r["A"]) == pytest.approx(0.5, abs=0.01) for r in rows)
