"""Tests for the mat-surface-adsorption skill.

Verifies site identification and coordinate extraction from matcalc AdsorptionCalc
results, preventing regressions where site names defaulted to 'unknown' (issue #49).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPT = ROOT / "skills/mat-surface-adsorption/scripts/calculate_adsorption.py"


def _load_skill_module():
    spec = importlib.util.spec_from_file_location("calculate_adsorption", SKILL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adsorption_mod = _load_skill_module()


@pytest.mark.base
def test_process_adsorption_results_extracts_matcalc_keys():
    """Regression test for issue #49: AdsorptionCalc returns 'adsorption_site', not 'site'."""
    raw_results = [
        {
            "adsorption_site": "ontop",
            "adsorption_site_coord": [1.27, 0.65, 14.37],
            "adsorption_site_index": 0,
            "adsorption_energy": -0.506,
            "adslab_energy": -32.52,
            "slab_energy": -17.89,
            "adsorbate_energy": -14.12,
        },
        {
            "adsorption_site": "bridge",
            "adsorption_site_coord": [1.91, -0.25, 13.73],
            "adsorption_site_index": 0,
            "adsorption_energy": -0.282,
            "adslab_energy": -32.30,
            "slab_energy": -17.89,
            "adsorbate_energy": -14.12,
        },
        {
            "adsorption_site": "hollow",
            "adsorption_site_coord": [1.27, -0.55, 13.52],
            "adsorption_site_index": 0,
            "adsorption_energy": -0.246,
            "adslab_energy": -32.27,
            "slab_energy": -17.89,
            "adsorbate_energy": -14.12,
        },
    ]

    sites, most_stable = adsorption_mod.process_adsorption_results(raw_results)

    assert len(sites) == 3
    assert sites[0]["site"] == "ontop"
    assert sites[1]["site"] == "bridge"
    assert sites[2]["site"] == "hollow"

    for site in sites:
        assert site["site"] != "unknown"
        assert "site_coord" in site

    assert sites[0]["site_coord"] == [1.27, 0.65, 14.37]

    assert most_stable is not None
    assert most_stable["site"] == "ontop"
    assert most_stable["adsorption_energy"] == -0.506


@pytest.mark.base
def test_process_adsorption_results_handles_fallback_and_missing():
    """Verify backwards compatibility with legacy 'site' key and fallback to 'unknown'."""
    raw_results = [
        {
            "site": "legacy_ontop",
            "adsorption_energy": -0.45,
        },
        {
            "adsorption_energy": -0.15,
        },
        {
            "adsorption_site": "bridge",
            "adsorption_energy": None,  # Should be skipped
        },
    ]

    sites, most_stable = adsorption_mod.process_adsorption_results(raw_results)

    assert len(sites) == 2
    assert sites[0]["site"] == "legacy_ontop"
    assert sites[1]["site"] == "unknown"
    assert most_stable["site"] == "legacy_ontop"


@pytest.mark.base
def test_process_adsorption_results_empty():
    """Verify graceful handling of empty results list."""
    sites, most_stable = adsorption_mod.process_adsorption_results([])
    assert sites == []
    assert most_stable is None


@pytest.mark.base
def test_matcalc_adslabs_keys_contract():
    """Verify contract with matcalc's AdsorptionCalc data structure."""
    from pymatgen.core import Structure, Lattice, Molecule
    from matcalc import AdsorptionCalc
    from ase.calculators.calculator import Calculator, all_changes

    class MockCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties or ["energy"], system_changes)
            self.results = {
                "energy": -10.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.zeros(6),
            }

    cu = Structure(
        Lattice.cubic(3.6),
        ["Cu", "Cu", "Cu", "Cu"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    co = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.15]])

    calc = AdsorptionCalc(MockCalculator(), max_steps=1)
    # dry_run returns raw adslab dictionaries before PES relaxation
    adslabs = calc.calc_adslabs(
        adsorbate=co,
        bulk=cu,
        miller_index=(1, 1, 1),
        dry_run=True,
    )

    assert len(adslabs) > 0
    first = adslabs[0]
    # Ensure matcalc uses 'adsorption_site' and does NOT use 'site'
    assert "adsorption_site" in first
    assert "site" not in first

    # Simulate an adsorption energy to run through our processor
    for slab in adslabs:
        slab["adsorption_energy"] = -1.0

    sites, most_stable = adsorption_mod.process_adsorption_results(adslabs)
    for s in sites:
        assert s["site"] in {"ontop", "bridge", "hollow"}
        assert s["site"] != "unknown"
