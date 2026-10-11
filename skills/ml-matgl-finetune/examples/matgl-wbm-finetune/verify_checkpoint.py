"""
Verify that a train_matgl.py checkpoint round-trips through MatGLWrapper.

Loads ``fine_tuned_model.pth`` with ``MatGLWrapper.load_checkpoint()`` in a fresh
process, predicts the structure that ``train_matgl.py`` recorded in
``finetune_record.json`` (``result.reference_prediction``), and compares it with
the energy the fine-tuned potential predicted at the end of training. The
pretrained (not fine-tuned) energy is reported too, to show that the weights did
change.

Usage:
    python verify_checkpoint.py --run-dir .agents/test/wbm/ft \
        --val-data .agents/test/wbm/val_data.json

Requirements:
    - Environment: mlip
"""

import argparse
import json
from pathlib import Path

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper


def to_atoms(item: dict) -> Atoms:
    """
    Build ASE Atoms from a prepared-data record.

    Args:
        item: Record with a pymatgen or ASE-dict ``structure``.

    Returns:
        ASE Atoms.
    """
    s = item["structure"]
    if "lattice" in s and "sites" in s:
        return AseAtomsAdaptor.get_atoms(Structure.from_dict(s))
    return Atoms(
        numbers=s["numbers"], positions=s["positions"], cell=s["cell"], pbc=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a fine-tuned MatGL checkpoint against its recorded prediction."
    )
    parser.add_argument("--run-dir", required=True, help="train_matgl.py --output-dir")
    parser.add_argument(
        "--val-data", required=True, help="Validation JSON used in training"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run = Path(args.run_dir)
    record = json.loads((run / "finetune_record.json").read_text())
    ref = record["result"]["reference_prediction"]
    data = json.loads(Path(args.val_data).read_text())
    atoms = to_atoms(data[ref["index"]])
    assert len(atoms) == ref["num_atoms"]

    pretrained = MatGLWrapper(model_name=record["model"], device=args.device)
    pretrained.load()
    e_pre = pretrained.static_calculation(atoms.copy())["energy"]

    tuned = MatGLWrapper(model_name=record["model"], device=args.device)
    tuned.load_checkpoint(str(run / "fine_tuned_model.pth"))
    e_tuned = tuned.static_calculation(atoms.copy())["energy"]

    out = {
        "checkpoint": "fine_tuned_model.pth",
        "structure": f"{ref['dataset']}[{ref['index']}] ({ref['num_atoms']} atoms)",
        "energy_recorded_after_training_eV": ref["energy_eV"],
        "energy_reloaded_checkpoint_eV": e_tuned,
        "abs_difference_meV": abs(e_tuned - ref["energy_eV"]) * 1000,
        "energy_pretrained_eV": e_pre,
        "fine_tuning_shift_meV": (e_tuned - e_pre) * 1000,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
