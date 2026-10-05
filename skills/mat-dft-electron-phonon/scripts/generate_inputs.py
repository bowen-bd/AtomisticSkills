"""
Generate and serialize the Jobflow DAG for computing electron-phonon coupling (Temperature bandgap shift).

Usage:
    python generate_inputs.py --output elph_flow.json

Requirements:
    - Environment: cpu (run with: venv/run cpu python ...)
    - Required packages: atomate2, phonopy, pymatgen, jobflow
"""

import argparse
from pymatgen.core import Structure
from atomate2.vasp.flows.elph import ElectronPhononMaker


def main():
    parser = argparse.ArgumentParser(
        description="Generate ElectronPhononMaker flow DAG for Silicon."
    )
    parser.add_argument(
        "--output",
        default="elph_flow.json",
        help="Output JSON path to save the DAG representation.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Also submit the flow with jobflow-remote (requires --project and --worker).",
    )
    parser.add_argument("--project", help="jobflow-remote project to submit to.")
    parser.add_argument("--worker", help="jobflow-remote worker to run the jobs on.")
    args = parser.parse_args()
    if args.submit and not (args.project and args.worker):
        parser.error("--submit requires --project and --worker")

    # Silicon primitive cell (FCC)
    si_structure = Structure(
        lattice=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]],
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )

    # Initialize the automated Electron-Phonon workflow
    # This automatically includes:
    # 1. Structure relaxation (Tight)
    # 2. Phonon execution (generating supercell displacements and extracting force constants)
    # 3. Supercell uniform random displacements based on Bose-Einstein occupations for specified temperatures
    # 4. Dense static bandgap calculations for each displaced configuration
    # 5. Averaging bandgaps to extract the ZGMR (Zero-Point Renormalization) and temperature shifts
    maker = ElectronPhononMaker(
        name="Si_Electron_Phonon",
        temperatures=(
            0,
            300,
            600,
        ),  # Evaluate T=0K (quantum fluctuations), 300K, and 600K
        min_supercell_length=10.0,  # Minimum supercell length in Angstroms
    )

    flow = maker.make(si_structure)

    # Write the flow, as --output promises, so it can be inspected or
    # submitted later. Submitting is opt-in: the target project and worker
    # belong to the user, and nothing should reach a queue by default.
    from monty.serialization import dumpfn

    dumpfn(flow, args.output)
    print(
        f"Wrote the Electron-Phonon workflow ({len(flow.jobs)} top-level jobs) to {args.output}"
    )

    if args.submit:
        from jobflow_remote import submit_flow

        flow_ids = submit_flow(flow, project=args.project, worker=args.worker)
        print(f"Submitted to {args.project}/{args.worker}: {flow_ids}")


if __name__ == "__main__":
    main()
