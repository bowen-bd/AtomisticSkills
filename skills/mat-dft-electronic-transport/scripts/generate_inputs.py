"""
Generate and serialize the Jobflow DAG for computing electronic transport properties using AMSET and VASP.

Usage:
    python generate_inputs.py --output amset_flow.json

Requirements:
    - Environment: cpu (run with: venv/run cpu python ...)
    - Required packages: atomate2, pymatgen, jobflow, amset
"""

import argparse
from pymatgen.core import Structure
from atomate2.vasp.flows.amset import VaspAmsetMaker


def main():
    parser = argparse.ArgumentParser(
        description="Generate VaspAmsetMaker flow DAG for GaAs."
    )
    parser.add_argument(
        "--output",
        default="amset_flow.json",
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

    # GaAs primitive cell (FCC lattice)
    gaas = Structure(
        lattice=[[0.0, 2.825, 2.825], [2.825, 0.0, 2.825], [2.825, 2.825, 0.0]],
        species=["Ga", "As"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )

    # Initialize the automated AMSET workflow
    # This automatically includes:
    # 1. Structure relaxation
    # 2. Dense uniform band structure
    # 3. Elastic tensor calculation
    # 4. Deformation potential calculation
    # 5. Static & dielectric constants
    # 6. Final AMSET execution
    maker = VaspAmsetMaker(
        name="GaAs_AMSET_Transport",
        doping=(1e16, 1e17, 1e18),  # Carrier concentrations in cm^-3
        temperatures=(300.0, 400.0),  # Temperatures in K
        use_hse_gap=False,  # Keep fast PBE for demonstration
    )

    flow = maker.make(gaas)

    # Write the flow, as --output promises, so it can be inspected or
    # submitted later. Submitting is opt-in: the target project and worker
    # belong to the user, and nothing should reach a queue by default.
    from monty.serialization import dumpfn

    dumpfn(flow, args.output)
    print(
        f"Wrote the Amset workflow ({len(flow.jobs)} top-level jobs) to {args.output}"
    )

    if args.submit:
        from jobflow_remote import submit_flow

        flow_ids = submit_flow(flow, project=args.project, worker=args.worker)
        print(f"Submitted to {args.project}/{args.worker}: {flow_ids}")


if __name__ == "__main__":
    main()
