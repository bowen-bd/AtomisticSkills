import argparse
import os
import sys

from atomate2.vasp.jobs.core import RelaxMaker
from pymatgen.core import Structure


def submit_test_job(project_name: str | None = None):
    """Submit a test job to a remote cluster project."""
    try:
        from jobflow_remote.submission import submit_flow
    except ImportError:
        print("Error: jobflow-remote is required to submit remote jobs.")
        sys.exit(1)
    # standard Si structure
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # fast relaxation
    job = RelaxMaker().make(structure)
    job.name = "Remote Test Job"

    # Submit to remote
    project = project_name or os.environ.get("ATOMATE2_REMOTE_PROJECT")
    if not project:
        print(
            "Error: No remote project specified. Pass --project or set ATOMATE2_REMOTE_PROJECT."
        )
        sys.exit(1)

    print(f"Submitting job to project: {project}")
    try:
        submit_flow(job, project=project)
        print("Flow submitted successfully.")
        print("Now run the following command to push it to the remote worker:")
        print(f"  jf runner run -p {project}")
        print("\nThen check status with:")
        print(f"  jf remote status -p {project}")
    except Exception as e:
        print(f"Failed to submit flow: {e}")
        print(f"\nCheck if your ~/.jfremote/projects/{project}.yaml is valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a test job to a remote worker."
    )
    parser.add_argument(
        "--project",
        default=None,
        help="jobflow-remote project name (defaults to ATOMATE2_REMOTE_PROJECT env var)",
    )
    args = parser.parse_args()
    submit_test_job(args.project)
