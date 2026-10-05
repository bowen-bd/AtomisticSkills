# Environment Variables & Configuration Guide

This document lists the environment variables and configuration settings used by AtomisticSkills for the `venv/run` launcher, MCP servers, and skill scripts.

## Configuration File (`~/.config/atomistic_skills.yaml`)

Rather than exporting individual environment variables, settings can be defined in `~/.config/atomistic_skills.yaml` (or the legacy path `~/.atomistic_skills.yaml`). Environment variables of the same name take precedence over this configuration file.

Example template:
```yaml
# How skills and MCP servers run: auto (default), uv, docker, podman, apptainer
ATOMISTIC_RUNTIME: auto

# Materials Project API Key (mat-db-mp, base server, many workflows)
MP_API_KEY: "your_mp_api_key_here"

# Hugging Face token for gated models such as FairChem UMA
HF_TOKEN: "your_hf_token_here"

# Remote project for DFT calculations via atomate2 / jobflow-remote
ATOMATE2_REMOTE_PROJECT: "my_cluster"

# Path to ORCA binary for chem-dft-orca-* skills (x86_64 only)
ORCA_BINARY_PATH: "/path/to/orca_directory/orca"

# Optional container image registry and tag (for testing forks or custom images)
ATOMISTIC_IMAGE_REGISTRY: "ghcr.io/learningmatter-mit"
ATOMISTIC_IMAGE_TAG: "2.0.0"
```

## Runtime & Launcher Variables

| Variable | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `ATOMISTIC_RUNTIME` | Execution backend for `venv/run`. Options: `auto`, `uv`, `docker`, `podman`, `apptainer`, `singularity`. | `auto` | `apptainer` |
| `ATOMISTIC_WORKSPACE` | Workspace directory where simulation results and research folders are saved. Defaults to current checkout inside a repo, or the working directory. | `$PWD` or repo root | `/home/user/my-atomistic-project` |
| `ATOMISTIC_MODEL_CACHE` | Directory for downloaded model checkpoints and Apptainer SIF container files. | `~/.cache/atomisticskills` | `/scratch/user/atomistic-cache` |
| `ATOMISTIC_IMAGE_REGISTRY` | Container image registry namespace. | `ghcr.io/learningmatter-mit` | `ghcr.io/myfork` |
| `ATOMISTIC_IMAGE_TAG` | Tag for the container images. | Matches release version (`2.0.0`) | `2.0.0` |
| `UV_CACHE_DIR` | Cache directory for `uv` downloads and wheels. | uv standard default | `/scratch/user/uv-cache` |
| `CURRENT_RESEARCH_DIR` | Active research directory for the current session. Automatically managed by tools. | (set at runtime) | `/path/to/research/2026-10-04_melting_point` |

## Service API Keys & Credentials

| Variable | Description | Required By | Example |
| :--- | :--- | :--- | :--- |
| `MP_API_KEY` | API Key for Materials Project database access. | `base` server, pymatgen | `abc123def456` |
| `HF_TOKEN` | Hugging Face user access token. Required for FairChem UMA, which is gated: request access at https://huggingface.co/facebook/UMA first. | `fairchem` server | `hf_...` |
| `SSL_CERT_FILE` | CA bundle for Python's TLS. `venv/run` sets it to the system bundle when unset (needed on RHEL-family hosts); set it yourself behind a proxy with its own CA. | all | `/etc/pki/tls/certs/ca-bundle.crt` |
| `ATOMISTIC_TORCH_CUDA` | Torch build for the GPU environments (`mlip`, `fairchem`, `adit`, `diffcsp`, `mattergen`, `scd`): `cu130` (CUDA 13, driver ≥ 580) or `cu126` (CUDA 12.6, driver ≥ 525). Chosen from the NVIDIA driver when unset. | `venv/run` | `cu126` |
| `UV_PYTHON_PREFERENCE` | Which Python uv builds environments on. `venv/run` uses `only-managed` (a uv-managed CPython with headers) unless set. | `venv/run` | `only-managed` |
| `ORCA_BINARY_PATH` | Full path to the external ORCA executable. | `chem-dft-orca-*` skills | `/opt/orca/orca` |
| `ATOMATE2_REMOTE_PROJECT` | Jobflow-remote project name for remote cluster job submission. | `atomate2` server | `my_cluster` |
| `ATOMATE2_CONFIG_FILE` | Optional path to custom atomate2 config YAML. | `atomate2` server | `~/.config/atomate2/config.yaml` |
| `OPENALEX_EMAIL` | Contact email for polite OpenAlex API queries. | Literature skills | `user@institution.edu` |
| `UNPAYWALL_EMAIL` | Contact email for Unpaywall open-access queries. | Literature skills | `user@institution.edu` |
| `ELSEVIER_API_KEY` | API Key for Elsevier ScienceDirect access (optional). | Literature skills | `key_value` |

## Source Checkouts

Some research stacks import an upstream repository that is not a package. Each
is looked up through a variable, with a default location:

| Variable | Repository | Default | Used by |
| :--- | :--- | :--- | :--- |
| `ADIT_REPO` | ADiT (all-atom-diffusion-transformer) | `adit` next to this project | `adit` server, `ml-generative-adit` |
| `DIFFCSP_REPO` | DiffCSP++ (DiffCSP-PP) | `DiffCSP-PP` next to this project | `diffcsp` server, `ml-generative-diffcsp` |
| `REACT_OT_DIR` | React-OT | `~/.cache/atomisticskills/react-ot` | `chem-react-ot` |
| `SCD_REPO_DIR` | SelfConditionedDenoisingAtoms | `~/.cache/atomisticskills/SelfConditionedDenoisingAtoms` | `ml-property-predict-scd` |

When a server runs in a container, `venv/run` passes `ADIT_REPO` and
`DIFFCSP_REPO` on and mounts those checkouts at the same path.
