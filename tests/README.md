# Testing Guide

## Overview

This project uses **pytest** with a multi-environment strategy: the MCP servers
run in different uv projects (`venv/<name>`), so each server's tests run in its
environment, through the launcher. `tests/conftest.py` skips tests whose marker
needs another environment.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and environment detection
├── test_*.py                # Launcher, skills, uv projects, images, tool CLI (cpu)
├── base/ atomate2/ drugdisc/ smol/ orca/   # CPU-stack servers (cpu)
├── mace/ matgl/             # MLIP servers (mlip)
├── fairchem/                # FairChem server (fairchem)
├── adit/ diffcsp/ mattergen/  # Generative servers (their own environments)
└── utils/                   # Shared utilities, incl. NValchemi batch tests (mlip, fairchem)
```

## Running Tests

```bash
# Launcher, skill conventions, uv projects, images and the tool CLI (what CI runs)
venv/run cpu python -m pytest tests/test_launcher.py tests/test_skill_runtime.py \
    tests/test_uv_projects.py tests/test_images_and_manifests.py tests/test_tool_cli.py

# One server's tests, in its environment
venv/run cpu python -m pytest tests/base tests/atomate2 tests/drugdisc tests/smol
venv/run mlip python -m pytest tests/mace tests/matgl
venv/run fairchem python -m pytest tests/fairchem
venv/run mattergen python -m pytest tests/mattergen

# Or by marker
venv/run mlip python -m pytest -m mace
```

## Test Markers

Tests are marked by the server they cover:

- `@pytest.mark.base`, `atomate2`, `drugdisc`, `smol`, `orca` - CPU-stack servers (`cpu`; also `mlip`, `fairchem`)
- `@pytest.mark.mace`, `matgl` - MLIP servers (`mlip`)
- `@pytest.mark.fairchem` - FairChem server (`fairchem`)
- `@pytest.mark.adit`, `diffcsp`, `mattergen` - generative servers (their own environments)

## Auto-Skip Behavior

Tests automatically skip if run in the wrong environment:

```python
@pytest.mark.mace
def test_mace_feature():
    # Skipped unless run in the mlip environment:
    #   venv/run mlip python -m pytest tests/mace
    ...
```

## Breaking Changes Tested

### 1. `load_structure_from_file()` Returns Pymatgen Structure

**Old behavior**: Returned ASE Atoms
**New behavior**: Returns Pymatgen Structure

**Test**: `tests/base/test_structure_utils.py::TestLoadStructureFromFile`

### 2. `save_structure()` Standardization

**New function**: Handles both ASE Atoms and Pymatgen Structures

**Test**: `tests/base/test_structure_utils.py::TestSaveStructure`

### 3. Materials Project Query Simplification

**Changed functions**: `get_structure_by_formula`, `get_structure_by_chemsys`, `get_structure_by_id`

**Test**: `tests/base/test_structure_utils.py::TestMaterialsProjectQueries`

## New Features Tested

### 1. Atomic Feature Extraction

**MACE**: `tests/mace/test_mace_wrapper.py::TestMACEPredictAtomicFeatures`
**MatGL**: `tests/matgl/test_matgl_wrapper.py::TestMatGLPredictAtomicFeatures`

### 2. Base Relax Structure Implementation

**Test**: `tests/base/test_base_mlip.py::TestRelaxStructureBase`

## Common Fixtures

Available in all tests via `conftest.py`:

- `current_env` - The uv project running the tests (`venv/<name>`)
- `skip_if_wrong_env` - Auto-skip if wrong environment
- `tmp_cif_file` - Temporary Si2 CIF structure
- `sample_structure` - Pymatgen Structure (Si2)
- `sample_ase_atoms` - ASE Atoms (Si2)
- `tmp_research_dir` - Temporary research directory
- `mock_mp_api_key` - Mock Materials Project API key

## Troubleshooting

### Test fails with import error

**Problem**: The tests ran in another environment (for example a bare
`pytest`, outside the launcher).

**Solution**: Run them through the launcher in the server's environment, e.g.
`venv/run mlip python -m pytest tests/mace`. `venv/run --doctor` lists the
environments and how this host runs them.

### Test skips unexpectedly

**Problem**: Auto-skip triggered: the marker needs another environment, or a
package has no wheel for this architecture (SCINE on aarch64).

**Solution**: The skip reason names the environment to use.

## CI

`.github/workflows/uv-envs.yml` runs the launcher, skill, uv-project, image and
tool-CLI suites, and each environment's MCP smoke test, on x86_64 and arm64.

Environment CI runs on pull requests, pushes to `main`, version tags, and manual
requests. Branch pushes do not duplicate the pull request jobs. Changes only
under `docs/` are excluded, but GitHub evaluates PR path filters against the
whole PR diff, so a docs-only follow-up on a code PR can still rerun CI.

Container builds run when the image workflow's path filters match (container
files, dependency definitions, and the image workflow itself; MCP server changes
also trigger publication on `main`). CUDA image builds can take much longer
than the environment checks, especially when native extensions need compiling
or large cache layers need uploading. These builds remain required for changes
that affect the container setup.
