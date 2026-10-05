# Conda environments

Since 2.0.0, AtomisticSkills runs on three uv projects (`venv/cpu`, `venv/mlip`,
`venv/fairchem`) through `venv/run`. The conda environments here are the few
stacks that do not fit a modern uv project. Each directory has an `install.sh`.
Skills that need one declare it as `metadata.conda_env` in their `SKILL.md`.

| Environment | Used by | Why it stays on conda |
| :--- | :--- | :--- |
| `adit-agent` | `ml-generative-adit` (MCP server `adit`) | not on PyPI, compiled extensions |
| `diffcsp-agent` | `ml-generative-diffcsp` (MCP server `diffcsp`) | not on PyPI, compiled PyG extensions |
| `mattergen-agent` | `ml-generative-mattergen` (MCP server `mattergen`) | pins torch 2.2 (cu118) and source-built PyG extensions |
| `msms-agent` (env name `ms-gen`) | `chem-msms-predict` | ICEBERG (`ms-pred`) needs an old torch stack |
| `react-ot-agent` | `chem-react-ot` | old torch / torch-geometric requirements; also holds its model downloader |
| `scd-agent` | `ml-property-predict-scd` examples | the example training runs relaunch inside it; the skill's own scripts run on `venv/mlip` |
| `mace-agent`, `matgl-agent`, `fairchem-agent` | `mat-lammps-md` | `install_lammps.sh` builds LAMMPS with each MLIP's C++ plugin |

The three generative environments are also the source of the `generative`
container image. `docker/Dockerfile.cuda` installs them from the lockfiles in
`<env>/lock/`, which `docker/export_locks.py` writes. `configure_mcp.py` points
the `adit`, `diffcsp` and `mattergen` servers at these environments when it
finds them under your conda installation, and otherwise at the image.

The environments that the uv projects replaced (`base-agent`, `atomate2-agent`,
`drugdisc-agent`, `smol-agent`, `nmr-agent`, `phasefield-agent`,
`calphad-agent`, `xrd-agent`, `orca-agent`, `drugmd-agent`, `void-agent`) were
removed in 2.0.0. They remain in the git history of the 1.x releases. See
[docs/changes/2.0.0-migration.md](../docs/changes/2.0.0-migration.md) for the
mapping to uv projects and extras.
