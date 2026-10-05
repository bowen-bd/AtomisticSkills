# Conda environments

Since 2.0.0, every skill and MCP server runs from a uv project under `venv/`
through `venv/run`: the shared projects `cpu`, `mlip` and `fairchem`, and one
pinned project per research stack (`adit`, `diffcsp`, `mattergen`, `msms`,
`reactot`, `scd`). Nothing here is needed to run them.

Two things still use conda:

| Environment | Used by | Why |
| :--- | :--- | :--- |
| `mace-agent`, `matgl-agent`, `fairchem-agent` | `mat-lammps-md` | `install_lammps.sh` compiles LAMMPS against each MLIP's C++ library in these environments |
| `adit-agent`, `diffcsp-agent`, `mattergen-agent` | building the arm64 `generative` container image | PyG publishes no aarch64 wheels, so the image compiles the extensions with CUDA from these lockfiles; on x86_64 the same servers run from `venv/adit`, `venv/diffcsp` and `venv/mattergen` |

Each directory has an `install.sh`. `docker/Dockerfile.cuda` builds the
`generative` image from the lockfiles in `<env>/lock/`, which
`docker/export_locks.py` writes. Users do not build these environments: on
aarch64, `venv/run` pulls the published `generative` image.

The other 1.x environments (`base-agent`, `atomate2-agent`, `drugdisc-agent`,
`smol-agent`, `nmr-agent`, `phasefield-agent`, `calphad-agent`, `xrd-agent`,
`orca-agent`, `drugmd-agent`, `void-agent`, `react-ot-agent`, `msms-agent`
and `scd-agent`) were removed in 2.0.0; they remain in the git history of the
1.x releases. See [docs/changes/2.0.0-migration.md](../docs/changes/2.0.0-migration.md)
for the mapping to uv projects and extras.
