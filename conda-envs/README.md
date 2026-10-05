# Conda environments

Since 2.0.0, AtomisticSkills runs on three uv projects (`venv/cpu`, `venv/mlip`,
`venv/fairchem`) through `venv/run`. The conda environments here are the few
stacks that do not fit a modern uv project. Each directory has an `install.sh`.
Skills that need one declare it as `metadata.conda_env` in their `SKILL.md`.

| Environment | Used by | Why it stays on conda |
| :--- | :--- | :--- |
| `adit-agent`, `diffcsp-agent`, `mattergen-agent` | the arm64 `generative` container image | their lockfiles build that image; on x86_64 the servers run from the uv projects `venv/adit`, `venv/diffcsp`, `venv/mattergen` |
| `scd-agent` | `ml-property-predict-scd` examples | the example training runs relaunch inside it (migration to uv in progress) |
| `mace-agent`, `matgl-agent`, `fairchem-agent` | `mat-lammps-md` | `install_lammps.sh` builds LAMMPS with each MLIP's C++ plugin |

The generative servers do not need these environments on a host: `venv/run`
runs them from their uv projects on x86_64 and from the `generative` image on
aarch64. `docker/Dockerfile.cuda` builds that image from the lockfiles in
`<env>/lock/`, which `docker/export_locks.py` writes.

The environments that the uv projects replaced (`base-agent`, `atomate2-agent`,
`drugdisc-agent`, `smol-agent`, `nmr-agent`, `phasefield-agent`,
`calphad-agent`, `xrd-agent`, `orca-agent`, `drugmd-agent`, `void-agent`, and
`react-ot-agent`, now `venv/reactot`, and `msms-agent`, now `venv/msms`) were removed in 2.0.0. They remain in the git history of the 1.x releases. See
[docs/changes/2.0.0-migration.md](../docs/changes/2.0.0-migration.md) for the
mapping to uv projects and extras.
