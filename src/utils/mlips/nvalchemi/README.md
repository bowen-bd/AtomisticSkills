# NValchemi integration

NValchemi is an optional experimental batch engine. Public wrapper and MCP
calls default to native sequential execution; `use_nvalchemi=True` selects it
for that request only. See the [skill](../../../../skills/ml-mlip-nvalchemi/SKILL.md)
for confirmed 0.2.0 dynamics limitations before enabling it.

The model factories adapt MACE, MatGL and FairChem to the batched execution
paths in `MLIPModel`. `nvalchemi_utils.py` provides conversion, convergence,
trajectory and stream-management helpers.

`neighbor_list.make_neighbor_list_hook()` is the shared constructor for
neighbor hooks in fixed relaxation, inflight relaxation and batch MD. For
toolkit 0.2.x it invalidates geometry-dependent allocations when cells,
periodicity or the atom partition change. Installed packages are not modified.
See the [release investigation](../../../../docs/verification/nvalchemi-neighbor-cache.md)
for the reproduction, scope and measured overhead. Revalidate this compatibility
guard when changing the toolkit lock.
