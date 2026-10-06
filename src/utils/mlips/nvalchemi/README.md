# NValchemi integration

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
