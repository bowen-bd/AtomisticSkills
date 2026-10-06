# Draft upstream issue: NValchemi neighbor-list cache

Draft report for [NVIDIA/nvalchemi-toolkit](https://github.com/NVIDIA/nvalchemi-toolkit),
not yet filed. When it is filed, link the issue here and in
[nvalchemi-neighbor-cache.md](nvalchemi-neighbor-cache.md), which describes how
AtomisticSkills 2.0.0 guards against it in the meantime.

---

## NeighborListHook silently loses periodic neighbors after same-size cell changes

`NeighborListHook` in release 0.2.0 and in upstream main at
[`4fe9560`](https://github.com/NVIDIA/nvalchemi-toolkit/commit/4fe95603572831ba2f6adad4cfe0102bf7f8cc08)
caches geometry-dependent neighbor-list allocations using only the total atom
count and the graph count.

An inflight replacement or a variable-cell update can change the required
periodic images without changing either count. Calling the hook again then
silently produces a different graph from a fresh hook at the identical geometry.

On an NVIDIA A100, two one-atom periodic systems were changed from 8 Å cubic
cells to 2 Å cells, with a 5 Å cutoff:

| Method | Reused hook (before fix) | Fresh hook / fixed hook |
| --- | ---: | ---: |
| Automatic | 0 edges | 160 edges |
| `batch_naive` | 52 edges | 160 edges |

The same script on an NVIDIA GB10 (aarch64, CUDA 13, release 0.2.0) gives the
same counts.

The reproduction needs no trained model and no downstream code:

```python
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat

def batch(length):
    return Batch.from_data_list([
        AtomicData(
            positions=torch.zeros(1, 3),
            atomic_numbers=torch.tensor([29]),
            cell=torch.eye(3).unsqueeze(0) * length,
            pbc=torch.ones(1, 3, dtype=torch.bool),
        ) for _ in range(2)
    ]).to("cuda")

cfg = NeighborConfig(cutoff=5.0, format=NeighborListFormat.COO)
stage = DynamicsStage.BEFORE_COMPUTE
for method in (None, "batch_naive"):
    reused = NeighborListHook(cfg, method=method, max_neighbors=1024)
    reused(HookContext(batch=batch(8.0)), stage)
    changed, reference = batch(2.0), batch(2.0)
    reused(HookContext(batch=changed), stage)
    fresh = NeighborListHook(cfg, method=method, max_neighbors=1024)
    fresh(HookContext(batch=reference), stage)
    print(method, changed.neighbor_list.shape[0], reference.neighbor_list.shape[0])
```

Realistic geometries reach it too. Refilling a batch slot with a same-size
structure in a different cell (two-atom Si polymorphs, 5–6 Å cutoffs) loses
4–20 of 132–208 edges. Gradual compression or shear of 4–12% also loses edges.
Both searches are described in
[nvalchemi-neighbor-cache.md](nvalchemi-neighbor-cache.md).

### Proposed fix

Before overwriting the staging values, compare the cell, the periodicity and the
batch partition with the previous ones:
- a change invalidates the geometry-dependent allocations, the adaptive neighbor
  capacity and the Verlet references;
- ordinary fixed-cell steps keep their buffers.

The fix belongs in NValchemi; downstream code should not need to invalidate the
cache itself.

This change puts correctness first. It uses host-side metadata comparisons and
reallocates whenever a cell changes. Reusing allocations within validated
geometric bounds can be optimized separately; no throughput improvement is
claimed.

### Regression coverage

The proposed tests cover:
- COO and MATRIX formats;
- automatic and naive method selection;
- zero and nonzero skin;
- same-size replacement and in-place cell updates;
- PBC changes;
- changing per-system atom counts at constant total dimensions.

Edge comparisons include image shifts and ignore the nondeterministic CUDA
enumeration order.

Results:
- The first eight new regressions fail on unmodified main.
- With the fix, the original 117 hook tests and all 18 new regressions pass
  (135 passed).
- Ruff lint/format and license-header checks pass. The full pre-commit
  pipeline was not run.
- The reproduction above was also run on unmodified and fixed main: automatic
  0 → 160 edges, `batch_naive` 52 → 160, with fresh hooks returning 160 both
  times.

Environment:
- Validation used CPython 3.12.13, PyTorch 2.14.1+cu126 and an NVIDIA A100 80 GB.
- Release integration used the nvalchemi-toolkit 0.2.0 source plus this fix,
  with the installed ops 0.4.1.
- Main-branch tests used the ops 0.5.0-rc source that main's `pyproject.toml`
  specifies.
- Installed packages were not modified.

---

## Separate release issue: inactive geometry and energy disagree

In release 0.2.0,
[`BaseDynamics.step()`](https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.2.0/nvalchemi/dynamics/base.py)
restores the coordinates and cells of inactive systems after model evaluation,
but leaves the newly evaluated outputs in place.

In a mixed TensorNet variable-cell run, this returned −340.4951 eV for a frozen
Si64 geometry whose fresh energy was −343.1376 eV. Snapshotting the first
converged state avoids observing this later corruption, but the returned live
batch is still inconsistent.

Main already publishes outputs under the active mask and tests it. Please
include that repair in the next release or in a 0.2 maintenance release. This is
distinct from the neighbor-cache issue, and no downstream recomputation
workaround has been added.
