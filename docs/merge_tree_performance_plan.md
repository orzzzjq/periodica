# Merge tree backend: performance plan

**Status (2026-10-21): phases 1-5 implemented** (`merge_tree.cpp` rewritten; `grid_merge_tree`;
native `MergeTree`; `-c opt` default build; `barcode(..., drop_zero_length)`). Results at the end.

Target workload: ~126K Materials Project charge densities (`scripts/process_chgcars.py`),
3D grids with a median of 1.3M points (90th percentile 3.7M, 99th 7.5M, max 157M;
2.3e11 points in total), one periodic merge tree + barcode per grid.

Correctness harness for every step: `tests/test_grid_barcode.py` (golden barcodes of seven
grid inputs matched within 1e-8 in canonical form, arc-order/orientation invariance, lattice
scaling, gudhi cross-check in 3D), plus `tests/test_periodic_grid.py` and
`tests/test_hidden_points.py`. A change is acceptable only if all three pass unchanged.

## Baseline (mp-1, 80^3 = 512K vertices, 1.5M arcs, one core)

| build | merge_tree | barcode | max RSS |
| --- | --- | --- | --- |
| current `make build` (bazel default = fastbuild, no optimization) | 24.8 s | 3.0 s | 0.8 GB |
| same code, `bazel build -c opt` | 1.8 s | 0.3 s | 0.8 GB |

Optimized build, random 3D grids: 64K vertices 0.18 s, 262K 0.86 s, 1M 3.7 s, 2.1M 7.9 s,
i.e. ~3.7 µs per vertex for the tree and ~0.7 µs for the barcode, close to linear.
Memory is ~1.2 KB per vertex (Python arrays included).

Whole dataset with the optimized build alone: ~280 CPU hours, ~9 h on 32 workers. The
99th-percentile grid needs ~9 GB, the largest ~190 GB, so memory, not time, is what limits
parallelism on the big grids.

## Where the time goes (from reading `periodica/cpp/merge_tree.cpp`)

1. **No compiler optimization.** Eigen and the templated containers are dramatically slower
   at `-O0`; this is 14x by itself.
2. **Heap-heavy data layout.** Every `Vertex` owns an Eigen dynamic `drift` vector and a
   dynamic `Lattice` matrix, every `Arc` a dynamic shift vector, and `beams` is a
   `vector<vector<Event>>` with one vector per vertex. That is three to four heap blocks per
   vertex and one per arc, scattered in memory; `std::sort` on the arcs moves these objects.
3. **Catenation path does real work for every non-tree arc.** In a grid ~2/3 of the arcs
   close a cycle. For each, the code builds a `Lattice` from the cycle vector, joins it with
   the component lattice through a Hermite normal form, computes the volume, and then
   usually discards the event because nothing changed. Nearly all of these cycle vectors are
   zero (contractible cycles) and the rest almost always lie in the existing lattice.
4. **Merger path always joins lattices.** Most mergers join two bounded components whose
   lattices are empty; the HNF join is still executed.
5. **Component relabeling.** Union is by size with a linked list through the smaller
   component, updating every member's root and drift. This is O(n log n) and cache-hostile;
   there is no path compression.
6. **Python round trips.** `merge_tree` returns the tree as n Python lists of tuples
   (512K lists for mp-1) and `barcode` converts them back into C++ before extracting bars.
   Inputs are copied from int64 numpy arrays into column-major `MatrixXi`.
7. **Arc materialization on the Python side.** `periodic_grid` builds 3n arcs with shifts
   in numpy (fast, 0.15 s, but it triples the input size that crosses into C++).

## Plan

Each phase ends with the three test scripts passing and a timing on mp-1 and on a 128^3
random grid. Expected cumulative speedups are estimates against the current build.

### Phase 1: build optimized (1 line, 14x, no code change)

Add `-c opt` to `BAZEL_FLAGS` in the Makefile so `make build` / `make` produce an optimized
`_periodica.so`; keep a `make build-debug` (fastbuild) target for debugging. Optionally
strip the `.so`. This alone makes the full dataset feasible (~9 h on 32 workers).

### Phase 2: fast paths in the union-find loop (expected 2-4x on top)

- Catenation: compute the cycle vector `v = drift[x] + shift - drift[y]` into a fixed-size
  integer array; if `v == 0` continue immediately; if the root lattice already has full
  rank continue; if `v` is in the span of the current basis (rank 1: parallel check,
  rank 2: cross product with the normal) continue. Only genuine rank or volume changes reach
  the HNF, which is a handful of times per component.
- Merger: skip the lattice join when either lattice is empty; when both are non-empty,
  join as today.
- Keep the semantics of event emission identical (same checks on exponent and ratio), so
  the barcode is bit-for-bit the same except for the order of equal-time events, which the
  canonical comparison in the test absorbs.

### Phase 3: data layout (expected 2-3x more, 5-10x less memory)

- Struct-of-arrays with fixed dimension: `drift` as `int32[3]` per vertex, the lattice basis
  as a `3x3 int32` plus rank stored per root only (or in a side table indexed by root),
  `old`, `oldId`, `size`, `root`, `next` as flat arrays.
- Arcs: do not build `Arc` objects. Sort an index array (or `(filtration, index)` pairs) and
  read source, target, shift from the input matrices through `Eigen::Ref` without copying
  (take row-major int32 from Python, or convert once in C++).
- Events: births are implicit (the vertex filtration), so store only the events of roots in
  one flat `vector<Event>` with per-root offsets, and have `addRightEnd` /
  `constructBarcodes` iterate it. Target: ~100 bytes per vertex, so the 157M-point grid fits
  in ~16 GB.
- Replace the linked-list relabel with classic union-find (union by size, path compression)
  carrying the drift as a potential along parent pointers, compressed during `find`. Fewer
  memory touches and no per-member loop on union.

### Phase 4: grid-native entry point (removes the arc materialization, expected 1.5-2x)

For a lower-star filtration the arc order is implied by the vertex order: sort the n vertex
values once, then insert vertices in increasing order and union each new vertex with its
already-inserted neighbors along the d grid directions at time f(v). This needs no explicit
arc list or arc sort (n log n on floats instead of 3n log 3n on objects), takes `(U, values,
directions, sublevel)` directly and produces the same union-find event sequence. Expose it as
`_periodica.grid_merge_tree` and use it in `Periodica.periodic_grid` / `merge_tree` when the
complex came from a grid, keeping the generic `merge_tree` for Delaunay/Voronoi/file inputs.

### Phase 5: keep the tree native (removes the Python round trips)

Return the merge tree as a pybind class holding the C++ event storage; `barcode` and
`print_merge_tree` take that object, and a `to_list()` method reproduces today's list of
lists for compatibility. For mp-1 this removes two conversions of 512K Python lists.

### Phase 6: dataset-level throughput

- `scripts/process_chgcars.py --workers N` already parallelizes over files; after Phase 3
  the memory per worker is small enough to run ~48 workers on this machine. Sort the work by
  grid size and run the few huge grids (> 20M points) in a separate low-parallelism queue.
- Optional: compute the barcode in C++ right after the tree (one call per grid) and return
  only bars, so the per-file Python overhead is a single native call.

### Not planned

- Multithreading inside one merge tree: the union-find loop is sequential and the dataset
  gives ample file-level parallelism.
- Changing floating point types: filtration values stay `double` (the barcode goldens are
  matched at 1e-8).

## Results after phases 1-5 (same machine, one core)

| grid | merge_tree before | merge_tree after | barcode before | barcode after (all bars / drop_zero_length) | max RSS before → after |
| --- | --- | --- | --- | --- | --- |
| mp-1, 80³ = 512K vertices | 24.8 s | 0.20 s (grid-native), 0.34 s (explicit arcs) | 3.0 s | 0.16 s / 0.05 s | 0.8 GB → 0.24 GB |
| random 100³ = 1M (143K bars) | 3.7 s (opt build) | 0.65 s | 0.76 s | 0.30 s / 0.06 s | 0.58 GB |
| random 128³ = 2.1M (300K bars) | 7.9 s (opt build) | 1.7 s | 1.5 s | 0.66 s / 0.14 s | 0.92 GB |
| random 200³ = 8M (1.1M bars) | | 8.0 s | | 2.9 s / 0.70 s | 2.5 GB (Python field and bars included) |

`scripts/process_chgcars.py mp-1` (load, grid, tree, barcode, curves, two files): 28 s → 0.3 s,
same barcode and curve vectors as before.

What changed, by phase: (1) `-c opt` in `make build`, `make build-debug` kept; (2) contractible
cycles (zero cycle vector) and fully periodic components skip the lattice work, empty lattices
are not joined; (3) `Lattice` in fixed-size integer storage with the same HNF, union-find with
path compression carrying the drift as a potential, births implicit, events in one flat array
linked per beam with a per-beam death index and death time instead of copied right-end
events, arcs sorted as `(filtration, index)` pairs; (4) `gridMergeTree` inserts grid points in
value order and unions with already-inserted neighbors along the grid directions, so no arc
list exists (`periodic_grid(..., build_arcs=False)` skips the Python arrays as well); (5)
`MergeTree` pybind class, list-like for the web UI and `to_list()`, `barcode` and
`print_merge_tree` accept both forms, `barcode(..., drop_zero_length)` with a sequential fast
path for the trivial beams. All goldens in `tests/test_grid_barcode.py` are reproduced by both
the grid-native and the explicit-arc path.

Remaining per-vertex cost is the union-find itself (~1 µs per vertex on 8M grids, cache
bound) and, when all bars are requested, Python tuple creation. Phase 6 (scheduling by grid
size, a separate queue for the largest grids) is a matter of running
`scripts/process_chgcars.py --workers N` sorted by size; at ~1 µs per point the whole dataset
(2.3e11 points) is ~65 CPU hours.
