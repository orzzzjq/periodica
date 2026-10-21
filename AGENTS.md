# CLAUDE.md

## What this project is

Periodica is a C++/Python research library for topological analysis of **periodic point sets**
(e.g. crystal structures): given a lattice basis and points in the unit cell, it computes
periodic Delaunay/Voronoi quotient complexes, **periodic merge trees**, and topological
descriptors (barcodes, persistence diagrams, persistence images).

Theory: Edelsbrunner & Heiss, *Merge Trees of Periodic Filtrations* (arXiv:2408.16575), and
Osang/Rouxel-Labbé/Teillaud, *Generalizing CGAL Periodic Delaunay Triangulations* (ESA 2020).
See `cecam_poster.pdf` for an overview.

## Build

```
make            # = setup + build; works on a fresh machine
make setup      # installs what is missing: uv, bazelisk, node/npm (brew or ~/.local), venv, node_modules
```

- `make setup` only installs what is missing; it never builds, and is a fast no-op once everything is present. `USE_BREW=0` forces the
  no-root download path (into `TOOLS_PREFIX`, default `~/.local`), whose `bin/` the Makefile
  prepends to `PATH` for all recipes. A C/C++ compiler is the one prerequisite it cannot install.
- Build system is **Bazel with bzlmod** (`MODULE.bazel`); use `bazelisk` if `bazel` is absent.
  `make build` builds with `-c opt` (`make build-debug` for an unoptimized build with debug
  info; the merge tree is >10x slower without optimization, so never benchmark a debug build).
- `make` bootstraps a **uv-managed Python 3.11 venv** at `.venv` with numpy/matplotlib/scipy,
  then runs `bazel build --enable_bzlmod //:periodica_so` and copies
  `bazel-bin/_periodica.so` → `periodica/_periodica.so`.
- After editing any C++ file you must rebuild (`make build`) before Python picks up changes.
- `make rebuild` = `bazel clean --expunge` + build (slow; dependencies are re-downloaded/rebuilt).
- C++20. CGAL is used header-only with `-DCGAL_NO_GMP=1 -DCGAL_NO_MPFR=1`
  (Boost multiprecision backend instead of GMP/MPFR).
- Third-party deps (all vendored via `MODULE.bazel` http_archive/git rules, no system installs):
  pybind11 2.13.6, Eigen 3.4.0, Boost 1.86 (headers), CGAL 6.0.1, GUDHI (pinned commit).

## Layout

```
periodica/cpp/          C++ core, compiled into _periodica.so
  delaunay.{h,cpp}      Geometry engine (~1200 lines): 2D/3D (weighted) Delaunay skeletons,
                        Euclidean MST, lattice basis reduction (reducedBasis), Dirichlet
                        domains, canonical point copies, periodic Delaunay/Voronoi quotient
                        complexes built via a 3x Dirichlet-domain construction
                        (pointsIn3xDomain, fullDelaunaySkeleton, fullVoronoiSkeleton).
  merge_tree.{h,cpp}    Periodic merge tree (namespace PMT) from a quotient complex;
                        barcode extraction.
  persistence_image.*   Persistence images from barcodes (impl in persistence_image_impl.h).
  periodica.cpp         pybind11 bindings only (module name: _periodica).
  auxiliary.{h,cpp}     Small shared helpers.
periodica/core.py       High-level Python `Periodica` class + all matplotlib visualization.
web/server/app.py       FastAPI backend for the web UI: POST /api/compute wraps the full
                        Delaunay pipeline and returns draw-ready JSON (resolved arc
                        coordinates, Dirichlet polytopes, barcodes with death=null for
                        infinite bars, persistence images). Run with `make web`.
web/frontend/           Vite + React + TypeScript frontend: react-three-fiber scene
                        (cell, points, quotient arcs, exact filtration surfaces/regions
                        on a slider; 2D ortho / 3D orbit) + Plotly descriptor tabs
                        (barcode/diagram/image).
                        Dev: `npm run dev` (proxies /api to :8000); `npm run build` puts
                        dist/ where the FastAPI app serves it statically.
periodica/__init__.py   Re-exports _periodica symbols + Periodica.
main.py                 Example driver (2D/3D lattices, weighted points, quotient files).
analysis.py             Extra analysis scripts.
examples/               Quotient-complex text files (format below).
BUILD.bazel             Targets: :periodica_core (cc_library), :_periodica.so (cc_binary,
                        linkshared), aliases :periodica and :periodica_so.
```

## Typical pipeline (Python API, `periodica/core.py`)

```python
from periodica.core import Periodica
p = Periodica()
p.set_geometry({'d': 2, 'U': np.eye(2), 'n_points': 1, 'points': np.array([[0.5, 0.5]]).T})
# NOTE: points are column-major, shape (d, n_points)
p.quotient_complex('delaunay')   # or 'voronoi'
p.merge_tree()
p.print_merge_tree()
p.plot_barcodes(); p.plot_curves(); p.plot_diagram(); p.plot_images()
p.plot_geometry('delaunay', show=True, slidebar=True)   # interactive filtration slider
# Alternative entry point: p.load_quotient_complex('examples/example_2d_1.txt')
```

Key native functions exposed by `_periodica` (see `periodica/cpp/periodica.cpp` for the full
list): `periodic_delaunay`, `periodic_voronoi`, `full_delaunay`, `full_voronoi`,
`merge_tree(n, d, V, arcs, arc_filtration, arc_shift, vertex_filtration=)` (returns a list-like
`MergeTree` object), `grid_merge_tree(d, U, shape, f, directions)` (grid complexes, no arc list;
what `Periodica.merge_tree` uses after `periodic_grid`), `barcode(d, tree, drop_zero_length=False)`,
`image(barcode, size, min, max)`, `dirichlet_domain`,
`canonical_points`, `points_in_3x_domain`, `reduced_basis`, `euclidean_mst`.
The curve descriptor (total bar multiplicity alive at each filtration value, per dimension) is
pure Python: `curves_from_barcodes` / `curve_value` in `core.py`, `Periodica.curves()`.
`curves_to_vectors` / `Periodica.curve_vectors(interval, resolution, save_to_file)` sample the
curves into a `(d+1) × resolution` float32 matrix for ML use (saved with `np.save`; the default
interval is the `plot_curves` x-range, `barcode_range(bcodes)`).
`Periodica.save_barcodes(file, meta=None, dtype=np.float64, drop_zero_length=True)` / `load_barcodes(file)`
(`write_barcodes` / `read_barcodes` at module level) persist barcodes as a compressed `.npz` (keys
`format`, `d`, `dims`, `bars`, `meta`); a loaded instance recovers every descriptor without geometry or
merge tree. Zero-length bars are ignored by all descriptors and plot ranges and dropped on save.

## Quotient-complex file format (`examples/*.txt`)

```
dimension:
<d>
lattice:
<d x d basis rows>
vertices:
<n>
<id> <filtration value>          # one per vertex
arcs:
<m>
<u> <v> <filtration> <shift…>    # shift is a d-dimensional integer lattice vector
```

Arcs carry **lattice shift vectors** — an arc from u to v with shift s connects u to the
copy of v translated by U·s. Shift-vector sign conventions are a recurring source of bugs
(see recent commits: "negate shift vector when swapping root").

## Geometry file format (`examples/*_geometry.txt`)

Native input format for a lattice + points (+ optional weights). Loaded/saved by the
web UI (File Load/Save in the input panel, parser in `web/frontend/src/geometry.ts`) and by
`Periodica.load_geometry` / `save_geometry`:

```
geometry                  # header line naming the format
dimension:
<d>                       # 2 or 3
lattice:
<d rows of the matrix U>  # the lattice vectors are the COLUMNS of the block
coordinates:              # optional section; fractional (default) | real
fractional
points:
<n>
<d coords per row, optionally followed by a weight>   # arity uniform per file
```

`#` starts a comment, blank lines are ignored. Fractional coordinates are basis
coefficients (real = U·p).

## Grid file format (`examples/3d_electron_density.txt`)

Native input for a periodic scalar field on a uniform fractional grid, consumed by
`Periodica.load_grid` / `save_grid` → `periodic_grid(U, values, sublevel=True)` (grid
quotient complex with lower-star filtration; `sublevel=False` builds it on the negated
field, i.e. analyzes superlevel sets; same lexical rules and lattice convention as above):

```
grid                      # header line naming the format
dimension:
<d>
lattice:
<d rows of the matrix U>
shape:
<N1 ... Nd>
values:
<one row of the last axis (Nd numbers) per line, C order (first index slowest)>
```

`values[i1,...,id]` is the function value at the fractional grid point (i1/N1, ..., id/Nd).

## Conventions & gotchas

- Points matrices are `d × n` (points as columns), lattice basis `U` is `d × d`.
- Only d = 2 and 3 are supported.
- This is research code: no CI, no linter config. `tests/*.py` are plain-assert scripts
  (`.venv/bin/python tests/<name>.py`). `tests/test_grid_barcode.py` holds golden barcodes of grid
  inputs (matched within 1e-8 in canonical form) plus a gudhi cross-check: any change to
  `merge_tree.cpp` must keep it passing (`--regenerate` rewrites the goldens on purpose only).
  `tests/test_hidden_points.py` covers hidden-point handling in the weighted pipeline.
- Weighted (regular) triangulations can **hide** a point whose power cell is empty
  (`||p−q||² ≤ w_q − w_p`). `periodicDelaunay` drops hidden points, remaps vertex indices, and
  returns the kept original indices as a 4th tuple element; `core.py` warns and exposes
  `hidden_points` / `kept_points`.
- Plots open matplotlib windows (`show=True`); pass `show=False` in headless contexts.
- `bazel-*` symlinks, `build/`, `tmp/`, `__pycache__/`, and the checked-in `.svg`/`.so`
  files are generated artifacts.
- Commit style: short conventional-ish messages, e.g. `fix(merge-tree): …`.
