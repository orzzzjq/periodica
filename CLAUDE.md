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
make            # = requirements + build
```

- Build system is **Bazel with bzlmod** (`MODULE.bazel`); use `bazelisk` if `bazel` is absent.
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
p.plot_barcodes(); p.plot_diagram(); p.plot_images()
p.plot_geometry('delaunay', show=True, slidebar=True)   # interactive filtration slider
# Alternative entry point: p.load_quotient_complex('examples/example_2d_1.txt')
```

Key native functions exposed by `_periodica` (see `periodica/cpp/periodica.cpp` for the full
list): `periodic_delaunay`, `periodic_voronoi`, `full_delaunay`, `full_voronoi`,
`merge_tree(n, d, V, arcs, arc_filtration, arc_shift, vertex_filtration=)`,
`barcode(d, tree)`, `image(barcode, size, min, max)`, `dirichlet_domain`,
`canonical_points`, `points_in_3x_domain`, `reduced_basis`, `euclidean_mst`.

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

## Conventions & gotchas

- Points matrices are `d × n` (points as columns), lattice basis `U` is `d × d`.
- Only d = 2 and 3 are supported.
- This is research code: no CI, no linter config. `tests/test_hidden_points.py` is a plain-assert
  script (`.venv/bin/python tests/test_hidden_points.py`) covering hidden-point handling in the
  weighted pipeline plus basic regressions; otherwise verify changes by running `main.py` examples
  and comparing merge trees / barcodes.
- Weighted (regular) triangulations can **hide** a point whose power cell is empty
  (`||p−q||² ≤ w_q − w_p`). `periodicDelaunay` drops hidden points, remaps vertex indices, and
  returns the kept original indices as a 4th tuple element; `core.py` warns and exposes
  `hidden_points` / `kept_points`.
- Plots open matplotlib windows (`show=True`); pass `show=False` in headless contexts.
- `bazel-*` symlinks, `build/`, `tmp/`, `__pycache__/`, and the checked-in `.svg`/`.so`
  files are generated artifacts.
- Commit style: short conventional-ish messages, e.g. `fix(merge-tree): …`.
