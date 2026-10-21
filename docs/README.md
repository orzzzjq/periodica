# Periodica — Python interface

Periodica computes topological descriptors of **periodic point sets** (crystal structures,
periodic simulations) and of **periodic scalar fields**. Given a lattice basis and the
points of one unit cell, it builds the periodic (weighted) Delaunay or Voronoi *quotient
complex*, computes the **periodic merge tree** of its filtration, and derives descriptors:
signed barcodes, curves, persistence diagrams and persistence images.

The Python package `periodica` has two layers:

| Layer | Where | What |
| --- | --- | --- |
| `periodica.core.Periodica` | `periodica/core.py` | High-level pipeline object: input loading, quotient complexes, merge tree, descriptors, matplotlib plots. |
| `periodica._periodica` | compiled from `periodica/cpp/` | Native functions (CGAL/Eigen): Delaunay/Voronoi geometry, merge tree, barcode, persistence image. Re-exported at package level (`from periodica import merge_tree`). |

Theory: Edelsbrunner & Heiss, *Merge Trees of Periodic Filtrations*
([arXiv:2408.16575](https://arxiv.org/abs/2408.16575)); see also `cecam_poster.pdf`.

---

## Contents

1. [Setup](#setup)
2. [Quick start](#quick-start)
3. [Conventions](#conventions)
4. [The `Periodica` class](#the-periodica-class)
   - [Input](#input)
   - [Quotient complex](#quotient-complex)
   - [Merge tree](#merge-tree)
   - [Descriptors](#descriptors)
   - [Plotting](#plotting)
5. [Module-level helpers](#module-level-helpers-periodicacore)
6. [Native functions](#native-functions-periodica_periodica)
7. [File formats](#file-formats)
8. [Gotchas](#gotchas)

---

## Setup

```sh
make                 # one-time: installs tools, creates .venv, builds periodica/_periodica.so
make build           # after editing any C++ file
```

There is no `pip install`; the package is imported from the source tree. Run scripts from
the repository root with the project venv:

```sh
.venv/bin/python main.py
MPLBACKEND=Agg .venv/bin/python my_script.py     # headless (no plot windows)
```

Runtime dependencies: `numpy`, `scipy`, `matplotlib` (installed into `.venv` by `make`).
Python 3.11; lattice dimension `d` must be 2 or 3.

```python
import numpy as np
from periodica.core import Periodica       # high-level API
from periodica import _periodica           # native functions (also: from periodica import merge_tree, ...)
```

---

## Quick start

One point per unit cell of the square lattice:

```python
import numpy as np
from periodica.core import Periodica

p = Periodica()
p.set_geometry({
    'd': 2,
    'U': np.eye(2),                        # lattice vectors are the COLUMNS of U
    'n_points': 1,
    'points': np.array([[0.5, 0.5]]).T,    # shape (d, n_points): points are columns
})
p.quotient_complex('delaunay')             # or 'voronoi'
p.merge_tree()
p.print_merge_tree()
print(p.barcodes())

p.plot_all_descriptors(show=True)          # barcode | curve | diagram | image, one row per dimension
p.plot_geometry('delaunay', show=True, slidebar=True)   # Delaunay in the 3x domain + filtration slider
```

Output:

```
** Periodic merge tree **
0:
 time: -0.000, monomial: 1.000 piR^2
 time: 0.250, monomial: 2.000 R^1
 time: 0.250, monomial: 1.000 R^0
[[(-0.0, 0.25, -1.0), (-0.0, inf, 1.0)],
 [(-0.0, 0.25, -1.0), (-0.0, 0.25, 1.0)],
 [(-0.0, 0.25, 1.0)]]
```

Reading: below filtration value 0.25 every point is its own component, and a ball of
radius *R* contains about `1·πR²` of them. At 0.25 (edges of length 1 enter, alpha value
`(1/2)² = 0.25`) the points connect along both lattice directions and a single unbounded
component remains (`1·R⁰`). The barcodes encode the same information per power of *R*
(see [Descriptors](#descriptors)).

Other entry points, same downstream calls:

```python
p = Periodica(); p.load_geometry('examples/2d_geometry.txt')            # lattice + points (+ weights)
p = Periodica(); p.load_quotient_complex('examples/2d_quotient_complex_1.txt')   # a ready-made complex
p = Periodica(); p.load_grid('examples/2d_periodic_function.txt', sublevel=False) # periodic scalar field
```

---

## Conventions

**Shapes.** The lattice basis `U` is `d × d` with the lattice vectors as **columns**.
Point matrices are `d × n` with points as **columns**. Arc lists are `m × 2` integer arrays;
shift matrices are `d × m` (one shift column per arc). Weights are length-`n` vectors.

**Coordinates.** `set_geometry` takes real (Cartesian) coordinates. Geometry files may use
fractional coordinates; `load_geometry` converts them with `U @ p`.

**Filtration scale (power distance).** All filtration values of the point-set pipeline live
on the scale of the power distance `f_i(x) = ‖x − p_i‖² − w_i`:

- a vertex (point) enters at `−w_i` (0 for unweighted points),
- a Delaunay arc enters at its weighted alpha value, the squared radius of its smallest
  orthogonal ball (for an unweighted Gabriel edge of length `ℓ`, that is `(ℓ/2)²`),
- the sublevel set at value `f` is the union of balls of radius `sqrt(f + w_i)`, which is
  what the filtration slider in `plot_delaunay` draws.

The Voronoi quotient complex is filtered by the **negated** alpha values of the dual
Delaunay simplices (a Voronoi vertex enters at `−alpha(dual top simplex)`, a Voronoi arc at
`−alpha(dual facet)`), so its merge tree describes the empty space between the balls as
it shrinks.

**Weighted points.** A point whose power cell is empty (`‖p − q‖² ≤ w_q − w_p` for some `q`)
is *hidden*: it is not a vertex of the regular triangulation. `periodic_delaunay` drops such
points, remaps arc endpoints, prints a `[WARNING]`, and records them in `hidden_points`.

**Quotient complex.** The periodic complex modulo lattice translations. Vertices are the
points of one unit cell (after hiding), arcs carry an integer **shift vector** `s`: the arc
`(u, v, s)` connects `u` to the copy of `v` translated by `V @ s`, where the columns of `V`
(`Periodica.V`) are the lattice vectors the shifts refer to. For the Delaunay/Voronoi
pipeline `V` is the *reduced* basis (`_periodica.reduced_basis(U)`, an obtuse superbase with
`d + 1` columns; only the first `d` are used), for grids and quotient files it is `U` / the
file's lattice block.

**Periodic merge tree.** Each vertex orbit starts a *beam*. As the filtration value grows,
components merge with each other and with their own lattice translates. The tree records, for
every stage of a component, how many of its copies meet a ball of radius *R*, as a monomial
`c · κ_k R^k`: a bounded component replicates like volume (`k = d`), a component that wraps
around `j` independent lattice directions grows like `R^(d−j)`, and a component wrapping in
all `d` directions is unique (`k = 0`). `κ_k` is the volume of the unit `k`-ball
(`1, 2, π, 4π/3`), shown in the printed monomials; coefficients of the `R^d` term are
`1/|det U|` per bounded component.

**Barcodes.** `barcodes()` returns `d + 1` lists, one per power of *R*: `bcodes[k]` holds the
signed bars `(birth, death, multiplicity)` of the `R^k` coefficient, `death = inf` for
infinite bars. Multiplicities can be negative and fractional: the *sum of multiplicities of
the bars alive at `f`* (bars are alive on `[birth, death)`) is the `R^k` coefficient at `f`.
That running sum is the **curve** descriptor (`curves()`).

---

## The `Periodica` class

`Periodica()` takes no arguments. State lives in attributes that the methods set; later
stages call earlier ones on demand (`merge_tree()` builds the Delaunay quotient complex if
none exists, `barcodes()` builds the tree, `images()`/`curves()`/plots build the barcodes).

| Stage | Methods | Attributes set |
| --- | --- | --- |
| Input | `set_geometry`, `set_weights`, `load_geometry`, `save_geometry` | `d`, `U`, `n_points`, `input_points`, `points`, `weights` |
| Quotient complex | `quotient_complex`, `periodic_delaunay`, `periodic_voronoi`, `periodic_grid`, `load_grid`, `save_grid`, `load_quotient_complex` | `V`, `n_quotient_vertices`, `quotient_arcs`, `quotient_arc_filtration`, `quotient_arc_shift`, `quotient_vertex_filtration`, (`kept_points`, `hidden_points`), (`grid_values`, `grid_directions`) |
| Merge tree | `merge_tree`, `print_merge_tree` | `tree` |
| Descriptors | `barcodes`, `curves`, `curve_vectors`, `images`, `save_barcodes`, `load_barcodes` | `bcodes`, `filtration_curves`, `persistence_images`, (`barcode_meta`) |
| Plots | `plot_barcodes`, `plot_curves`, `plot_diagram`, `plot_images`, `plot_all_descriptors`, `plot_delaunay`, `plot_voronoi`, `plot_geometry` | — |

The methods `periodic_delaunay`, `periodic_voronoi`, `periodic_grid` and `merge_tree` print a
`[TIME] <name> took: … sec` line.

### Input

#### `set_geometry(INPUT)`

`INPUT` is a dict:

| key | type | meaning |
| --- | --- | --- |
| `'d'` | int | 2 or 3 |
| `'U'` | `(d, d)` float array | lattice basis, vectors as columns |
| `'n_points'` | int | number of points in the unit cell |
| `'points'` | `(d, n_points)` array | real coordinates, points as columns |
| `'weights'` | `(n_points,)` array, optional | weights `w_i` for the power (regular) triangulation |

The raw points are kept in `input_points`. The points actually used (`points`) get a tiny,
seeded Gaussian perturbation (`Periodica.PERTURBATION = 1e-9`) that stands in for symbolic
perturbation: exactly degenerate inputs (cocircular/cospherical tuples, as in most ideal
crystals) would otherwise be triangulated inconsistently across lattice copies. Set
`Periodica.PERTURBATION = 0.0` before `set_geometry` to disable it. Points may lie outside
the unit cell; they are canonicalized internally.

#### `set_weights(weights)`

Replace the weights (length `n_points`). Call before (re)building the quotient complex.

#### `load_geometry(file)` / `save_geometry(file, coordinates='real')`

Read/write the [geometry file format](#geometry-file). Loading converts fractional
coordinates to real ones and calls `set_geometry` (weights default to 0). Saving writes
`input_points` (unperturbed); `coordinates='fractional'` converts through `U` first. The
weight column is written only if some weight is non-zero.

#### `load_quotient_complex(file)`

Read a [quotient-complex file](#quotient-complex-file) directly into the quotient-complex
attributes (`V`, `n_quotient_vertices`, `quotient_vertex_filtration`, `quotient_arcs`,
`quotient_arc_filtration`, `quotient_arc_shift`). No point geometry is involved, so only
the merge tree, descriptors and descriptor plots are available afterwards.

#### `periodic_grid(U, values, sublevel=True)` / `load_grid(file, sublevel=True)` / `save_grid(file)`

Quotient complex of a **periodic scalar field** sampled on a uniform fractional grid.
`values` is a `(N1, …, Nd)` array; `values[i1, …, id]` is the value at the fractional grid
point `(i1/N1, …, id/Nd)` of the lattice `U`. Every grid point becomes a vertex (filtration =
its value); it is connected to its neighbours along the `d` shortest linearly independent
directions of the fine grid lattice `U · diag(1/N)` (for skewed cells these need not be the
grid axes; the chosen integer directions are stored in `grid_directions`, `d × d`). Arcs that
wrap around the cell carry the corresponding lattice shift. Arc filtration is lower-star
(the max of the endpoint values).

`sublevel=False` builds the complex on the **negated** field, so the merge tree analyzes
superlevel sets (maxima of the field, e.g. charge density peaks). `grid_values` keeps the raw
input either way, `save_grid` writes it back in the [grid file format](#grid-file).
`periodic_grid` also deletes stale `tree`/`bcodes`/`filtration_curves`/`persistence_images`
from a previous run on the same instance.

```python
g = Periodica()
g.load_grid('examples/3d_electron_density.txt', sublevel=False)   # 48x48x28 grid → 64512 vertices
g.merge_tree()                      # one beam per grid point: do not print_merge_tree()
g.plot_all_descriptors(show=True, same_range=False)
```

For large grids `periodic_grid(U, values, sublevel, build_arcs=False)` skips the explicit arc
arrays (three arcs per grid point, needed only by the plots and the web UI; the merge tree
generates them on the fly) and `barcodes(drop_zero_length=True)` leaves out the one
zero-length bar per non-critical grid point. An 80³ grid then takes well under a second.

### Quotient complex

#### `quotient_complex(complex_type='delaunay')`

Dispatches to `periodic_delaunay()` or `periodic_voronoi()`; any other string raises.

#### `periodic_delaunay()`

Builds the periodic (weighted) Delaunay quotient complex of `points`/`weights` in the lattice `U`
through the native `periodic_delaunay`. Sets

| attribute | shape | meaning |
| --- | --- | --- |
| `V` | `(d, d+1)` | reduced lattice basis the shift vectors refer to |
| `kept_points` | `(n_kept,)` int | original indices of the points that survived hiding |
| `hidden_points` | list | original indices that were hidden (a `[WARNING]` is printed if non-empty) |
| `n_quotient_vertices` | int | `= len(kept_points)` |
| `quotient_vertex_filtration` | `(n_kept,)` | `-weights[kept_points]` |
| `quotient_arcs` | `(m, 2)` int | arc endpoints, indexing the *kept* points |
| `quotient_arc_filtration` | `(m,)` | weighted alpha values |
| `quotient_arc_shift` | `(d, m)` int | lattice shift per arc |

Requires `weights` to have length `n_points` (defaults to zeros).

#### `periodic_voronoi()`

Builds the periodic (power) Voronoi quotient complex: vertices are the Voronoi vertices of one
unit cell (dual to the Delaunay top simplices), arcs are Voronoi edges (dual to Delaunay
facets). Sets the same attributes as above except `kept_points`/`hidden_points`, with
`quotient_vertex_filtration` and `quotient_arc_filtration` equal to the **negated** alpha
values of the dual simplices.

### Merge tree

#### `merge_tree()` → `tree`

Calls the native `merge_tree` on the quotient complex (building the Delaunay complex first
if none exists); for a grid complex it calls `grid_merge_tree`, which generates the arcs on
the fly from the field and the grid directions. `tree` is a native `MergeTree` object that
behaves like a list with one entry per **beam** (one beam per quotient vertex, in vertex
order): `len(tree)`, `tree[i]`, iteration, and `tree.to_list()` for a plain list. Each beam
is a time-ordered list of events

```
(time, coefficient, exponent, child)
```

| field | meaning |
| --- | --- |
| `time` | filtration value of the event |
| `coefficient`, `exponent` | the beam's monomial after the event is `coefficient · κ_exponent · R^exponent` (`κ_k` = unit `k`-ball volume) |
| `child == -1` | birth of the beam, or a merge of the component with its own lattice translate (exponent drops) |
| `child == j`, `j ≠` beam index | beam `j` merged into this beam at `time` (this beam continues) |
| `child ==` beam index | final event of the beam: it merges into another beam |

The native `merge_tree` validates the input and raises `ValueError` if an arc has a smaller
filtration value than one of its endpoints, if shapes disagree, or if `V` is not `d × (≥ d)`.

#### `print_merge_tree()`

Human-readable dump of `tree` (written by C++ `printf`, so it can appear out of order with
Python `print` output unless Python runs unbuffered). For `examples/2d_quotient_complex_1.txt`:

```
** Periodic merge tree **
0:
 time: 1.000, monomial: 1.000 piR^2
 time: 5.000, monomial: 1.000 piR^2 <- child: 1
 time: 7.000, monomial: 2.828 R^1
 time: 9.000, monomial: 1.000 R^0
1:
 time: 3.000, monomial: 1.000 piR^2
 time: 5.000, monomial: 1.000 piR^2 - merge ->
```

and the corresponding `tree`:

```python
[[(1.0, 1.0, 2, -1), (5.0, 1.0, 2, 1), (7.0, 1.4142135623730951, 1, -1), (9.0, 1.0, 0, -1)],
 [(3.0, 1.0, 2, -1), (5.0, 1.0, 2, 1)]]
```

Vertex 1 (born at 3) merges into vertex 0's component at 5. At 7 the merged component meets
its own translate along the diagonal shift `(1, 1)` and becomes an infinite strip; parallel
strips repeat with spacing `1/√2`, so about `2R / (1/√2) = 2√2 R ≈ 2.828 R` of them cross a
ball of radius *R*. At 9 the strips connect across the second direction and a single
component remains.

### Descriptors

#### `barcodes()` → `bcodes`

`list` of `d + 1` lists of `(birth, death, multiplicity)` tuples (`death = inf` for infinite
bars, multiplicities signed), `bcodes[k]` for the `R^k` coefficient. See
[Conventions](#conventions). For the quick-start square lattice:

```python
bcodes[2] == [(0.0, 0.25, 1.0)]                       # πR² term: 1 on [0, 0.25)
bcodes[1] == [(0.0, 0.25, -1.0), (0.0, 0.25, 1.0)]    # R¹ term: cancels (both merges at 0.25)
bcodes[0] == [(0.0, 0.25, -1.0), (0.0, inf, 1.0)]     # R⁰ term: 0 on [0, 0.25), 1 afterwards
```

#### `curves()` → `filtration_curves`

One `(times, values)` pair per dimension (`curves_from_barcodes(bcodes)`): `times` are the
distinct event times in increasing order and `values[i]` is the total multiplicity alive on
`[times[i], times[i+1])` (0 before `times[0]`, `values[-1]` after the last time), i.e. the
`R^k` coefficient as a right-continuous step function. Evaluate with `curve_value`.

#### `curve_vectors(interval=None, resolution=100, save_to_file=None)` → `(d+1, resolution)` array

The curves as fixed-length vectors for machine learning
(`curves_to_vectors(filtration_curves, …)`): row `k` is the `R^k` curve evaluated at the
`resolution` points of `np.linspace(xmin, xmax, resolution)`, endpoints included.
`interval=(xmin, xmax)` defaults to the x-range of `plot_curves`: from the smallest birth
to the largest finite endpoint over the bars of positive length, padded by 12 % below and
5 % above (`barcode_range(bcodes)`; the module-level `curves_to_vectors` computes the same
range from the curves' event times, `curve_range(curves)`). Values are `float32`.
`save_to_file='path.npy'` writes the matrix with `np.save` (numpy appends `.npy` if the path
has no extension); read it back with `np.load`. When building a dataset, pass the same
explicit `interval` and `resolution` for every structure so the vectors are comparable.

```python
vec = p.curve_vectors(interval=(-0.1, 0.5), resolution=64, save_to_file='out/square.npy')
vec.shape, vec.dtype          # ((3, 64), dtype('float32')) for d = 2
np.load('out/square.npy')     # same matrix
```

#### `images(size=100)` → `persistence_images`

One `size × size` `numpy` array per dimension. All `d + 1` images share the same square
range `barcode_range(bcodes, (0.12, 0.12))` on both axes: the smallest birth and largest
finite endpoint over the bars of positive length (an infinite bar counts with its birth),
padded by 12 % of that span. Each bar of positive finite length deposits a Gaussian bump
(GUDHI persistence-image code, discrete 11×11 Gaussian filter) at `(birth, death)` scaled
by its **signed multiplicity**, so images can be negative. Infinite bars and zero-length
bars are excluded. Rows index the death axis, columns the birth axis (plot with
`origin='lower'`).

#### `save_barcodes(file, meta=None, dtype=np.float64, drop_zero_length=True)` / `load_barcodes(file)`

Persist the barcodes so the descriptors can be recovered later without recomputing the
quotient complex and the merge tree. `save_barcodes` writes a compressed `.npz`
([format below](#barcode-file)); `meta` is an optional JSON-serializable dict stored with
it. By default bars with `death == birth` are dropped: they carry no persistence, no
descriptor or plot range depends on them, and in grid barcodes they are the vast majority
(one per non-critical grid point; the 3D electron-density example shrinks from 139 KB to
1.3 KB). `dtype=np.float32` halves the bar storage at about 1e-7 relative error; float64
round-trips exactly.

`load_barcodes` sets `d`, `bcodes` and `barcode_meta` on the instance and discards any
quotient complex, merge tree and derived descriptors, so `barcodes()` returns the loaded
bars and `curves()`, `curve_vectors()`, `images()` and the descriptor plots give the same
results as on the original instance. `merge_tree()`, `print_merge_tree()` and the geometry
plots need the original input and raise. Load into a fresh instance unless you mean to
replace the current state.

```python
p.save_barcodes('out/structure.npz', meta={'source': 'examples/3d_geometry.txt', 'complex': 'voronoi'})

q = Periodica()
q.load_barcodes('out/structure.npz')          # sets q.d, q.bcodes, q.barcode_meta
q.curve_vectors(interval=(-1.0, 0.5), resolution=64)
q.plot_all_descriptors(show=True)

d, bcodes, meta = read_barcodes('out/structure.npz')   # without a Periodica instance
```

### Plotting

All plotting functions use matplotlib. `show=True` blocks on `plt.show()`; pass
`show=False` (and set `MPLBACKEND=Agg`) in headless contexts. Descriptor plots have one
panel per dimension, **top panel = dimension `d`, bottom = dimension 0**, labelled
`·R⁰`, `·2R¹`, `·πR²`, `·(4π/3)R³`. Bars/points are annotated with their multiplicity.

| method | draws | notes |
| --- | --- | --- |
| `plot_barcodes(show=True, ax=None)` | bars of `bcodes[k]`, infinite bars run to the right edge | saves `barcode.svg` |
| `plot_curves(show=True, ax=None)` | step plot of `curves()`, x-range aligned with the barcode | saves `curve.svg` |
| `plot_diagram(show=True, ax=None)` | (birth, death) scatter, infinite deaths drawn at the top edge | saves `diagram.svg` |
| `plot_images(same_range=True, show=True, ax=None)` | `images()` with a `Spectral_r` colormap symmetric around 0; `same_range` shares one colour scale across dimensions | saves `image.svg` |
| `plot_all_descriptors(same_range=True, show=True)` | the four above side by side (`(d+1) × 4` grid) | saves `all_descriptors.svg` |

`ax` lets you draw into your own axes: pass an array of `d + 1` axes (index 0 = dimension
`d`). When `ax` is given, nothing is shown or saved. Without `ax` the SVG is written to the
current working directory after the window closes.

Geometry plots (point-set inputs only):

| method | draws |
| --- | --- |
| `plot_delaunay(show=True, animation_gif=None, ax=None, slidebar=False)` | reduced basis vectors, Dirichlet domain and its 3× dilation, the full Delaunay skeleton of the 3× domain (dashed) with the quotient arcs highlighted (blue). 2D: `slidebar=True` adds a filtration slider that grows the balls of radius `sqrt(f + w_i)`. 3D: `animation_gif='out.gif'` saves a rotating view (needs ImageMagick). Re-runs `periodic_delaunay()`. |
| `plot_voronoi(show=True, animation_gif=None, use_circumcenter=False, ax=None, slidebar=False)` | everything from `plot_delaunay` plus the full Voronoi skeleton (dash-dot) and the periodic Voronoi arcs (red). `use_circumcenter=True` places Voronoi vertices at the power centers of the dual simplices (the true Voronoi vertices; plain circumcenters when unweighted), `False` at their centroids. Positions only; filtration values are unaffected. |
| `plot_geometry(TYPE, show=False, slidebar=False, use_circumcenter=True)` | `TYPE == 'delaunay'` → `plot_delaunay`, otherwise `plot_voronoi`. Note the default `show=False`. |

---

## Module-level helpers (`periodica.core`)

| function | description |
| --- | --- |
| `curves_from_barcodes(barcodes)` | `[(times, values), …]`, one per barcode: running sum of signed multiplicities (bars alive on `[birth, death)`, zero-length bars skipped). |
| `curve_value(times, values, t)` | Value of such a curve at `t` (array-friendly). |
| `barcode_span(barcodes)` | `(lo, hi)`: smallest birth and largest finite endpoint over the bars of positive length (infinite bars count with their birth). Zero-length bars are ignored here and everywhere downstream; raises if no bar has positive length. |
| `barcode_range(barcodes, pad=(0.12, 0.05))` | `barcode_span` padded by the given fractions of the span: the x-range of the descriptor plots (`(0.12, 0.05)` for barcode and curve, `(0.12, 0.12)` for diagram and image) and the default interval of `Periodica.curve_vectors`. |
| `curve_range(curves, pad=(0.12, 0.05))` | The same range computed from the curves' event times. Default interval of `curves_to_vectors`. |
| `write_barcodes(file, d, barcodes, meta=None, dtype=np.float64, drop_zero_length=True)` | Write barcodes to a compressed `.npz` ([Barcode file](#barcode-file)). |
| `read_barcodes(file)` | `(d, barcodes, meta)` from such a file, the barcodes in the form `Periodica.barcodes()` produces; `ValueError` on a malformed file. |
| `curves_to_vectors(curves, interval=None, resolution=100, save_to_file=None, dtype=np.float32)` | Uniform samples of each curve on `interval` as a `(len(curves), resolution)` array; optionally saved with `np.save`. |
| `power_grid_shape(U, budget=160_000, n_min=8, n_max=96)` | Grid shape `(N1, …, Nd)` for sampling a field on the cell of `U` with roughly isotropic spacing and about `budget` samples in total (each axis clipped to `[n_min, n_max]`). |
| `sample_power_field(U, points, weights, shape)` | Samples the periodic power distance `min_{i,z} ‖x − (p_i + U z)‖² − w_i` on the fractional grid of `U`, returned as a `shape` array (C order, same layout as `periodic_grid` values). Exact for arbitrary lattices (candidate shifts enumerated in the 3× Dirichlet domain of the reduced basis). |
| `grid_anchor_indices(U, shape, positions)` | Ravelled (C-order) indices of the fractional grid points nearest to the real positions `(d, m)`; positions may lie outside the cell. |

The grid field of a point set can thus be fed back through `periodic_grid` as a sampled
alternative to the exact Delaunay filtration:

```python
U, P, w = p.U, p.points, p.weights
shape = power_grid_shape(U)
q = Periodica()
q.periodic_grid(U, sample_power_field(U, P, w, shape))
q.merge_tree(); q.barcodes()
```

`_parse_geometry_text` / `_parse_grid_text` (and `_SectionParser`) are the parsers behind
`load_geometry` / `load_grid`; they are private but usable on in-memory strings.

---

## Native functions (`periodica._periodica`)

All matrices are `numpy` arrays (Eigen on the C++ side): float for coordinates, int for
indices and shifts. Invalid shapes raise `ValueError`.

| function | returns |
| --- | --- |
| `reduced_basis(U)` | `(d, d+1)` obtuse superbase `V` (columns sum to zero); the first `d` columns are a reduced basis of `U`. |
| `reduced_basis_coeffs(U)` | `(V, T)` with `V = U @ T`, `T` integer `(d, d+1)`. |
| `dirichlet_domain(V)` | `(A, b)`: the Dirichlet (Voronoi) domain of the lattice is `{x : A x ≤ b}`. |
| `canonical_points(A, b, points)` | The lattice translate of each point lying in the Dirichlet domain, `(d, n)`. |
| `points_in_3x_domain(V, A, b, canonical_points)` | `(P, I, S)`: all lattice copies of the canonical points inside the 3× dilated domain, `P` `(d, m)`, `I` `(m,)` original index of each copy, `S` `(d, m)` the lattice shift of each copy. |
| `periodic_delaunay(U, points, weights)` | `(arcs, filtration, shift, kept)` of the Delaunay quotient complex: `arcs` `(m, 2)` indexing the kept points, `filtration` `(m,)`, `shift` `(d, m)`, `kept` `(n_kept,)` original indices. |
| `periodic_voronoi(U, points, weights, useCircumCenter=False)` | `(voronoi_points, arcs, point_filtration, arc_filtration, shift)`: Voronoi vertices of one cell `(d, l)`, arcs `(m, 2)`, negated alpha values of vertices and arcs, shifts `(d, m)`. |
| `full_delaunay(U, points, weights)` | `(P, edges)`: points of the 3× domain (same order as `points_in_3x_domain`) and the edges `(m, 2)` of their weighted Delaunay triangulation. |
| `full_voronoi(U, points, weights, useCircumCenter=False)` | `(voronoi_points, edges)`: the dual Voronoi skeleton of that triangulation. |
| `merge_tree(n, d, V, arcs, arc_filtration, arc_shift, vertex_filtration=empty)` | The periodic merge tree (a `MergeTree`, list-like over beams, see [Merge tree](#merge-tree)). `V` is `(d, ≥d)`; its first `d` columns define the lattice (shift `s` ↦ `V[:, :d] @ s`) and `1/|det|` scales the `R^d` multiplicities. An empty `vertex_filtration` means all zeros. |
| `grid_merge_tree(d, U, shape, f, directions)` | The merge tree of the lower-star filtration of a periodic grid without an arc list: `f` is the field flattened in C order over `shape` (negated for superlevel sets), `directions` `(d, d)` the integer grid steps (columns) joining neighbors, as `Periodica.grid_directions`. |
| `MergeTree` | The tree object: `d`, `n`, `len`, indexing, iteration, `to_list()`, `MergeTree.from_list(list)`. |
| `print_merge_tree(tree)` | Prints a tree (`MergeTree` or list form). |
| `barcode(d, tree, drop_zero_length=False)` | `d + 1` lists of `(birth, death, multiplicity)`; `drop_zero_length` leaves out bars with `death == birth`. |
| `image(barcode, size, min, max)` | `(size, size)` persistence image of one barcode over `[min, max]²`. |
| `delaunay_skeleton(points)` / `weighted_delaunay_skeleton(points, weights)` | Edges `(m, 2)` of the (weighted) Delaunay triangulation of a finite point set (not periodic). |
| `euclidean_mst(points)` | Edges `(n−1, 2)` of the Euclidean minimum spanning tree of a finite point set. |

Example: reproduce what `plot_delaunay` computes, without matplotlib.

```python
import numpy as np
from periodica import _periodica

U = np.eye(2); P = np.array([[0.25, 0.7, 0.0], [0.15, 0.25, 0.75]]); w = np.zeros(3)
V = _periodica.reduced_basis(U)
A, b = _periodica.dirichlet_domain(V)
canonical = _periodica.canonical_points(A, b, P)
P3, I, S = _periodica.points_in_3x_domain(V, A, b, canonical)   # copies, their source index, their shift
P_full, edges = _periodica.full_delaunay(U, P, w)                # Delaunay of those copies
arcs, f, shift, kept = _periodica.periodic_delaunay(U, P, w)     # the quotient complex
tree = _periodica.merge_tree(len(kept), 2, V, arcs, f, shift, -w[kept])
bars = _periodica.barcode(2, tree)
```

---

## File formats

`#` starts a comment and blank lines are ignored in the geometry and grid formats; tokens are
whitespace-separated and sections appear in a fixed order. In every format the lattice block
is the matrix printed row by row, so **the lattice vectors are its columns**. Examples live in
`examples/`.

### Geometry file

Lattice + points (+ optional weights). `Periodica.load_geometry` / `save_geometry`; also the
Load/Save format of the web UI.

```
geometry                  # header naming the format
dimension:
2
lattice:
1 0                       # matrix U row by row (columns = lattice vectors)
0 1
coordinates:              # optional section: fractional (default) | real
fractional
points:
3                         # count, then one point per line: d coords [+ weight], arity uniform per file
0.25 0.15 0.03
0.7  0.25 0.004
0    0.75 0
```

### Grid file

Periodic scalar field on a uniform fractional grid. `Periodica.load_grid` / `save_grid`.

```
grid
dimension:
2
lattice:
1 0
0 1
shape:
20 20                     # N1 ... Nd
values:                   # prod(N1..N_{d-1}) rows of Nd numbers, C order (first index slowest)
0.397004 0.362832 ... (20 numbers)
...
```

`values[i1, …, id]` is the value at the fractional grid point `(i1/N1, …, id/Nd)`.

### Quotient-complex file

A quotient complex given directly. `Periodica.load_quotient_complex`. This older format has
a stricter reader: header lines are skipped verbatim (so trailing comments on them are
tolerated), data lines must contain exactly the listed numbers separated by single spaces,
and no comments or blank lines are allowed.

```
dimension:
2
lattice:
1 0                       # matrix V row by row; a shift s means the translation V @ s
0 1
vertices:
2                         # count, then: <id> <filtration value>
0 1.0
1 3.0
arcs:
3                         # count, then: <u> <v> <filtration> <shift_1 ... shift_d>
0 1 5.0 0 0
1 0 9.0 1 0
1 0 7.0 1 1
```

### Barcode file

Compressed numpy archive (`.npz`) written by `save_barcodes` / `write_barcodes` and read by
`load_barcodes` / `read_barcodes`, or by plain `np.load`:

| key | content |
| --- | --- |
| `format` | the string `periodica-barcodes-1` |
| `d` | lattice dimension, 2 or 3 |
| `dims` | `(n,)` uint8, the dimension `k` (power of *R*) of each bar |
| `bars` | `(n, 3)` float64 (or float32) rows `birth, death, multiplicity`; infinite deaths are stored as `inf` |
| `meta` | JSON string, `{}` unless `meta=` was given |

Bars are concatenated over dimensions in increasing `k`, so empty dimensions are
representable. Zero-length bars are omitted unless `drop_zero_length=False`.

---

## Gotchas

- **Column-major everywhere.** `points` is `(d, n)`, `U`'s columns are the lattice vectors,
  shift matrices are `(d, m)`. Passing `(n, d)` points gives shape errors or wrong results.
- **Stale results on reuse.** `barcodes()`, `images()`, `curves()` and the plots only compute
  what is missing. After calling `set_geometry`/`set_weights`/`quotient_complex` again on the
  same instance, call `merge_tree()` (and `barcodes()` …) explicitly, or use a fresh
  `Periodica()`. Only `periodic_grid` and `load_barcodes` clear downstream results themselves.
- **Hidden points shift indices.** With weights, `quotient_arcs` index the kept points; map
  back with `kept_points[idx]`.
- **Degenerate inputs** rely on the built-in perturbation (`PERTURBATION = 1e-9`); if you
  disable it, exactly cocircular/cospherical configurations can yield an inconsistent
  quotient complex.
- **Voronoi filtration values are negative** (negated alpha values). Individual barcodes can
  be empty (e.g. dimension 0 of some Voronoi complexes); the descriptor plots and `images()`
  skip empty ones when computing their ranges, and raise if *every* barcode is empty.
- **Zero-length bars** (`death == birth`) are ignored by every descriptor and by the plot
  ranges, and `save_barcodes` drops them by default. They still sit in `bcodes` and appear as
  diagonal points of `plot_diagram` until a barcode file is reloaded.
- **Plots write SVGs** (`barcode.svg`, `curve.svg`, `diagram.svg`, `image.svg`,
  `all_descriptors.svg`) into the current working directory whenever they own the figure.
- **Mixed stdout.** `print_merge_tree` and the `[WARNING]`/`[TIME]` lines come from different
  buffers (C `printf` vs Python); run with `python -u` if the order matters in a log.
- **Rebuild after C++ changes** (`make build`); Python imports the copied
  `periodica/_periodica.so`, not the Bazel output.
- **Large grids.** The merge tree has one beam per grid point; `print_merge_tree()` on a
  3D grid prints tens of thousands of beams. Downsample the field before loading if needed.
