"""Plain-assert tests for the barcode of grid inputs (periodic cubical complex
-> periodic merge tree -> signed barcode), matched within EPS.

Run with:   .venv/bin/python tests/test_grid_barcode.py
            .venv/bin/python tests/test_grid_barcode.py --regenerate   # rewrite goldens

1. Golden barcodes: every case in CASES (example grid files, the seeded random
   grids and the mp-1 charge density in tests/data/grid_inputs) is compared to
   tests/data/grid_barcodes/<name>.npz, bar by bar, within EPS, through both
   backends: grid_merge_tree (arcs generated on the fly, what merge_tree()
   uses for grids) and the generic merge_tree on the explicit arc arrays.
   This is the regression harness for the merge tree backend: an optimized
   backend must reproduce these files. Barcodes are compared in CANONICAL form (see
   canonical): zero-length bars dropped, bars with equal (birth, death) summed,
   bars with zero total multiplicity dropped. The signed barcode of a component
   whose periodicity rank jumps by more than one at a single filtration value
   contains intermediate +/- bars that cancel, and their split depends on the
   order of equal-time arcs, so only the canonical form is well defined.
2. Arc order invariance: a shuffled quotient complex gives the same canonical
   barcode.
3. Lattice: times are combinatorial (they do not depend on U when the grid
   directions are the axes); multiplicities scale with the lattice volumes.
4. gudhi cross-check in 3D (identity and skewed lattice): merger times match
   the finite dim-0 bars of gudhi's periodic cubical complex, wrap times match
   the births of its infinite dim-1 bars.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from periodica import _periodica  # noqa: E402
from periodica.core import Periodica, read_barcodes, write_barcodes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / 'tests/data/grid_inputs'
GOLDEN = ROOT / 'tests/data/grid_barcodes'
EPS = 1e-8
inf = np.inf


# ---- helpers ----

def canonical(bcodes, eps=EPS):
    """Per dimension: sorted (birth, death, multiplicity) with zero-length bars
    dropped, bars with equal (birth, death) (within eps) merged by summing the
    multiplicities, and bars of zero total multiplicity dropped."""
    out = []
    for bars in bcodes:
        rows = sorted((float(b), float(d), float(m)) for b, d, m in bars
                      if not (np.isfinite(d) and abs(d - b) <= eps))
        merged = []
        for b, d, m in rows:
            if merged and abs(merged[-1][0] - b) <= eps and (
                    (np.isinf(d) and np.isinf(merged[-1][1])) or abs(merged[-1][1] - d) <= eps):
                merged[-1][2] += m
            else:
                merged.append([b, d, m])
        out.append([(b, d, m) for b, d, m in merged if abs(m) > eps])
    return out


def assert_match(got, ref, eps=EPS, label=''):
    """Canonical barcodes equal within eps (times absolute, multiplicities absolute)."""
    got, ref = canonical(got, eps), canonical(ref, eps)
    assert len(got) == len(ref), f'{label}: {len(got)} vs {len(ref)} dimensions'
    for k, (g, r) in enumerate(zip(got, ref)):
        assert len(g) == len(r), f'{label} dim {k}: {len(g)} bars vs {len(r)} expected'
        for (gb, gd, gm), (rb, rd, rm) in zip(g, r):
            ok = abs(gb - rb) <= eps and abs(gm - rm) <= eps and (
                (np.isinf(gd) and np.isinf(rd)) or abs(gd - rd) <= eps)
            assert ok, f'{label} dim {k}: got {(gb, gd, gm)} expected {(rb, rd, rm)}'


def grid(U, values, sublevel=True):
    p = Periodica()
    p.periodic_grid(np.asarray(U, dtype=float), np.asarray(values, dtype=float), sublevel=sublevel)
    return p


def from_input_file(name):
    z = np.load(INPUTS / f'{name}.npz')
    return grid(z['U'], z['values'], bool(z['sublevel']))


def from_grid_file(file, sublevel):
    p = Periodica()
    p.load_grid(ROOT / file, sublevel=sublevel)
    return p


# name -> builder of a Periodica instance holding the grid quotient complex
CASES = {
    '2d_function_sublevel': lambda: from_grid_file('examples/2d_periodic_function.txt', True),
    '2d_function_superlevel': lambda: from_grid_file('examples/2d_periodic_function.txt', False),
    '3d_density_superlevel': lambda: from_grid_file('examples/3d_electron_density.txt', False),
    '2d_random_skew': lambda: from_input_file('2d_random_skew'),
    '3d_random_skew': lambda: from_input_file('3d_random_skew'),
    '3d_random_small_N': lambda: from_input_file('3d_random_small_N'),
    'mp-1_s2': lambda: from_input_file('mp-1_s2'),
}


def compute(name):
    p = CASES[name]()
    p.merge_tree()
    return p, p.barcodes()


def generic_barcodes(p, arcs=None, arc_filtration=None, arc_shift=None):
    """Barcodes through the generic backend on explicit arcs (by default the
    arrays periodic_grid built), bypassing the grid-native path."""
    arcs = p.quotient_arcs if arcs is None else arcs
    arc_filtration = p.quotient_arc_filtration if arc_filtration is None else arc_filtration
    arc_shift = p.quotient_arc_shift if arc_shift is None else arc_shift
    tree = _periodica.merge_tree(p.n_quotient_vertices, p.d, p.V, arcs, arc_filtration, arc_shift,
                                 p.quotient_vertex_filtration)
    return _periodica.barcode(p.d, tree)


# ---- 1. golden barcodes ----

def regenerate():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name in CASES:
        p, bc = compute(name)
        write_barcodes(GOLDEN / f'{name}.npz', p.d, bc, meta={'case': name, 'eps': EPS})
        print(f'wrote {name}: {[len(b) for b in canonical(bc)]} canonical bars per dimension')


def test_golden():
    for name in CASES:
        p, bc = compute(name)
        d, ref, meta = read_barcodes(GOLDEN / f'{name}.npz')
        assert d == p.d, name
        assert_match(bc, ref, label=f'{name} (grid_merge_tree)')
        assert_match(generic_barcodes(p), ref, label=f'{name} (merge_tree on explicit arcs)')
        # build_arcs=False: same tree without the arc arrays
        q = Periodica()
        q.periodic_grid(p.U, p.grid_values, sublevel=p.grid_field[0] == p.grid_values.ravel()[0], build_arcs=False)
        assert not hasattr(q, 'quotient_arcs')
        q.merge_tree()
        assert_match(q.barcodes(), ref, label=f'{name} (build_arcs=False)')
        print(f'PASS golden {name} ({[len(b) for b in canonical(bc)]} bars per dimension)')


# ---- 2. arc order invariance ----

def test_arc_order():
    rng = np.random.default_rng(11)
    for name in ['2d_random_skew', '3d_random_skew', '3d_random_small_N']:
        p, bc = compute(name)
        for _ in range(3):
            perm = rng.permutation(len(p.quotient_arcs))
            arcs = p.quotient_arcs[perm].copy()
            arc_f = p.quotient_arc_filtration[perm]
            shift = p.quotient_arc_shift[:, perm].copy()
            # also flip the orientation of some arcs (negated shift)
            flip = rng.random(len(perm)) < 0.5
            arcs[flip] = arcs[flip][:, ::-1]
            shift[:, flip] = -shift[:, flip]
            assert_match(generic_barcodes(p, arcs, arc_f, shift), bc, label=f'{name} shuffled')
    print('PASS arc order and orientation invariance')


# ---- 3. lattice: times are combinatorial, multiplicities geometric ----

def test_lattice():
    rng = np.random.default_rng(5)
    for shape in [(6, 7), (4, 5, 6)]:
        d = len(shape)
        vals = rng.uniform(0, 1, shape)
        p0, bc0 = grid(np.eye(d), vals), None
        p0.merge_tree()
        bc0 = canonical(p0.barcodes())
        assert np.isclose(sum(m for b, dd, m in bc0[0] if np.isinf(dd)), 1.0), bc0[0]

        # scaled lattice c*I: same times, multiplicities in exponent e scale by c^-e
        c = 2.5
        pc = grid(c * np.eye(d), vals)
        pc.merge_tree()
        bcc = canonical(pc.barcodes())
        for e in range(d + 1):
            assert len(bcc[e]) == len(bc0[e]), (e, bcc[e], bc0[e])
            for (b1, d1, m1), (b0, d0, m0) in zip(bcc[e], bc0[e]):
                assert abs(b1 - b0) <= EPS and ((np.isinf(d1) and np.isinf(d0)) or abs(d1 - d0) <= EPS)
                assert abs(m1 - m0 * c ** (-e)) <= EPS, (e, m1, m0)

        # skewed lattice whose fine-grid directions are still the axes: same
        # times; bounded components weigh 1/|det U|; the fully wrapped
        # component weighs 1
        U = np.eye(d) + 0.15 * np.triu(np.ones((d, d)), 1)
        pu = grid(U, vals)
        assert sorted(map(tuple, np.abs(pu.grid_directions).T)) == sorted(map(tuple, np.eye(d, dtype=int)))
        pu.merge_tree()
        bcu = canonical(pu.barcodes())
        for e in range(d + 1):
            assert [(b, dd) for b, dd, m in bcu[e]] == [(b, dd) for b, dd, m in bc0[e]] or all(
                abs(x[0] - y[0]) <= EPS and ((np.isinf(x[1]) and np.isinf(y[1])) or abs(x[1] - y[1]) <= EPS)
                for x, y in zip(bcu[e], bc0[e])), e
        vol = abs(np.linalg.det(U))
        assert all(abs(abs(m) - 1 / vol) <= EPS for b, dd, m in bcu[d]), bcu[d]
        assert np.isclose(sum(m for b, dd, m in bcu[0] if np.isinf(dd)), 1.0), bcu[0]
    print('PASS lattice scaling (times combinatorial, multiplicities geometric)')


# ---- 4. gudhi cross-check in 3D ----

def ridge_field_3d(n, seed):
    """Random interior block (values < 1.4) surrounded by a 'ridge' on the
    three faces i = n-1, j = n-1, k = n-1 with values >= 1.8 increasing with
    i + j + k (every ridge cell has a smaller neighbor, so the ridge creates
    no components): all mergers happen below 1.4, all wraps at or above 1.8,
    within a single component."""
    rng = np.random.default_rng(seed)
    vals = np.zeros((n, n, n))
    vals[: n - 1, : n - 1, : n - 1] = rng.uniform(0.0, 1.4, (n - 1, n - 1, n - 1))
    for idx in np.ndindex(vals.shape):
        if n - 1 in idx:
            vals[idx] = 1.8 + 0.001 * sum(idx) + 1e-6 * np.ravel_multi_index(idx, vals.shape)
    return vals


def test_against_gudhi_3d():
    import gudhi

    for n, seed, U in [(5, 1, np.eye(3)), (7, 2, np.eye(3)),
                       (6, 3, np.array([[1.0, 0.2, 0.1], [0.0, 1.1, 0.15], [0.0, 0.0, 0.9]]))]:
        vals = ridge_field_3d(n, seed)
        p = grid(U, vals)
        assert sorted(map(tuple, np.abs(p.grid_directions).T)) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        p.merge_tree()
        bc = canonical(p.barcodes())
        vol = abs(np.linalg.det(U))

        pcc = gudhi.PeriodicCubicalComplex(vertices=vals, periodic_dimensions=[True, True, True])
        pers = pcc.persistence(min_persistence=0)
        ref_merge = sorted((b, d) for dim, (b, d) in pers if dim == 0 and np.isfinite(d) and d - b > EPS)
        ref_wrap = sorted(b for dim, (b, d) in pers if dim == 1 and np.isinf(d))
        assert len(ref_wrap) == 3

        # mergers: finite bars of the R^3 coefficient below the ridge, weight 1/vol
        mergers = sorted((b, d) for b, d, m in bc[3] if d < 1.5)
        assert all(abs(m - 1 / vol) <= EPS for b, d, m in bc[3]), bc[3]
        assert len(mergers) == len(ref_merge) and np.allclose(mergers, ref_merge, atol=EPS), (mergers, ref_merge)

        # wraps: the surviving bounded bar dies at the first wrap; the finite
        # deaths of the lower coefficients are exactly the three wrap times
        survivor = [(b, d) for b, d, m in bc[3] if d >= 1.5]
        assert len(survivor) == 1 and np.isclose(survivor[0][0], vals.min()), survivor
        assert abs(survivor[0][1] - ref_wrap[0]) <= EPS, (survivor, ref_wrap)
        deaths = sorted({d for e in range(3) for b, d, m in bc[e] if np.isfinite(d)})
        assert len(deaths) == 3 and np.allclose(deaths, ref_wrap, atol=EPS), (deaths, ref_wrap)
        # every component bar is born at the global minimum
        births = {round(b, 9) for e in range(3) for b, d, m in bc[e]}
        assert births == {round(vals.min(), 9)}, births
        infinite = [(b, m) for b, d, m in bc[0] if np.isinf(d)]
        assert len(infinite) == 1 and abs(infinite[0][1] - 1.0) <= EPS, infinite
        print(f'PASS gudhi 3D cross-check ({n}^3, {"skewed" if U[0, 1] else "identity"}: '
              f'{len(ref_merge)} mergers, wraps at {np.round(ref_wrap, 4)})')


if __name__ == '__main__':
    if '--regenerate' in sys.argv:
        regenerate()
        sys.exit(0)
    test_golden()
    test_arc_order()
    test_lattice()
    test_against_gudhi_3d()
    print('All grid barcode tests passed.')
