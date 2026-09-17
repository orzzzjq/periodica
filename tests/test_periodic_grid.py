"""Plain-assert tests for Periodica.periodic_grid (grid quotient complex).

Run with: .venv/bin/python tests/test_periodic_grid.py

Barcode convention: bcodes[k] holds the signed bars of the coefficient of
R^(d-k), so bounded-component (exponent-0) bars live in bcodes[d].
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from periodica.core import Periodica


def finite_positive_bars(bars, eps=1e-9):
    return sorted((b, d) for b, d, m in bars if np.isfinite(d) and d - b > eps and m > 0)


def brute_shortest_lengths(W, R=8):
    """Lengths of the d shortest linearly independent vectors of lattice W."""
    d = W.shape[0]
    cands = [np.array(c) for c in product(range(-R, R + 1), repeat=d) if any(c)]
    cands.sort(key=lambda c: np.linalg.norm(W @ c))
    chosen = []
    for c in cands:
        M = np.column_stack(chosen + [c])
        if np.linalg.matrix_rank(M) == len(chosen) + 1:
            chosen.append(c)
            if len(chosen) == d:
                break
    return [np.linalg.norm(W @ c) for c in chosen]


def grid(U, values):
    p = Periodica()
    p.periodic_grid(np.asarray(U, dtype=float), np.asarray(values, dtype=float))
    return p


# ---- 1. direction selection ----

def test_directions():
    # (a) identity lattice, uniform N: directions are the coordinate axes
    p = grid(np.eye(2), np.zeros((3, 3)))
    assert sorted(map(tuple, np.abs(p.grid_directions).T)) == [(0, 1), (1, 0)]
    p = grid(np.eye(3), np.zeros((2, 2, 2)))
    assert sorted(map(tuple, np.abs(p.grid_directions).T)) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    # (b) skewed lattice + anisotropic N: lengths match the brute-force
    # successive minima of the fine lattice
    for U, N in [
        (np.array([[1.0, 0.9], [0.0, 0.1]]), (4, 6)),
        (np.array([[1.0, 0.45, 0.1], [0.0, 0.7, 0.6], [0.0, 0.0, 0.8]]), (3, 4, 5)),
    ]:
        p = grid(U, np.zeros(N))
        W = U @ np.diag(1.0 / np.array(N))
        got = sorted(np.linalg.norm(W @ p.grid_directions[:, k]) for k in range(len(N)))
        assert np.allclose(got, brute_shortest_lengths(W)), (got, brute_shortest_lengths(W))
    print('PASS direction selection (axes / skewed / anisotropic)')

    # (c) flat fine lattice: the 3 shortest relevant vectors are coplanar,
    # greedy must skip the in-plane diagonal and take the long direction
    U = np.array([[2.0, -0.8, 0.0], [0.0, 1.8, 0.0], [0.0, 0.0, 20.0]])
    N = (2, 2, 2)
    p = grid(U, np.zeros(N))
    W = U @ np.diag(1.0 / np.array(N))
    lens = sorted(np.linalg.norm(W @ p.grid_directions[:, k]) for k in range(3))
    assert np.allclose(lens, brute_shortest_lengths(W)), lens
    assert lens[2] > 5, f'coplanar third direction not skipped: {lens}'
    print('PASS direction selection (flat lattice rank trap)')


# ---- 2. merge events match periodic cubical persistence (gudhi) ----

def test_against_gudhi():
    import gudhi

    # random interior block; on the last row + column a strictly increasing
    # "snake" ridge (each ridge cell has a smaller ridge/interior neighbor,
    # so no ridge minima): all mergers happen below 1.4, all wraps at or
    # above 1.8, and exactly one exponent-0 bar dies at the first wrap
    for n, seed in [(4, 3), (9, 5)]:
        rng = np.random.default_rng(seed)
        vals = np.zeros((n, n))
        vals[: n - 1, : n - 1] = rng.uniform(0.0, 1.4, (n - 1, n - 1))
        ridge = [(n - 1, j) for j in range(n)] + [(i, n - 1) for i in range(n - 1)]
        for k, ij in enumerate(ridge):
            vals[ij] = 1.8 + 0.01 * k

        p = grid(np.eye(2), vals)
        p.merge_tree()
        bc = p.barcodes()
        exp0 = finite_positive_bars(bc[2])
        mergers = [bar for bar in exp0 if bar[1] < 1.5]
        survivor = [bar for bar in exp0 if bar[1] >= 1.5]
        assert len(survivor) == 1 and np.isclose(survivor[0][0], vals.min()), survivor

        # V-construction (vertices=): lower-star on grid vertices with
        # 2d-adjacency, matching our complex; top_dimensional_cells would
        # give 8-adjacency in dim 0
        pcc = gudhi.PeriodicCubicalComplex(vertices=vals, periodic_dimensions=[True, True])
        ref = sorted((b, d) for dim, (b, d) in pcc.persistence()
                     if dim == 0 and np.isfinite(d) and d - b > 1e-9)
        assert len(mergers) == len(ref) and np.allclose(mergers, ref), (mergers, ref)
        print(f'PASS gudhi cross-check ({n}x{n}: {len(ref)} finite dim-0 bars match)')


# ---- 3. wrap (catenation) thresholds by hand ----

def test_wrap_events():
    # single minimum at (0,0); the cheapest horizontal wrap runs through
    # row 0 (completes at 0.2), the cheapest vertical wrap through column 0
    # (completes at 0.6)
    vals = np.array([
        [0.0, 0.1, 0.2],
        [0.5, 0.8, 0.85],
        [0.6, 0.87, 0.9],
    ])
    p = grid(np.eye(2), vals)
    p.merge_tree()
    bc = p.barcodes()

    exp0 = finite_positive_bars(bc[2])
    assert exp0 == [(0.0, 0.2)], exp0  # bounded phase ends at the first wrap
    deaths1 = sorted(d for b, d, m in bc[1] if np.isfinite(d))
    assert np.allclose(deaths1, [0.2, 0.6]), deaths1  # 1-periodic phase [0.2, 0.6)
    inf0 = [(b, m) for b, d, m in bc[0] if not np.isfinite(d)]
    fin0 = [(b, d) for b, d, m in bc[0] if np.isfinite(d)]
    assert len(inf0) == 1 and np.isclose(inf0[0][1], 1.0), inf0  # rank-2 from 0.6 on
    assert len(fin0) == 1 and np.isclose(fin0[0][1], 0.6), fin0
    print('PASS wrap thresholds (0.2 horizontal, 0.6 vertical)')


# ---- 4. small N: self-loops and parallel edges ----

def test_small_N():
    # N_1 = 1: the vertical direction produces self-loop arcs with shift
    vals = np.array([[0.0, 0.1, 0.2, 0.3, 0.4]])
    p = grid(np.eye(2), vals)
    self_loops = np.sum(p.quotient_arcs[:, 0] == p.quotient_arcs[:, 1])
    assert self_loops == 5, self_loops
    p.merge_tree()
    bc = p.barcodes()
    assert any(not np.isfinite(d) for b, d, m in bc[0])
    # the self-loop at the minimum wraps the component at its birth
    deaths1 = [d for b, d, m in bc[1] if np.isfinite(d)]
    assert np.isclose(min(deaths1), 0.0), deaths1

    # N_1 = 2: parallel edges (two distinct torus edges between each pair)
    p = grid(np.eye(2), np.arange(6).reshape(2, 3) / 10.0)
    pairs = {tuple(sorted(a)) for a in p.quotient_arcs.tolist() if a[0] != a[1]}
    counts = [np.sum([tuple(sorted(a)) == q for a in p.quotient_arcs.tolist()]) for q in pairs]
    assert max(counts) == 2, counts
    p.merge_tree()
    p.barcodes()
    print('PASS small N (self-loops, parallel edges)')


# ---- 5. grid file format ----

def test_grid_file():
    import tempfile
    from periodica.core import _parse_grid_text

    root = Path(__file__).resolve().parent.parent
    # the 2D example is exactly the wrap-threshold field
    p = Periodica()
    p.load_grid(root / 'examples' / 'grid_2d_1.txt')
    assert np.array_equal(p.grid_values, [[0.0, 0.1, 0.2], [0.5, 0.8, 0.85], [0.6, 0.87, 0.9]])
    p.merge_tree()
    assert finite_positive_bars(p.barcodes()[2]) == [(0.0, 0.2)]

    q = Periodica()
    q.load_grid(root / 'examples' / 'grid_3d_1.txt')
    assert q.d == 3 and q.grid_values.shape == (2, 2, 2)
    assert np.allclose(q.U, 2 * np.eye(3))
    q.merge_tree()
    q.barcodes()

    # save/load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / 'rt.txt'
        for src in (p, q):
            src.save_grid(f)
            r = Periodica()
            r.load_grid(f)
            assert np.array_equal(r.grid_values, src.grid_values)
            assert np.allclose(r.U, src.U)
            assert np.array_equal(r.grid_directions, src.grid_directions)

    # parse errors
    def expect_error(text, frag):
        try:
            _parse_grid_text(text)
        except ValueError as e:
            assert frag in str(e), (frag, str(e))
            return
        raise AssertionError(f'no error for {frag!r}')

    head = 'grid:\n1\ndimension:\n2\nlattice:\n1 0\n0 1\n'
    expect_error('geometry:\n1\n', "first line must be 'grid:'")
    expect_error('grid:\n2\n', 'unsupported grid format version')
    expect_error(head + 'shape:\n3\n', 'shape must be 2 positive integers')
    expect_error(head + 'shape:\n3 0\n', 'shape must be 2 positive integers')
    expect_error(head + 'shape:\n2 3\nvalues:\n1 2 3\n4 5\n', 'expected 3 numbers (3 values), got 2')
    expect_error(head + 'shape:\n2 3\nvalues:\n1 2 3\n', 'unexpected end of file (expected value row 2 of 2)')
    expect_error(head + 'shape:\n2 2\nvalues:\n1 2\n3 4\n5 6\n', 'unexpected content after the values')
    print('PASS grid file format (examples, round-trip, parse errors)')


# ---- 6. arc bookkeeping invariants ----

def test_invariants():
    rng = np.random.default_rng(11)
    U = np.array([[1.0, 0.4, 0.0], [0.0, 1.1, 0.3], [0.0, 0.0, 0.9]])
    vals = rng.uniform(0, 1, (3, 4, 2))
    p = grid(U, vals)
    n, d = vals.size, 3
    assert p.n_quotient_vertices == n
    assert p.quotient_arcs.shape == (d * n, 2)
    assert p.quotient_arc_shift.shape == (d, d * n)
    assert p.quotient_arc_filtration.shape == (d * n,)
    f = vals.ravel()
    assert np.array_equal(p.quotient_vertex_filtration, f)
    src, tgt = p.quotient_arcs[:, 0], p.quotient_arcs[:, 1]
    assert np.array_equal(p.quotient_arc_filtration, np.maximum(f[src], f[tgt]))
    # every arc goes to the grid point displaced by one of the directions
    N = np.array(vals.shape)
    gs = np.array(np.unravel_index(src, vals.shape))
    gt = np.array(np.unravel_index(tgt, vals.shape))
    disp = gt + p.quotient_arc_shift * N[:, None] - gs
    key = {tuple(p.grid_directions[:, k]) for k in range(d)}
    assert {tuple(c) for c in disp.T} == key
    p.merge_tree()
    p.barcodes()
    print('PASS arc bookkeeping invariants (3D skewed)')


if __name__ == '__main__':
    test_directions()
    test_against_gudhi()
    test_wrap_events()
    test_small_N()
    test_grid_file()
    test_invariants()
    print('All periodic_grid tests passed.')
