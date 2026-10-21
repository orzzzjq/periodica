"""Plain-assert tests for the barcode file (write_barcodes / read_barcodes /
Periodica.save_barcodes / load_barcodes): a compressed .npz holding the
d + 1 barcodes, from which the descriptors (curves, curve vectors, images,
plots) are recovered without the geometry, quotient complex or merge tree.

Run with: .venv/bin/python tests/test_barcode_file.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from periodica.core import (BARCODE_FILE_FORMAT, Periodica, barcode_range, read_barcodes,
                            write_barcodes)

inf = np.inf
ROOT = Path(__file__).resolve().parent.parent
SQUARE = {'d': 2, 'U': np.eye(2), 'n_points': 1, 'points': np.array([[0.5, 0.5]]).T}


def computed():
    """Fresh instances with barcodes from every kind of input."""
    cases = []
    for f, kind in [('examples/2d_geometry.txt', 'delaunay'), ('examples/2d_geometry.txt', 'voronoi'),
                    ('examples/3d_geometry.txt', 'delaunay'), ('examples/3d_geometry.txt', 'voronoi')]:
        p = Periodica()
        p.load_geometry(ROOT / f)
        p.quotient_complex(kind)
        p.merge_tree()
        p.barcodes()
        cases.append((f'{f} {kind}', p))
    p = Periodica()
    p.load_grid(ROOT / 'examples/2d_periodic_function.txt', sublevel=False)
    p.merge_tree()
    p.barcodes()
    cases.append(('2d grid superlevel', p))
    for f in ['examples/2d_quotient_complex_1.txt', 'examples/3d_quotient_complex.txt']:
        p = Periodica()
        p.load_quotient_complex(str(ROOT / f))
        p.barcodes()
        cases.append((f, p))
    return cases


def positive_length(bcodes):
    return [[b for b in bars if b[1] != b[0]] for bars in bcodes]


def test_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'b.npz')
        for name, p in computed():
            # everything kept: exact round trip, as float tuples
            p.save_barcodes(path, drop_zero_length=False)
            d, bc, meta = read_barcodes(path)
            assert d == p.d and bc == p.bcodes and meta == {}, name
            assert all(isinstance(b, tuple) and all(type(v) is float for v in b) for bars in bc for b in bars)
            assert any(dd == inf for _, dd, _ in bc[0]), name  # the infinite bar survives
            # default: zero-length bars dropped, nothing else changes
            p.save_barcodes(path)
            d, bc, _ = read_barcodes(path)
            assert bc == positive_length(p.bcodes), name
            # a fresh instance recovers identical descriptors from the file
            q = Periodica()
            q.load_barcodes(path)
            assert q.d == p.d and q.barcodes() == bc and q.barcode_meta == {}, name
            for (t1, v1), (t2, v2) in zip(p.curves(), q.curves()):
                assert np.array_equal(t1, t2) and np.array_equal(v1, v2), name
            assert barcode_range(p.bcodes) == barcode_range(q.bcodes), name
            assert np.array_equal(p.curve_vectors(resolution=50), q.curve_vectors(resolution=50)), name
            for a, b in zip(p.images(size=20), q.images(size=20)):
                assert np.array_equal(a, b), name
            print(f'PASS round trip: {name}')


def test_zero_length_and_size():
    p = Periodica()
    p.load_grid(ROOT / 'examples/2d_periodic_function.txt', sublevel=False)
    p.merge_tree()
    bc = p.barcodes()
    n_all = sum(map(len, bc))
    n_pos = sum(map(len, positive_length(bc)))
    assert n_all == 404 and n_pos < 40, (n_all, n_pos)  # one R^2 bar per grid point, few critical points
    with tempfile.TemporaryDirectory() as tmp:
        full, dropped = os.path.join(tmp, 'full.npz'), os.path.join(tmp, 'dropped.npz')
        p.save_barcodes(full, drop_zero_length=False)
        p.save_barcodes(dropped)
        assert os.path.getsize(dropped) < os.path.getsize(full) / 2, (os.path.getsize(dropped), os.path.getsize(full))
        # the images do not see zero-length bars, so both files reproduce them
        imgs = p.images(size=16)
        for path in (full, dropped):
            q = Periodica()
            q.load_barcodes(path)
            for a, b in zip(imgs, q.images(size=16)):
                assert np.array_equal(a, b), path
    print('PASS zero-length bars dropped (file size, images unchanged)')


def test_meta_and_dtype():
    p = Periodica()
    p.load_geometry(ROOT / 'examples/3d_geometry.txt')
    p.quotient_complex('voronoi')
    p.merge_tree()
    with tempfile.TemporaryDirectory() as tmp:
        meta = {'source': '3d_geometry.txt', 'complex': 'voronoi', 'weights': [0.01, 0.02, 0.03], 'nested': {'a': 1}}
        p.save_barcodes(os.path.join(tmp, 'm'), meta=meta)  # no extension: numpy appends .npz
        path = os.path.join(tmp, 'm.npz')
        assert os.path.exists(path)
        _, _, m = read_barcodes(path)
        assert m == meta, m
        q = Periodica()
        q.load_barcodes(path)
        assert q.barcode_meta == meta
        # float32: small relative error, infinite bars and zero-length bars preserved
        p32 = os.path.join(tmp, 'f32.npz')
        p.save_barcodes(p32, dtype=np.float32, drop_zero_length=False)
        with np.load(p32) as z:
            assert z['bars'].dtype == np.float32 and str(z['format']) == BARCODE_FILE_FORMAT
        _, bc32, _ = read_barcodes(p32)
        for bars, bars32 in zip(p.bcodes, bc32):
            assert len(bars) == len(bars32)
            for (b, d, m_), (b2, d2, m2) in zip(bars, bars32):
                assert abs(b - b2) <= 1e-6 * max(1.0, abs(b))
                assert (d == inf and d2 == inf) or abs(d - d2) <= 1e-6 * max(1.0, abs(d))
                assert abs(m_ - m2) <= 1e-6 * max(1.0, abs(m_))
                assert (d == b) == (d2 == b2)
    print('PASS meta round trip, .npz suffix, float32')


def test_instance_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'v.npz')
        src = Periodica()
        src.load_geometry(ROOT / 'examples/3d_geometry.txt')
        src.quotient_complex('voronoi')
        src.merge_tree()
        src.save_barcodes(path)
        loaded = positive_length(src.barcodes())

        # a loaded instance has only barcodes: descriptors work, the rest needs input
        q = Periodica()
        q.load_barcodes(path)
        assert q.barcodes() is q.bcodes and q.bcodes == loaded
        for call in (q.merge_tree, q.print_merge_tree, lambda: q.plot_delaunay(show=False)):
            try:
                call()
                assert False, 'must raise without input points'
            except Exception as e:
                assert 'No input points' in str(e), e
        q.plot_all_descriptors(show=False)
        assert os.path.exists('all_descriptors.svg')
        os.remove('all_descriptors.svg')

        # loading into an instance with a complex and a tree replaces them
        p = Periodica()
        p.set_geometry(SQUARE)
        p.quotient_complex('delaunay')
        p.merge_tree()
        p.curves()
        p.images(size=8)
        p.load_barcodes(path)
        assert p.d == 3 and p.barcodes() == loaded
        for attr in ('tree', 'quotient_arcs', 'filtration_curves', 'persistence_images'):
            assert not hasattr(p, attr), attr
        assert len(p.curves()) == 4 and len(p.images(size=8)) == 4

        # new geometry on a loaded instance recomputes instead of returning stale bars
        fresh = Periodica()
        fresh.set_geometry(SQUARE)
        fresh.quotient_complex('delaunay')
        expected = fresh.barcodes()
        p.set_geometry(SQUARE)
        p.quotient_complex('delaunay')
        assert p.barcodes() == expected and hasattr(p, 'tree')
    print('PASS instance state (barcodes() rule, reset on load, recompute on new input)')


def test_errors():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            read_barcodes(os.path.join(tmp, 'nope.npz'))
            assert False
        except FileNotFoundError:
            pass
        npy = os.path.join(tmp, 'x.npy')
        np.save(npy, np.zeros(3))
        nokey = os.path.join(tmp, 'nokey.npz')
        np.savez_compressed(nokey, d=np.array(2))

        def bad(**over):
            arrs = dict(format=np.array(BARCODE_FILE_FORMAT), d=np.array(2, dtype=np.int8),
                        dims=np.array([0, 2], dtype=np.uint8),
                        bars=np.array([[0.0, inf, 1.0], [0.0, 0.5, 1.0]]), meta=np.array('{}'))
            arrs.update(over)
            path = os.path.join(tmp, 'bad.npz')
            np.savez_compressed(path, **arrs)
            return path

        # the baseline archive is valid; the empty middle dimension is kept
        assert read_barcodes(bad()) == (2, [[(0.0, inf, 1.0)], [], [(0.0, 0.5, 1.0)]], {})
        invalid = [npy, nokey,
                   bad(format=np.array('something-else')),
                   bad(d=np.array(4, dtype=np.int8)),
                   bad(dims=np.array([0, 3], dtype=np.uint8)),
                   bad(dims=np.array([0], dtype=np.uint8)),
                   bad(bars=np.array([[0.0, inf, 1.0], [1.0, 0.5, 1.0]])),
                   bad(bars=np.array([[np.nan, 1.0, 1.0], [0.0, 0.5, 1.0]])),
                   bad(bars=np.array([[0.0, 1.0, inf], [0.0, 0.5, 1.0]])),
                   bad(bars=np.zeros((2, 2)))]
        for path in invalid:
            try:
                read_barcodes(path)
                assert False, path
            except ValueError:
                pass
        for d, bc in [(4, [[]] * 5), (2, [[], []])]:
            try:
                write_barcodes(os.path.join(tmp, 'w.npz'), d, bc)
                assert False, (d, bc)
            except ValueError:
                pass
    print('PASS error handling')


if __name__ == '__main__':
    test_round_trip()
    test_zero_length_and_size()
    test_meta_and_dtype()
    test_instance_state()
    test_errors()
    print('All barcode-file tests passed.')
