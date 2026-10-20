"""Plain-assert tests for the curve vectorization (curve_range /
curves_to_vectors / Periodica.curve_vectors): a uniform sampling of each
curve on an interval, as a (d + 1) x resolution float32 matrix that can be
saved with np.save for downstream ML.

Run with: .venv/bin/python tests/test_curve_vectors.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from periodica.core import (Periodica, barcode_range, curve_range, curve_value, curves_from_barcodes,
                            curves_to_vectors)

inf = np.inf
ROOT = Path(__file__).resolve().parent.parent


def brute_force(bars, t):
    return sum(m for b, d, m in bars if b <= t < d)


def plot_range(bcodes, right_pad):
    # the x-range formula of plot_curves / plot_barcodes (right_pad = 0.05)
    xmin = min(map(lambda b: min(map(lambda x: x[0], b)), filter(len, bcodes)))
    xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), filter(len, bcodes)))
    span = xmax - xmin
    return xmin - 0.12 * span, xmax + right_pad * span


def test_hand_built():
    bars0 = [(0.0, 2.0, 1.0), (1.0, inf, 0.5), (2.0, 3.0, -0.25)]
    bars1 = [(0.5, 1.5, 2.0)]
    bars2 = []                                   # an empty barcode: all-zero row
    curves = curves_from_barcodes([bars0, bars1, bars2])

    # explicit interval, endpoints included, values match brute force
    vec = curves_to_vectors(curves, interval=(-1.0, 4.0), resolution=11)
    assert vec.shape == (3, 11) and vec.dtype == np.float32, (vec.shape, vec.dtype)
    x = np.linspace(-1.0, 4.0, 11)
    assert x[0] == -1.0 and x[-1] == 4.0
    for k, bars in enumerate([bars0, bars1, bars2]):
        want = np.array([brute_force(bars, t) for t in x])
        assert np.allclose(vec[k], want), (k, vec[k], want)
    assert np.all(vec[2] == 0)

    # default interval = padded event-time span
    lo, hi = curve_range(curves)
    assert np.isclose(lo, 0.0 - 0.12 * 3.0) and np.isclose(hi, 3.0 + 0.05 * 3.0), (lo, hi)
    vec = curves_to_vectors(curves, resolution=7)
    x = np.linspace(lo, hi, 7)
    assert np.allclose(vec[0], [brute_force(bars0, t) for t in x])

    # zero-length bars have no curve events: curve_range ignores them while
    # barcode_range (the plots' formula) counts them
    bars_z = bars0 + [(5.0, 5.0, 1.0)]
    assert curve_range(curves_from_barcodes([bars_z])) == curve_range(curves_from_barcodes([bars0]))
    lo, hi = barcode_range([bars_z])
    assert np.isclose(lo, 0.0 - 0.12 * 5.0) and np.isclose(hi, 5.0 + 0.05 * 5.0), (lo, hi)
    assert np.allclose(barcode_range([bars0, bars1, bars2]), curve_range(curves))

    # dtype and resolution edge cases
    assert curves_to_vectors(curves, dtype=np.float64).dtype == np.float64
    one = curves_to_vectors(curves, interval=(1.0, 1.0), resolution=1)
    assert one.shape == (3, 1) and np.allclose(one[:, 0], [1.5, 2.0, 0.0]), one
    print('PASS hand-built curves (values, interval, dtype, resolution)')


def test_errors():
    curves = curves_from_barcodes([[(0.0, 1.0, 1.0)]])
    for bad in (0, -3):
        try:
            curves_to_vectors(curves, resolution=bad)
            assert False, 'resolution <= 0 must raise'
        except ValueError:
            pass
    try:
        curves_to_vectors(curves, interval=(1.0, 0.0))
        assert False, 'reversed interval must raise'
    except ValueError:
        pass
    empty = curves_from_barcodes([[], []])
    try:
        curves_to_vectors(empty)
        assert False, 'no events and no interval must raise'
    except ValueError:
        pass
    assert np.all(curves_to_vectors(empty, interval=(0.0, 1.0), resolution=5) == 0)
    try:
        barcode_range([[], []])
        assert False, 'all-empty barcodes must raise'
    except ValueError:
        pass
    print('PASS error handling')


def test_save_and_load():
    curves = curves_from_barcodes([[(0.0, 2.0, 1.0), (1.0, inf, 0.5)], [(0.5, 1.5, 2.0)]])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'vec.npy')
        vec = curves_to_vectors(curves, interval=(0.0, 2.0), resolution=50, save_to_file=path)
        back = np.load(path)
        assert back.dtype == np.float32 and back.shape == (2, 50)
        assert np.array_equal(back, vec)
        # 2 x 50 float32 + the 128-byte npy header
        assert os.path.getsize(path) == 2 * 50 * 4 + 128, os.path.getsize(path)
        # numpy appends .npy when the path has no extension
        curves_to_vectors(curves, interval=(0.0, 2.0), resolution=4, save_to_file=os.path.join(tmp, 'noext'))
        assert os.path.exists(os.path.join(tmp, 'noext.npy'))
    print('PASS np.save round trip')


def test_periodica_method():
    cases = [('examples/2d_geometry.txt', 'delaunay'), ('examples/2d_geometry.txt', 'voronoi'),
             ('examples/3d_geometry.txt', 'delaunay'), ('examples/3d_geometry.txt', 'voronoi')]
    for f, kind in cases:
        p = Periodica()
        p.load_geometry(ROOT / f)
        p.quotient_complex(kind)
        p.merge_tree()
        # computes the curves itself, one row per dimension
        vec = p.curve_vectors(resolution=32)
        assert hasattr(p, 'filtration_curves')
        assert vec.shape == (p.d + 1, 32) and vec.dtype == np.float32, (f, kind, vec.shape)
        # default interval is the x-range plot_curves uses: barcode_range must
        # stay in sync with the plots' inline formula (both paddings)
        lo, hi = barcode_range(p.bcodes)
        assert np.allclose((lo, hi), plot_range(p.bcodes, 0.05)), (f, kind, (lo, hi))
        assert np.allclose(barcode_range(p.bcodes, (0.12, 0.12)), plot_range(p.bcodes, 0.12))
        assert np.array_equal(vec, curves_to_vectors(p.filtration_curves, (lo, hi), 32))
        # rows are the curves sampled on that interval
        x = np.linspace(lo, hi, 32)
        for k, (times, values) in enumerate(p.filtration_curves):
            assert np.allclose(vec[k], curve_value(times, values, x), atol=1e-6)
            assert np.allclose(vec[k], [brute_force(p.bcodes[k], t) for t in x], atol=1e-6)
        # the R^0 row ends on the infinite bars' multiplicity
        inf_mult = sum(m for _, d, m in p.bcodes[0] if np.isinf(d))
        assert np.isclose(vec[0, -1], inf_mult), (f, kind, vec[0, -1], inf_mult)
    # quotient-complex input and a dataset-style fixed interval
    p = Periodica()
    p.load_quotient_complex(str(ROOT / 'examples/2d_quotient_complex_1.txt'))
    vec = p.curve_vectors(interval=(0.0, 10.0), resolution=101)
    x = np.linspace(0.0, 10.0, 101)
    for k in range(3):
        assert np.allclose(vec[k], [brute_force(p.bcodes[k], t) for t in x], atol=1e-6)
    print('PASS Periodica.curve_vectors on example inputs')


if __name__ == '__main__':
    test_hand_built()
    test_errors()
    test_save_and_load()
    test_periodica_method()
    print('All curve-vector tests passed.')
