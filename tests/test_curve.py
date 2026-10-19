"""Plain-assert tests for the curve descriptor (curves_from_barcodes /
curve_value / Periodica.curves): the curve value at t must equal the total
multiplicity of the bars alive at t, with bars alive on [birth, death).

Run with: .venv/bin/python tests/test_curve.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from periodica.core import Periodica, curve_value, curves_from_barcodes

inf = np.inf
ROOT = Path(__file__).resolve().parent.parent


def brute_force(bars, t):
    return sum(m for b, d, m in bars if b <= t < d)


def check_curve(bars, times, values, rng, label):
    # right-continuous: the stored times are exactly the distinct event times
    # of the non-degenerate bars, in increasing order
    events = sorted({x for b, d, m in bars if d != b for x in (b, d) if np.isfinite(x)})
    assert np.array_equal(times, events), (label, times, events)
    assert len(values) == len(times)
    probes = list(times) + [t + 1e-9 for t in times] + [t - 1e-9 for t in times]
    if len(times):
        probes += list(rng.uniform(times[0] - 1, times[-1] + 1, 200))
    else:
        probes += [-1.0, 0.0, 1.0]
    for t in probes:
        got = float(curve_value(times, values, t))
        want = brute_force(bars, t)
        assert abs(got - want) < 1e-9, (label, t, got, want)


def test_hand_built():
    # ties at t=1 and t=2, a zero-length bar, negative and fractional
    # multiplicities, an infinite bar
    bars = [(0.0, 2.0, 1.0), (1.0, 1.0, 5.0), (1.0, inf, 0.5), (2.0, 3.0, -0.25), (0.0, 1.0, 2.0)]
    (times, values), = curves_from_barcodes([bars])
    assert times.tolist() == [0.0, 1.0, 2.0, 3.0], times
    assert np.allclose(values, [3.0, 1.5, 0.25, 0.5]), values
    assert float(curve_value(times, values, -1.0)) == 0.0
    assert float(curve_value(times, values, 100.0)) == 0.5
    check_curve(bars, times, values, np.random.default_rng(0), 'hand-built')
    # an empty barcode gives an empty curve that is 0 everywhere
    (times, values), = curves_from_barcodes([[]])
    assert len(times) == 0 and len(values) == 0
    assert np.array_equal(curve_value(times, values, [0.0, 1.0]), [0.0, 0.0])
    print('PASS hand-built barcode (ties, zero-length, signed, infinite)')


def test_random_barcodes():
    rng = np.random.default_rng(1)
    for trial in range(20):
        n = int(rng.integers(1, 40))
        births = np.round(rng.uniform(-1, 1, n), 1)  # coarse values: many ties
        lengths = np.round(rng.uniform(0, 1, n), 1)
        bars = []
        for b, l, m in zip(births, lengths, rng.uniform(-1, 1, n)):
            death = inf if rng.uniform() < 0.1 else b + l
            bars.append((float(b), float(death), float(m)))
        (times, values), = curves_from_barcodes([bars])
        check_curve(bars, times, values, rng, f'random {trial}')
    print('PASS random barcodes vs brute force (20 trials)')


def test_examples():
    rng = np.random.default_rng(2)
    cases = [('examples/2d_geometry.txt', 'delaunay'), ('examples/2d_geometry.txt', 'voronoi'),
             ('examples/3d_geometry.txt', 'delaunay'), ('examples/3d_geometry.txt', 'voronoi')]
    for f, kind in cases:
        p = Periodica()
        p.load_geometry(ROOT / f)
        p.quotient_complex(kind)
        p.merge_tree()
        curves = p.curves()
        assert len(curves) == p.d + 1
        for k, (times, values) in enumerate(curves):
            check_curve(p.bcodes[k], times, values, rng, f'{f} {kind} dim {k}')
        # the curve of dimension 0 ends on the infinite bars' multiplicity
        _, v0 = curves[0]
        inf_mult = sum(m for _, d, m in p.bcodes[0] if np.isinf(d))
        assert len(v0) == 0 and inf_mult == 0 or abs(v0[-1] - inf_mult) < 1e-9, (f, kind, v0, inf_mult)
    for f in ['examples/2d_quotient_complex_1.txt', 'examples/3d_quotient_complex.txt']:
        p = Periodica()
        p.load_quotient_complex(str(ROOT / f))
        for k, (times, values) in enumerate(p.curves()):
            check_curve(p.bcodes[k], times, values, rng, f'{f} dim {k}')
    print('PASS example geometries and quotient complexes')


def test_plots():
    p = Periodica()
    p.load_geometry(ROOT / 'examples/3d_geometry.txt')
    p.quotient_complex('delaunay')
    p.merge_tree()
    p.plot_curves(show=False)
    p.plot_all_descriptors(show=False)
    for f in ('curve.svg', 'all_descriptors.svg'):
        assert os.path.exists(f)
        os.remove(f)
    print('PASS plot_curves / plot_all_descriptors render')


if __name__ == '__main__':
    test_hand_built()
    test_random_barcodes()
    test_examples()
    test_plots()
    print('All curve tests passed.')
