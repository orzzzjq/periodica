"""Tests for hidden-point handling in the weighted periodic Delaunay pipeline.

In a weighted (regular/power) triangulation, a point p is hidden by a point q
when ||p - q||^2 <= w_q - w_p: its power cell is empty and it never appears as
a vertex of the triangulation. The pipeline must exclude such points from the
quotient complex and merge tree instead of leaving them as spurious isolated
components (which show up as extra infinite bars in the barcode).

Run from anywhere:  .venv/bin/python tests/test_hidden_points.py
"""

import math
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
from matplotlib import pyplot as plt
from periodica import _periodica
from periodica.core import Periodica


def count_infinite_bars(periodica):
    # A hidden point leaked into the merge tree is an isolated component that never
    # merges with any lattice translate; it shows up as a spurious infinite bar in
    # the top-dimensional barcode, so count infinite bars across all dimensions.
    return sum(1 for bars in periodica.barcodes() for bar in bars if math.isinf(bar[1]))


def build(d, points, weights):
    periodica = Periodica()
    periodica.set_geometry({
        'd': d,
        'U': np.eye(d),
        'n_points': points.shape[1],
        'points': points,
        'weights': np.asarray(weights, dtype=float),
    })
    periodica.quotient_complex('delaunay')
    periodica.merge_tree()
    return periodica


def check_hidden_case(name, d, points, weights, expected_hidden):
    p = build(d, points, weights)
    n_kept = points.shape[1] - len(expected_hidden)

    assert p.hidden_points == expected_hidden, \
        f'{name}: expected hidden points {expected_hidden}, got {p.hidden_points}'
    assert p.n_quotient_vertices == n_kept, \
        f'{name}: expected {n_kept} quotient vertices, got {p.n_quotient_vertices}'
    arcs = np.asarray(p.quotient_arcs)
    assert arcs.size > 0 and arcs.min() >= 0 and arcs.max() < n_kept, \
        f'{name}: arc endpoints out of range [0, {n_kept}): {arcs}'
    assert not np.isnan(np.asarray(p.quotient_arc_filtration)).any(), \
        f'{name}: NaN in arc filtration values'
    n_inf = count_infinite_bars(p)
    assert n_inf == 1, \
        f'{name}: expected exactly 1 infinite bar in total, got {n_inf} ' \
        f'(a spurious extra bar means a hidden point leaked into the merge tree)'
    print(f'PASS {name}')


def test_2d_hidden_point():
    # Point 1 is hidden by point 0: dist^2 = 0.0025 <= w_0 - w_1 = 0.09
    points = np.array([
        [0.25, 0.25],
        [0.30, 0.25],
    ]).T
    check_hidden_case('2d hidden point', 2, points, [0.09, 0.0], expected_hidden=[1])


def test_3d_hidden_point():
    points = np.array([
        [0.25, 0.25, 0.25],
        [0.30, 0.25, 0.25],
    ]).T
    check_hidden_case('3d hidden point', 3, points, [0.09, 0.0], expected_hidden=[1])


def test_2d_unweighted_control():
    # Same points, no weights: nothing is hidden and both points survive.
    points = np.array([
        [0.25, 0.25],
        [0.30, 0.25],
    ]).T
    check_hidden_case('2d unweighted control', 2, points, [0.0, 0.0], expected_hidden=[])


def test_large_weight_negative_filtration():
    # A weight large enough that some weighted alpha values are negative
    # (e.g. two copies of a w=0.5 point at distance 1: alpha = 0.25 - 0.5).
    # The filtration must use a signed sqrt, never producing NaN. Point 0
    # is also hidden here (dist^2 = 0.04 <= 0.5).
    points = np.array([
        [0.5, 0.5],
        [0.3, 0.5],
    ]).T
    p = build(2, points, [0.0, 0.5])
    assert p.hidden_points == [0], f'expected point 0 hidden, got {p.hidden_points}'
    filt = np.asarray(p.quotient_arc_filtration)
    assert not np.isnan(filt).any(), f'NaN in arc filtrations: {filt}'
    assert (filt < 0).any(), 'expected some negative (signed-sqrt) filtration values'
    n_inf = count_infinite_bars(p)
    assert n_inf == 1, f'expected exactly 1 infinite bar, got {n_inf}'
    print('PASS large weight (negative alpha, signed sqrt)')


def test_plot_delaunay_arc_positions():
    # Point 0 is hidden by point 1, so kept_points == [1] and the arc-endpoint
    # remap is non-identity: drawing arcs with P[:, s] instead of
    # P[:, kept_points[s]] would anchor them at the hidden point's position.
    points = np.array([
        [0.30, 0.25],
        [0.25, 0.25],
    ]).T
    weights = np.array([0.0, 0.09])

    p = Periodica()
    p.set_geometry({'d': 2, 'U': np.eye(2), 'n_points': 2,
                    'points': points, 'weights': weights})
    p.plot_delaunay(show=False)
    assert p.hidden_points == [0], f'expected point 0 hidden, got {p.hidden_points}'

    P, _ = _periodica.full_delaunay(np.eye(2), points, weights)
    kept_positions = P[:, p.kept_points].T
    hidden_position = P[:, 0]

    arcs = [l for l in plt.gcf().axes[0].get_lines() if l.get_color() == '#0000FE']
    assert arcs, 'no highlighted periodic Delaunay arcs were drawn'
    for line in arcs:
        x, y = line.get_xdata(), line.get_ydata()
        start = np.array([x[0], y[0]])
        assert any(np.allclose(start, kp) for kp in kept_positions), \
            f'arc starts at {start}, which is not a kept canonical point ' \
            f'{"(it is the hidden point!)" if np.allclose(start, hidden_position) else ""}'
    plt.close('all')
    print('PASS plot_delaunay arc positions with hidden point')


def test_quotient_file_regression():
    # The file-based pipeline bypasses periodic_delaunay and must be unaffected.
    p = Periodica()
    p.load_quotient_complex(os.path.join(REPO_ROOT, 'examples', 'example_2d_1.txt'))
    p.merge_tree()
    assert count_infinite_bars(p) == 1, 'file-based quotient complex regressed'
    print('PASS file-based quotient complex regression')


if __name__ == '__main__':
    test_2d_hidden_point()
    test_3d_hidden_point()
    test_2d_unweighted_control()
    test_large_weight_negative_filtration()
    test_plot_delaunay_arc_positions()
    test_quotient_file_regression()
    print('All hidden-point tests passed.')
