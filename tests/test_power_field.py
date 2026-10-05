"""Plain-assert tests for the sampled power-distance field helpers
(sample_power_field / power_grid_shape / grid_anchor_indices) and for the
weighted Voronoi vertices, which must sit at the peaks of that field.

Run with: .venv/bin/python tests/test_power_field.py
"""
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from periodica import _periodica
from periodica.core import grid_anchor_indices, power_grid_shape, sample_power_field


# ---- 1. cubic lattice, single unweighted site: pi = squared distance to Z^3

def test_cubic_single_site():
    N = 16
    U = np.eye(3)
    vals = sample_power_field(U, np.zeros((3, 1)), np.zeros(1), (N, N, N))
    # per-axis wrapped distance to the lattice, closed form
    ax = np.minimum(np.arange(N), N - np.arange(N)) / N
    ref = (ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    assert np.allclose(vals, ref, atol=1e-12), np.abs(vals - ref).max()
    # the cube's Voronoi vertex (.5,.5,.5) has pi = 3/4
    assert np.isclose(vals[N // 2, N // 2, N // 2], 0.75)
    anchor = grid_anchor_indices(U, (N, N, N), np.full((3, 1), 0.5))
    assert anchor[0] == np.ravel_multi_index((N // 2, N // 2, N // 2), (N, N, N))
    print('PASS cubic single site (closed form, Voronoi vertex, anchor)')


# ---- 2. weighted on-grid sites: field value at a site anchor is -w ----

def test_weighted_sites():
    N = 16
    U = np.eye(3)
    sites = np.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]])
    w = np.array([0.3, 0.1])
    vals = sample_power_field(U, sites, w, (N, N, N))
    assert np.isclose(vals[0, 0, 0], -0.3)
    assert np.isclose(vals[N // 2, N // 2, N // 2], -0.1)
    anchors = grid_anchor_indices(U, (N, N, N), sites)
    flat = vals.ravel()
    assert np.allclose(flat[anchors], -w)
    print('PASS weighted sites (pi(site) = -w)')


# ---- 3. skewed lattice vs brute force: catches an insufficient shift set --

def test_skewed_brute_force():
    rng = np.random.default_rng(3)
    for trial in range(5):
        while True:
            U = rng.uniform(-1, 1, (3, 3))
            if abs(np.linalg.det(U)) > 0.2:
                break
        n = 5
        sites = U @ rng.uniform(0, 1, (3, n))
        w = rng.uniform(0, 0.2, n)
        shape = (6, 7, 8)
        vals = sample_power_field(U, sites, w, shape)

        N = np.asarray(shape)
        idx = np.indices(shape).reshape(3, -1)
        X = U @ (idx / N[:, None])
        best = np.full(X.shape[1], np.inf)
        for z in product(range(-3, 4), repeat=3):
            off = U @ np.asarray(z, dtype=float)
            for i in range(n):
                c = sites[:, i] + off
                dd = ((X - c[:, None]) ** 2).sum(axis=0) - w[i]
                best = np.minimum(best, dd)
        assert np.allclose(vals.ravel(), best, atol=1e-9), \
            (trial, np.abs(vals.ravel() - best).max())
    print('PASS skewed lattices vs brute force (5 trials)')


# ---- 4. power_grid_shape budget and clipping ----

def test_grid_shape():
    N = power_grid_shape(np.eye(3))
    assert len(set(N)) == 1 and 8 <= N[0] <= 96
    assert np.prod(N) <= 160_000 * 1.25
    # anisotropic: long axis gets more samples, capped at 96
    N = power_grid_shape(np.diag([16.0, 3.2, 3.2]))
    assert N[0] == max(N) and N[0] <= 96 and min(N) >= 8
    assert np.prod(N) <= 160_000 * 1.25
    # tiny budget respects the floor
    N = power_grid_shape(np.eye(3), budget=10)
    assert N == (8, 8, 8)
    print('PASS power_grid_shape (budget, clipping)')



# ---- 5. weighted Voronoi vertices are power centers (peaks of pi) ----

def brute_force_power(U, sites, w, x, reach=4):
    """Power distances from x to every site copy within `reach` lattice steps."""
    d = U.shape[0]
    vals = []
    for z in product(range(-reach, reach + 1), repeat=d):
        c = sites + (U @ np.asarray(z, dtype=float))[:, None]
        vals.append(((x[:, None] - c) ** 2).sum(axis=0) - w)
    return np.concatenate(vals)


def test_voronoi_vertex_closed_form():
    # square lattice, corner site with weight a and an unweighted center site:
    # the triangle (0,0),(1,0),(.5,.5) has power center (.5, a) -- its
    # circumcenter (.5, 0) is only correct for a = 0
    a = 0.1
    U = np.eye(2)
    sites = np.array([[0.0, 0.5], [0.0, 0.5]])
    w = np.array([a, 0.0])
    vp, _, pf, _, _ = _periodica.periodic_voronoi(U, sites, w, True)
    got = sorted(map(tuple, np.round(vp.T % 1.0, 9)))
    want = sorted([(0.5, a), (0.5, 1 - a), (a, 0.5), (1 - a, 0.5)])
    assert np.allclose(got, want, atol=1e-9), (got, want)
    assert np.allclose(pf, -(0.5 - a) ** 2, atol=1e-9), pf
    print('PASS weighted Voronoi vertices (closed form on the square lattice)')


def test_voronoi_vertices_at_power_peaks():
    for d in (2, 3):
        for weighted in (False, True):
            for seed in range(4):
                rng = np.random.default_rng(seed)
                while True:
                    U = rng.uniform(-1, 1, (d, d))
                    if abs(np.linalg.det(U)) > 0.3:
                        break
                n = 6
                sites = U @ rng.uniform(0, 1, (d, n))
                w = rng.uniform(0, 0.05, n) if weighted else np.zeros(n)
                vp, _, pf, _, _ = _periodica.periodic_voronoi(U, sites, w, True)
                for k in range(vp.shape[1]):
                    vals = brute_force_power(U, sites, w, vp[:, k])
                    pi = vals.min()
                    # the vertex value is the field value there (negated for
                    # the Voronoi filtration) ...
                    assert abs(pi + pf[k]) < 1e-6, (d, weighted, seed, k, pi, pf[k])
                    # ... and at least d+1 sites are tied for the minimum
                    assert (vals < pi + 1e-6).sum() >= d + 1, (d, weighted, seed, k)
                # the 3x skeleton uses the same centers
                full_pts, _ = _periodica.full_voronoi(U, sites, w, True)
                for k in range(vp.shape[1]):
                    gap = np.abs(full_pts - vp[:, [k]]).sum(axis=0).min()
                    assert gap < 1e-9, (d, weighted, seed, k, gap)
    print('PASS Voronoi vertices at power-field peaks (2D/3D, weighted and not)')


if __name__ == '__main__':
    test_cubic_single_site()
    test_weighted_sites()
    test_skewed_brute_force()
    test_grid_shape()
    test_voronoi_vertex_closed_form()
    test_voronoi_vertices_at_power_peaks()
    print('All power-field tests passed.')
