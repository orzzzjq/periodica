"""FastAPI server exposing the Periodica pipeline to the web frontend.

Run from the repo root:
    .venv/bin/uvicorn app:app --app-dir web/server --reload --port 8000

All geometry is resolved into ready-to-draw coordinates server-side so the
frontend contains no index/shift logic.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('MPLBACKEND', 'Agg')  # core.py imports pyplot at module level

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.spatial import ConvexHull

from periodica import _periodica
from periodica.core import Periodica

app = FastAPI(title='Periodica API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class ComputeRequest(BaseModel):
    d: int = Field(ge=2, le=3)
    lattice: list[list[float]]
    points: list[list[float]]  # n rows of d coordinates
    weights: list[float]
    imageSize: int = Field(default=100, ge=10, le=400)


def polytope_2d(vertices):
    """Ordered outline loop of a 2D convex polytope."""
    hull = ConvexHull(vertices)
    loop = [vertices[i].tolist() for i in hull.vertices]
    return {'vertices': vertices.tolist(), 'outline': loop}


def polytope_3d(vertices):
    """Triangles plus true (non-coplanar-facet) edges of a 3D convex polytope.

    Same edge dedup as Periodica.draw_polytope: an edge shared only by
    coplanar hull facets is a triangulation artifact, not a polytope edge.
    """
    hull = ConvexHull(vertices)
    triangles = [simplex.tolist() for simplex in hull.simplices]

    edge_to_facets = {}
    for facet_idx, simplex in enumerate(hull.simplices):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(simplex[i]), int(simplex[j]))))
            edge_to_facets.setdefault(edge, []).append(facet_idx)

    edges = []
    for edge, facets in edge_to_facets.items():
        if len(facets) <= 1:
            edges.append(list(edge))
        else:
            ref_eq = hull.equations[facets[0]]
            if any(not np.allclose(ref_eq, hull.equations[f], atol=1e-9, rtol=1e-6)
                   for f in facets[1:]):
                edges.append(list(edge))

    return {'vertices': vertices.tolist(), 'edges': edges, 'triangles': triangles}


def polytope(d, vertices):
    return polytope_2d(vertices) if d == 2 else polytope_3d(vertices)


@app.post('/api/compute')
def compute(req: ComputeRequest):
    d = req.d
    U = np.array(req.lattice, dtype=float)
    points = np.array(req.points, dtype=float).T if req.points else np.zeros((d, 0))
    weights = np.array(req.weights, dtype=float)

    if U.shape != (d, d):
        raise HTTPException(400, f'lattice must be {d}x{d}')
    if abs(np.linalg.det(U)) < 1e-12:
        raise HTTPException(400, 'lattice basis is singular')
    if points.shape[0] != d or points.shape[1] == 0:
        raise HTTPException(400, f'need at least one point with {d} coordinates')
    if weights.shape != (points.shape[1],):
        raise HTTPException(400, 'weights length must equal number of points')
    if (weights < 0).any():
        raise HTTPException(400, 'weights must be non-negative')

    p = Periodica()
    p.set_geometry({'d': d, 'U': U, 'n_points': points.shape[1],
                    'points': points, 'weights': weights})
    try:
        p.quotient_complex('delaunay')
        p.merge_tree()
        barcodes = p.barcodes()
        images = p.images(req.imageSize)

        V = p.V
        A, b = _periodica.dirichlet_domain(V)
        P, I, _shifts = _periodica.points_in_3x_domain(V, A, b, points)
        P_full, full_edges = _periodica.full_delaunay(U, points, weights)
    except HTTPException:
        raise
    except Exception as e:  # geometric degeneracies etc.
        raise HTTPException(400, f'computation failed: {e}')

    # Quotient arcs resolved to coordinates (same mapping as plot_delaunay)
    kept = p.kept_points
    arcs = []
    for i in range(len(p.quotient_arcs)):
        s, t = p.quotient_arcs[i]
        shift = p.quotient_arc_shift[:, i]
        start = P_full[:, kept[s]]
        end = P_full[:, kept[t]] + V[:, :-1] @ shift
        arcs.append({
            'start': start.tolist(),
            'end': end.tolist(),
            'filtration': float(p.quotient_arc_filtration[i]),
            'shift': shift.tolist(),
        })

    finite_filtrations = [a['filtration'] for a in arcs if np.isfinite(a['filtration'])]
    # Slider bound: with large weights every filtration can be negative
    # (signed-sqrt radius scale), so fall back to the magnitudes.
    max_radius = max((abs(f) for f in finite_filtrations), default=1.0) or 1.0

    def encode_bar(bar):
        birth, death, multiplicity = bar
        return {
            'birth': float(birth),
            'death': None if np.isinf(death) else float(death),  # JSON has no Infinity
            'multiplicity': float(multiplicity),
        }

    # x-range of the persistence images, matching Periodica.images()
    xmin = min(min(bar[0] for bar in bars) for bars in barcodes)
    xmax = max(max(bar[1] if np.isfinite(bar[1]) else bar[0] for bar in bars) for bars in barcodes)
    xspan = xmax - xmin
    img_xmin, img_xmax = xmin - 0.12 * xspan, xmax + 0.12 * xspan

    return {
        'd': d,
        'basis': V[:, :d].T.tolist(),  # basis vectors as rows
        'domain1x': polytope(d, p.domain_vertices(A, b)),
        'domain3x': polytope(d, p.domain_vertices(A, b * 3)),
        'points': {
            'positions3x': P_full.T.tolist(),
            'originalIndex': np.asarray(I).reshape(-1).tolist(),
            'canonicalCount': int(points.shape[1]),
            'kept': kept.tolist(),
            'hidden': list(p.hidden_points),
            'weights': weights.tolist(),
        },
        'fullEdges': np.asarray(full_edges).tolist(),
        'quotientArcs': arcs,
        'maxRadius': float(max_radius),
        'barcodes': [[encode_bar(bar) for bar in bars] for bars in barcodes],
        'images': {
            'size': req.imageSize,
            'xmin': float(img_xmin),
            'xmax': float(img_xmax),
            'data': [np.asarray(img).tolist() for img in images],
        },
    }


# Serve the built frontend when it exists (production single-server mode)
_dist = os.path.join(REPO_ROOT, 'web', 'frontend', 'dist')
if os.path.isdir(_dist):
    app.mount('/', StaticFiles(directory=_dist, html=True), name='frontend')
