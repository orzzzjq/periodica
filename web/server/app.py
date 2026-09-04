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
    # set_geometry perturbs the points to break exact degeneracies; use the
    # perturbed copy for every native call so the whole response is consistent
    points = p.points
    try:
        p.quotient_complex('delaunay')
        p.merge_tree()
        barcodes = p.barcodes()
        images = p.images(req.imageSize)

        V = p.V
        A, b = _periodica.dirichlet_domain(V)
        # tile the CANONICAL points: full_delaunay canonicalizes internally,
        # so this keeps I aligned with P_full's copy order (a raw-points
        # tiling can enumerate the copies differently)
        canonical = _periodica.canonical_points(A, b, points)
        P, I, _shifts = _periodica.points_in_3x_domain(V, A, b, canonical)
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
            # quotient vertex indices of the endpoints (same index space as
            # the merge tree beams), for subtree-linked filtering
            'vStart': int(s),
            'vEnd': int(t),
        })

    finite_filtrations = [a['filtration'] for a in arcs if np.isfinite(a['filtration'])]
    # Slider bound fallback: filtration values are on the power-distance scale
    # and can all be negative with large weights, so use the magnitudes.
    max_radius = max((abs(f) for f in finite_filtrations), default=1.0) or 1.0

    def encode_bar(bar):
        birth, death, multiplicity = bar
        return {
            'birth': float(birth),
            'death': None if np.isinf(death) else float(death),  # JSON has no Infinity
            'multiplicity': float(multiplicity),
        }

    def encode_tree(tree):
        # merge tree as event beams: per quotient vertex a time-sorted list of
        # (time, coeff, exponent, child) — child -1 = plain monomial event,
        # child == beam index = own death, else id of the beam merging in.
        return [
            [[None if np.isinf(t) else float(t), float(c), int(e), int(ch)]
             for (t, c, e, ch) in beam]
            for beam in tree
        ]

    def encode_descriptors(barcodes, images, tree):
        # x-range of the persistence images, matching Periodica.images();
        # a dimension can have no bars at all (e.g. dim 0 of a Voronoi complex)
        xmin = min(min(bar[0] for bar in bars) for bars in barcodes if bars)
        xmax = max(max(bar[1] if np.isfinite(bar[1]) else bar[0] for bar in bars) for bars in barcodes if bars)
        xspan = xmax - xmin
        return {
            'barcodes': [[encode_bar(bar) for bar in bars] for bars in barcodes],
            'images': {
                'size': req.imageSize,
                'xmin': float(xmin - 0.12 * xspan),
                'xmax': float(xmax + 0.12 * xspan),
                'data': [np.asarray(img).tolist() for img in images],
            },
            'tree': encode_tree(tree),
        }

    # Voronoi descriptors + scene geometry from a single periodic_voronoi call
    # (circumcenter cell centers); failures don't break the response.
    voronoi = None
    voronoi_error = None
    voronoi_geometry = None
    try:
        cvp, pv_edges, pv_pf, pv_ef, pv_shift = _periodica.periodic_voronoi(U, points, weights, True)

        # descriptors: feed the quotient complex into the Periodica pipeline
        q = Periodica()
        q.d = d
        q.V = V
        q.n_quotient_vertices = cvp.shape[1]
        q.quotient_vertex_filtration = pv_pf
        q.quotient_arcs = pv_edges
        q.quotient_arc_filtration = pv_ef
        q.quotient_arc_shift = pv_shift
        q.merge_tree()
        voronoi = encode_descriptors(q.barcodes(), q.images(req.imageSize), q.tree)

        # geometry overlay, mirroring plot_voronoi
        vor_pts, vor_edges = _periodica.full_voronoi(U, points, weights, True)
        varcs = []
        for i in range(pv_edges.shape[0]):
            s, t = int(pv_edges[i, 0]), int(pv_edges[i, 1])
            start = cvp[:, s]
            end = cvp[:, t] + V[:, :-1] @ pv_shift[:, i]
            varcs.append({
                'start': start.tolist(),
                'end': end.tolist(),
                'filtration': float(pv_ef[i]),
                # endpoint (Voronoi vertex) filtration values, for the
                # cone approximation of the Voronoi filtration
                'fStart': float(pv_pf[s]),
                'fEnd': float(pv_pf[t]),
                # quotient (canonical) vertex indices of the endpoints, so
                # per-vertex quantities stay identical across periodic copies
                'vStart': s,
                'vEnd': t,
            })
        voronoi_geometry = {
            'points3x': vor_pts.T.tolist(),
            'fullEdges': np.asarray(vor_edges).tolist(),
            'arcs': varcs,
        }
    except Exception as e:
        voronoi_error = str(e)

    delaunay_desc = encode_descriptors(barcodes, images, p.tree)

    return {
        'voronoi': voronoi,
        'voronoiError': voronoi_error,
        'voronoiGeometry': voronoi_geometry,
        'd': d,
        'basis': V[:, :d].T.tolist(),  # basis vectors as rows
        'domain1x': polytope(d, p.domain_vertices(A, b)),
        'domain3x': polytope(d, p.domain_vertices(A, b * 3)),
        # Halfspace representation of the Dirichlet domain: x is inside the
        # 1x domain iff A·x <= b, inside the 3x domain iff A·x <= 3b.
        'domainA': A.tolist(),
        'domainB': b.reshape(-1).tolist(),
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
        'barcodes': delaunay_desc['barcodes'],
        'images': delaunay_desc['images'],
        'tree': delaunay_desc['tree'],
    }


# Serve the built frontend when it exists (production single-server mode)
_dist = os.path.join(REPO_ROOT, 'web', 'frontend', 'dist')
if os.path.isdir(_dist):
    app.mount('/', StaticFiles(directory=_dist, html=True), name='frontend')
