import sys
sys.path.insert(1, './build/Release')

import periodica
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import time
from itertools import combinations
import matplotlib.animation as animation

red = '#EE0E0E'
blue = '#0E0EEE'
green = '#0EEE0E'

# # ------------ for debug
# from scipy.spatial import Delaunay

# def scipy_delaunay(points_2xn):
#     tri = Delaunay(points_2xn.T)
    
#     edges = set()
#     for simplex in tri.simplices:
#         edges.add(tuple(sorted([simplex[0], simplex[1]])))
#         edges.add(tuple(sorted([simplex[1], simplex[2]])))
#         edges.add(tuple(sorted([simplex[2], simplex[0]])))
    
#     edges_array = np.array(list(edges), dtype=np.int32)
    
#     return edges_array
# # ------------ for debug

def test_delaunay_mst(n, d):
    def generate_random_points(n, d):
        points = np.random.rand(d, n)
        return points

    points = generate_random_points(n, d)
    delaunay_edges = periodica.delaunay_skeleton(points)
    emst_edges = periodica.euclidean_mst(points)

    fig = plt.figure()

    if d == 2:
        ax = fig.add_subplot()

        ax.scatter(*points, color='k', s=10)

        for s, t in delaunay_edges:
            ax.plot(*points[:,(s,t)], lw=1, color='k', alpha=0.2)

        for s, t in emst_edges:
            ax.plot(*points[:,(s,t)], lw=2, color=red)

        ax.set_aspect(1)

    # # ------------ for debug
    #     ax2 = fig.add_subplot(122)
    #     ax2.scatter(*points, color='k', s=10)
    #     for s, t in scipy_delaunay(points):
    #         ax2.plot(*points[:,(s,t)], lw=1, color='k', alpha=0.3)
    #     ax2.set_aspect(1)
    # # ------------ for debug

    else:
        ax = fig.add_subplot(projection='3d')

        ax.scatter(*points, color='k', s=10)

        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
        # ax.set_zlim([0, 1])

        for s, t in delaunay_edges:
            plt.plot(*points[:,(s,t)], lw=1, color='k', alpha=0.2)

        for s, t in emst_edges:
            plt.plot(*points[:,(s,t)], lw=2, color=red)

        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

    plt.show()

def test_periodic_delaunay(d):
    def get_domain_vertices(A, b):
        res = []
        if d == 2:
            for i in range(A.shape[0]):
                for j in range(i + 1, A.shape[0]):
                    _A = A[(i,j),:]
                    _b = np.array([b[i], b[j]])
                    if np.linalg.matrix_rank(_A) == d:
                        v = np.linalg.solve(_A, _b)
                        c = A @ v / b
                        if c.max() <= 1 + 1e-9:
                            res.append(v)
        else:
            for i in range(A.shape[0]):
                for j in range(i + 1, A.shape[0]):
                    for k in range(j + 1, A.shape[0]):
                        _A = A[(i,j,k),:]
                        _b = np.array([b[i], b[j], b[k]])
                        if np.linalg.matrix_rank(_A) == d:
                            v = np.linalg.solve(_A, _b)
                            c = A @ v / b
                            if c.max() < 1 + 1e-9:
                                res.append(v)
        return np.array(res)

    def draw_cell_boundary(U, ax, color=green):
        cell_vertices = []
        for mask in range(1<<d):
            v = np.zeros(d)
            for i in range(d):
                if mask & (1 << i):
                    v += U[:,i]
            cell_vertices.append(v)
        cell_vertices = np.array(cell_vertices)
        edges = []
        for i in range(1<<d):
            for j in range(i+1, 1<<d):
                if (i^j).bit_count() == 1:
                    edges.append((i, j))

        for e in edges:
            ax.plot(*cell_vertices[e,:].T, color=color, lw=1)
    
    def draw_polytope(A, b, ax, color='k', lw=1, ls='-', alpha=1):
        domain_vertices = get_domain_vertices(A, b)
        hull = ConvexHull(domain_vertices)

        if A.shape[1] == 2:
            for simplex in hull.simplices:
                ax.plot(domain_vertices[simplex, 0], domain_vertices[simplex, 1], color=color, lw=lw, ls=ls, alpha=alpha)
        else:
            for simplex in hull.simplices:
                ax.plot(domain_vertices[simplex, 0], domain_vertices[simplex, 1], domain_vertices[simplex, 2], color=color, lw=lw, ls=ls, alpha=alpha)
                ax.plot(domain_vertices[[simplex[-1], simplex[0]], 0], domain_vertices[[simplex[-1], simplex[0]], 1], domain_vertices[[simplex[-1], simplex[0]], 2], color=color, lw=lw, ls=ls, alpha=alpha)
                ax.plot_trisurf(*domain_vertices[simplex].T, linewidth=0, color=color, antialiased=True, alpha=0.1)
    
    def get_canonical_points(U, d, points, A, b):
        coeff = []
        res = []
        idx = set()
        def dfs(U, A, b):
            nonlocal coeff, res
            if len(coeff) == d:
                for j in range(points.shape[1]):
                    v = points[:,j].copy()
                    for i in range(d):
                        v += U[:, i] * coeff[i]
                    if (A @ v - b).max() <= 1e-5:
                        res.append(v)
                        idx.add(j)
                return
            for c in range(-1, 2):
                coeff.append(c)
                dfs(U, A, b)
                coeff.pop(-1)
        dfs(U, A, b)
        if len(idx) != points.shape[1]:
            print(f'Missing point: {len(idx)}/{len(points)}')
        return np.array(res)
    
    global visual

    n = 10

    # U = np.identity(d)
    U = np.random.rand(d, d) * 2 - 1

    # print(f'Orignial basis:\n{U}')

    points = U @ np.random.rand(d, n)

    fig = plt.figure()

    if d == 2:
        t1 = time.perf_counter()

        rU = periodica.reduced_basis(U)

        A, b =  periodica.dirichlet_domain(rU)
        
        canonical_points = periodica.canonical_points(A, b, points)

        working_points, I, S = periodica.points_in_3x_domain(rU, A, b, canonical_points)

        delaunay_edges = periodica.delaunay_skeleton(working_points)

        print(f'Running time: {time.perf_counter() - t1} s')
        
        if visual:
            ax = fig.add_subplot()

            # draw_cell_boundary(U, ax, blue)
            draw_cell_boundary(rU[:,:-1], ax, green)
            draw_polytope(A, b, ax, lw=1, alpha=1, ls='--')
            draw_polytope(A, b * 3, ax, lw=1, ls='-', alpha=1)

            ax.scatter(*working_points[:,canonical_points.shape[1]:], color=blue, s=5)

            ax.scatter(*canonical_points, color=red, s=5)
            
            E = []
            for s, t in delaunay_edges:
                color = red if s < canonical_points.shape[1] or t < canonical_points.shape[1] else 'k'
                alpha = 0.5 if s < canonical_points.shape[1] or t < canonical_points.shape[1] else 0.2
                ax.plot(*working_points[:,(s,t)], lw=1, color=color, alpha=alpha)
                if s < canonical_points.shape[1] or t < canonical_points.shape[1]:
                    if s > t:
                        z = s 
                        s = t
                        t = z
                    E.append((I[s], I[t]))

            ax.set_aspect(1)
    
    else:

        # draw_cell_boundary(U, ax, blue)
        t1 = time.perf_counter()

        rU = periodica.reduced_basis(U)

        A, b =  periodica.dirichlet_domain(rU)
        
        canonical_points = periodica.canonical_points(A, b, points)

        working_points, I, S = periodica.points_in_3x_domain(rU, A, b, canonical_points)

        delaunay_edges = periodica.delaunay_skeleton(working_points)

        mst_edges = periodica.euclidean_mst(working_points)

        print(f'Running time: {time.perf_counter() - t1} s')

        if visual:
            ax = fig.add_subplot(projection='3d')

            # draw_cell_boundary(rU[:,:-1], ax, green)
            draw_polytope(A, b * 3, ax, lw=1, ls='-', alpha=0.5)

            ax.scatter(*working_points[:,canonical_points.shape[1]:], color=blue, s=5)

            ax.scatter(*canonical_points, color=red, s=5)

            E = []
            for s, t in delaunay_edges:
                if s >= canonical_points.shape[1] and t >= canonical_points.shape[1]:
                    continue
                color = red if s < canonical_points.shape[1] or t < canonical_points.shape[1] else blue
                alpha = 0.5 if s < canonical_points.shape[1] or t < canonical_points.shape[1] else 0.2
                ax.plot(*working_points[:,(s,t)], lw=1, color=color, alpha=alpha)
                if s < canonical_points.shape[1] or t < canonical_points.shape[1]:
                    if s > t:
                        z = s 
                        s = t
                        t = z
                    E.append((I[s], I[t]))
            
            limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
            ax.set_box_aspect(np.ptp(limits, axis = 1))
        
        if animation:
            gif = animation.FuncAnimation(fig, lambda x: ax.view_init(azim=x), frames=np.arange(0, 362, 2), interval=100)
            gif.save('rotation.gif', dpi=80, writer='imagemagick')

    if visual:
        plt.show()

def test_merge_tree(d):    
    n = 2
    U = np.random.rand(d, d) * 2 - 1
    points = U @ np.random.rand(d, n)
    
    V = periodica.reduced_basis(U)

    edges, filtration, shift = periodica.periodic_delaunay(U, points)

    pmt = periodica.merge_tree(n, d, V, edges, filtration, shift)

    periodica.print_merge_tree(pmt)

    barcode = periodica.barcode(d, pmt)

    for lst in barcode:
        print(lst)


visual = True
animation = False

# test_periodic_delaunay(2)
test_merge_tree(3)

