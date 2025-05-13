import periodica
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import time
from itertools import combinations

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

def test_lattice_diameter(d):
    U = np.identity(d)
    # U = np.random.rand(d, d)
    diameter = periodica.lattice_diameter(U)

    fig = plt.figure()

    if d == 2:
        ax = fig.add_subplot()

        vertices = []
        for mask in range(1<<d):
            v = np.zeros(d)
            for i in range(d):
                if mask & (1 << i):
                    v += U[:,i]
            vertices.append(v)

        vertices = np.array(vertices)

        edges = []
        for i in range(1<<d):
            for j in range(i+1, 1<<d):
                if (i^j).bit_count() == 1:
                    edges.append((i, j))

        for e in edges:
            ax.plot(*vertices[e,:].T, color=green, lw=1)

        center = U @ np.ones(2) / 2
        circle = plt.Circle(center, diameter/2, color=blue, lw=1, fill=False)
        ax.add_patch(circle)

        ax.set_aspect(1)

    else:
        ax = fig.add_subplot(projection='3d')

        vertices = []
        for mask in range(1<<d):
            v = np.zeros(d)
            for i in range(d):
                if mask & (1 << i):
                    v += U[:,i]
            vertices.append(v)

        vertices = np.array(vertices)

        edges = []
        for i in range(1<<d):
            for j in range(i+1, 1<<d):
                if (i^j).bit_count() == 1:
                    edges.append((i, j))

        for e in edges:
            ax.plot(*vertices[e,:].T, color=green, lw=1)

        center = U @ np.ones(3) / 2
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v) * diameter * 3 / 2 + center[0]
        y = np.sin(u) * np.sin(v) * diameter * 3 / 2 + center[1]
        z = np.cos(v) * diameter * 3 / 2 + center[2]
        
        ax.plot_surface(x, y, z, color=blue, alpha=0.2)

        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

    plt.show()

def test_dirichlet_domain(d):
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
    
    def draw_polytope(A, b, ax, color='k', lw=1):
        domain_vertices = get_domain_vertices(A, b)
        hull = ConvexHull(domain_vertices)

        if A.shape[1] == 2:
            for simplex in hull.simplices:
                ax.plot(domain_vertices[simplex, 0], domain_vertices[simplex, 1], color=color)
        else:
            for simplex in hull.simplices:
                ax.plot(domain_vertices[simplex, 0], domain_vertices[simplex, 1], domain_vertices[simplex, 2], color=color, lw=lw, alpha=0.2)
                ax.plot(domain_vertices[[simplex[-1], simplex[0]], 0], domain_vertices[[simplex[-1], simplex[0]], 1], domain_vertices[[simplex[-1], simplex[0]], 2], color=color, lw=lw, alpha=0.2)

    def get_line_points(a, b, c, ref = [-1., 1.]):
        return np.array([[x, (c - a * x) / b] for x in ref])
    
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

    def reduce2d(b):
        # 2D equivalent of Selling's algorithm
        coeffs = np.zeros((3,3))
        reduced = True
        for i,j,h in [[0,1,2], [0,2,1], [1,2,0]]:
            coeffs[i,j] = np.dot(b[i], b[j])
            if coeffs[i,j] > 0:
                reduced = False
                break

        while not reduced:
            b[h] += b[i]
            b[i] = -b[i]
            b[2] = - b[1] - b[0]

            reduced = True
            # lattice is reduced if all coefficients are negative
            for i,j,h in [[0,1,2], [0,2,1], [1,2,0]]:
                coeffs[i,j] = np.dot(b[i], b[j])
                if coeffs[i,j] > 0:
                    reduced = False
                    break

        return b

    def reduce2dnew(b):
        # 2D equivalent of Selling's algorithm
        coeffs = np.zeros((3,3))
        reduced = True
        for i,j,h in [[0,1,2], [0,2,1], [1,2,0]]:
            coeffs[i,j] = np.dot(b[i], b[j])
            if coeffs[i,j] > 0:
                reduced = False
                break

        while not reduced:
            for i, j in [[0,0],[1,1],[0,1]]:
                coeffs[i,j] = np.dot(b[i], b[j])
            
            if 2 * abs(coeffs[0,1]) <= coeffs[0,0] and coeffs[0,0] <= coeffs[1,1]:
                # Lagrange-reduced
                b[1] = -b[1]
                b[2] = -b[0]-b[1]
                print(f'b2: {b[2]}')
                # break
            else:
                s = 1 if coeffs[0,1] > 0 else -1
                b[1] = b[1] - s * b[0]
                b[2] = -b[0]-b[1]
                print(f'b2: {b[2]}')

            # b[h] += b[i]
            # b[i] = -b[i]

            reduced = True
            # lattice is reduced if all coefficients are negative
            for i,j,h in [[0,1,2], [0,2,1], [1,2,0]]:
                coeffs[i,j] = np.dot(b[i], b[j])
                if coeffs[i,j] > 0:
                    reduced = False
                    break
        
        # print('coeffs:')
        # print(coeffs)

        return b

    def check_reduced(b):
        # Check that any pair of basis vectors enclose an obtuse angle.
        # Otherwise the basis is not reduced.
        for v1,v2 in combinations(b, 2):
            scal_prod = np.dot(v1,v2)
            magn_prod = np.linalg.norm(v1)*np.linalg.norm(v2)
            angle = np.arccos(scal_prod/magn_prod)
            # print(angle/np.pi*180, scal_prod)
            if angle < np.pi/2:
                print("Warning: Lattice basis not reduced!")
    
    n = 10

    # U = np.identity(d)
    U = np.random.rand(d, d) * 2 - 1
    # U = np.array([[0.1, 1], [0, 1]])
    # U = np.array([[-0.1359805, 0.16300457], [-0.84394018, 0.88592977]])

    print(f'Orignial basis:\n{U}')

    points = U @ np.random.rand(d, n)

    fig = plt.figure()

    if d == 2:
        ax = fig.add_subplot()

        # ax.scatter(*points, color='k', s=5)

        draw_cell_boundary(U, ax, blue)

        rU = periodica.reduced_basis(U)

        # bb = [U[:,i] for i in range(U.shape[1])]
        # bb.append(-sum(bb))

        # print(f'Extended basis:\n{np.array(bb).T}')
        
        # rbb = np.array(reduce2d(bb)).T

        # print(f'Reduced basis:\n{np.array(reduce2d(bb)).T}')
        # print(f'Reduced basis:\n{rbb}')

        check_reduced([rU[:,i] for i in range(rU.shape[1])])

        # rU = rbb

        # print(rU)

        draw_cell_boundary(rU[:,:-1], ax, green)

        # for multiplicity in [1]:
        A, b =  periodica.dirichlet_domain(rU[:,:-1])
        draw_polytope(A, b, ax, lw=0.1)
        
        canonical_points = periodica.canonical_points(A, b, points)

        draw_polytope(A, b * 3, ax, lw=0.5)

        working_points = periodica.domain_x3_points(rU[:,:-1], A, b, canonical_points)

        # print(working_points[0])
        # print(working_points[1])

        ax.scatter(*working_points[0][:,canonical_points.shape[1]:], color=blue, s=5)

        ax.scatter(*canonical_points, color=red, s=5)

        delaunay_edges = periodica.delaunay_skeleton(working_points[0])
        
        # mst_edges = periodica.euclidean_mst(working_points[0])
      
        for s, t in delaunay_edges:
            color = red if s < canonical_points.shape[1] or t < canonical_points.shape[1] else 'k'
            alpha = 0.5 if s < canonical_points.shape[1] or t < canonical_points.shape[1] else 0.2
            ax.plot(*working_points[0][:,(s,t)], lw=1, color=color, alpha=alpha)

        # for s, t in mst_edges:
        #     if s < canonical_points.shape[1] or t < canonical_points.shape[1]:
        #         ax.plot(*working_points[0][:,(s,t)], lw=1.5, color=red, alpha=1)

        ax.set_aspect(1)
    
    else:

        # draw_cell_boundary(U, ax, blue)
        t1 = time.perf_counter()

        rU = periodica.reduced_basis(U)

        A, b =  periodica.dirichlet_domain(rU[:,:-1])
        
        canonical_points = periodica.canonical_points(A, b, points)

        working_points = periodica.domain_x3_points(rU[:,:-1], A, b, canonical_points)

        delaunay_edges = periodica.delaunay_skeleton(working_points[0])

        mst_edges = periodica.euclidean_mst(working_points[0])

        print(f'Running time: {time.perf_counter() - t1} s')

        ax = fig.add_subplot(projection='3d')

        # draw_cell_boundary(rU[:,:-1], ax, green)
        draw_polytope(A, b * 3, ax)

        ax.scatter(*working_points[0][:,canonical_points.shape[1]:], color=blue, s=5)

        ax.scatter(*canonical_points, color=red, s=5)

        for s, t in delaunay_edges:
            if s >= canonical_points.shape[1] and t >= canonical_points.shape[1]:
                continue
            color = red if s < canonical_points.shape[1] or t < canonical_points.shape[1] else blue
            alpha = 0.5 if s < canonical_points.shape[1] or t < canonical_points.shape[1] else 0.2
            ax.plot(*working_points[0][:,(s,t)], lw=1, color=color, alpha=alpha)
        
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

    plt.show()

test_dirichlet_domain(3)
