import sys
sys.path.insert(1, './build/Release')

import re
import periodica
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import Polygon

from functools import wraps
from time import perf_counter

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print(f'func: {f.__name__} took: {te-ts} sec')
        return result
    return wrap

red = "#CD0000"
blue = "#7E7EFF"
green = "#30B830"

class Periodic:
    def generate_random_input(self, n, d):
        self.n = n      # number of points
        self.d = d      # dimension
        # self.U = np.random.rand(d, d) * 2 - 1       # original basis
        self.U = np.array([[0, np.sqrt(3)/2], [1, -1/2]])
        np.random.seed(4)
        self.points = self.U @ np.random.rand(d, n) # points in unit cell

    def periodic_delaunay(self):
        if not hasattr(self, 'points'):
            raise Exception('No input points')
        self.V = periodica.reduced_basis(self.U)    # reduced basis
        self.quotient_arcs, self.quotient_arc_filtration, self.quotient_arc_shift = periodica.periodic_delaunay(self.U, self.points)

    def quotient_complex(self, complex_type='delaunay'):
        if complex_type == 'delaunay':
            self.periodic_delaunay()
        elif complex_type == 'voronoi':
            pass
        else:
            raise Exception(f'Does not support complex type {complex_type}')

    def load_point_set(self, file):
        pass

    def load_quotient_complex(self, file):
        with open(file, 'r') as f:
            # dimension
            f.readline()
            self.d = int(f.readline())
            # lattice
            f.readline()
            self.V = []
            for i in range(self.d):
                self.V.append(list(map(float, f.readline().split(' '))))
            self.V = np.array(self.V)
            # vertices
            f.readline()
            self.n = int(f.readline())
            self.quotient_vertex_filtration = []
            for i in range(self.n):
                self.quotient_vertex_filtration.append(float(f.readline().split(' ')[-1]))
            self.quotient_vertex_filtration = np.array(self.quotient_vertex_filtration)
            # arcs
            f.readline()
            m = int(f.readline())
            self.quotient_arcs = []
            self.quotient_arc_filtration = []
            self.quotient_arc_shift = []
            for i in range(m):
                line = f.readline().split(' ')
                self.quotient_arcs.append(list(map(int, line[:2])))
                self.quotient_arc_filtration.append(float(line[2]))
                self.quotient_arc_shift.append(list(map(int, line[3:])))
            self.quotient_arcs = np.array(self.quotient_arcs)
            self.quotient_arc_filtration = np.array(self.quotient_arc_filtration)
            self.quotient_arc_shift = np.array(self.quotient_arc_shift).T
        
        # print(f'basis:\n{self.V}')
        # print(f'vertex filtration:\n{self.quotient_vertex_filtration}')
        # print(f'arcs:\n{self.quotient_arcs}')
        # print(f'arc filtration:\n{self.quotient_arc_filtration}')
        # print(f'arc shift:\n{self.quotient_arc_shift}')

    @timing
    def merge_tree(self):
        if not hasattr(self, 'quotient_arcs'):
            self.quotient_complex()
        if hasattr(self, 'quotient_vertex_filtration'):
            self.tree = periodica.merge_tree(self.n, self.d, self.V, self.quotient_arcs, self.quotient_arc_filtration, self.quotient_arc_shift, self.quotient_vertex_filtration)
        else:
            self.tree = periodica.merge_tree(self.n, self.d, self.V, self.quotient_arcs, self.quotient_arc_filtration, self.quotient_arc_shift)
        return self.tree

    def print_merge_tree(self):
        if not hasattr(self, 'tree'):
            self.merge_tree()
        periodica.print_merge_tree(self.tree)

    def barcodes(self):
        if not hasattr(self, 'tree'):
            self.merge_tree()
        self.bcodes = periodica.barcode(self.d, self.tree)
        return self.bcodes
    
    def images(self, size=100):
        if not hasattr(self, 'bcodes'):
            self.barcodes()

        inf = 1e+308
        xmin = min(map(lambda b: min(map(lambda x: x[0], b)), self.bcodes))
        xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), self.bcodes))
        xspan = xmax - xmin
        xmin, xmax = xmin - 0.12 * xspan, xmax + 0.12 * xspan
        
        self.persistence_images = []
        for i in range(self.d + 1):
            self.persistence_images.append(periodica.image(self.bcodes[i], size, xmin, xmax))
        return self.persistence_images

    def plot_barcodes(self, show=True):
        if not hasattr(self, 'bcodes'):
            self.barcodes()
        
        inf = 1e+308
        sep = 1
        labels = [r'$\cdot R^0$', r'$\cdot 2 R^1$', r'$\cdot\pi R^2$', r'$\cdot \frac{4\pi}{3}R^3$']

        fig, ax = plt.subplots(self.d + 1, 1)
        fig.set_size_inches(5, (self.d + 1) * 2)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)

        xmin = min(map(lambda b: min(map(lambda x: x[0], b)), self.bcodes))
        xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), self.bcodes))
        xspan = xmax - xmin
        xmin, xmax = xmin - 0.12 * xspan, xmax + 0.05 * xspan
        # print(f'xmin {xmin} xmax {xmax}')

        for i in range(self.d + 1):
            axi = ax[self.d + 1 - i - 1]
            N = len(self.bcodes[i])
            axi.set_xlim([xmin, xmax])
            ymin, ymax = -sep * (N - 1), 0
            ymin, ymax = ymin - sep, ymax + sep
            axi.set_ylim([ymin, ymax])
            axi.set_yticks([])
            axi.text(xmax - (xmax - xmin) * 0.01, ymax - (ymax - ymin) * 0.05, labels[i], horizontalalignment='right', verticalalignment='top')
            # print(f'dim-{i}: {N} bars | ymin {ymin} ymax {ymax}')
            for j in range(N):
                # print(f'{j}: {self.bcodes[i][j]}')
                birth, death, multiplicity = self.bcodes[i][j]
                y = j * -sep
                axi.plot([birth, death if death < inf else xmax], np.ones(2) * y, lw=2, color='k')
                axi.text(birth - xspan * 0.01, y, f'{multiplicity:.3f}', fontsize=8, horizontalalignment='right', verticalalignment='center')

        plt.get_current_fig_manager().set_window_title('Barcode')
        if show:
            plt.show()
        plt.savefig('barcode.svg')

    def plot_diagram(self, show=True):
        if not hasattr(self, 'bcodes'):
            self.barcodes()

        inf = 1e+308
        labels = [r'$\cdot R^0$', r'$\cdot 2 R^1$', r'$\cdot\pi R^2$', r'$\cdot \frac{4\pi}{3}R^3$']

        fig, ax = plt.subplots(self.d + 1, 1)
        fig.set_size_inches(5, (self.d + 1) * 2)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)

        xmin = min(map(lambda b: min(map(lambda x: x[0], b)), self.bcodes))
        xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), self.bcodes))
        xspan = xmax - xmin
        xmin, xmax = xmin - 0.12 * xspan, xmax + 0.12 * xspan
        xticks = np.linspace(0, xmax, 5)[:-1]
        # print(f'xmin {xmin} xmax {xmax}')

        for i in range(self.d + 1):
            axi = ax[self.d + 1 - i - 1]
            N = len(self.bcodes[i])
            axi.set_xlim([xmin, xmax])
            axi.set_ylim([xmin, xmax])
            axi.set_xticks(xticks, [f'{x:.1f}' for x in xticks], fontsize=9)
            axi.set_yticks(xticks, [f'{x:.1f}' for x in xticks], fontsize=9)
            axi.text(xmax - (xmax - xmin) * 0.05, xmax - (xmax - xmin) * 0.05, labels[i], horizontalalignment='right', verticalalignment='top')
            axi.plot([xmin,xmax], [xmin,xmax], lw=1, linestyle='--', color='gray', alpha=0.25)
            axi.set_aspect(1)
            # print(f'dim-{i}: {N} points')
            for j in range(N):
                # print(f'{j}: {self.bcodes[i][j]}')
                birth, death, multiplicity = self.bcodes[i][j]
                axi.scatter(birth, death if death < inf else xmax, s=2, color='k')
                axi.text(birth + xspan * 0.05, death if death < inf else xmax, f'{multiplicity:.3f}', fontsize=8, horizontalalignment='left', verticalalignment='center')

        plt.get_current_fig_manager().set_window_title('Diagram')
        if show:
            plt.show()
        plt.savefig('diagram.svg')
            
    def plot_images(self, same_range=True, show=True):
        if not hasattr(self, 'persistence_images'):
            self.images()

        inf = 1e+308
        labels = [r'$\cdot R^0$', r'$\cdot 2 R^1$', r'$\cdot\pi R^2$', r'$\cdot \frac{4\pi}{3}R^3$']

        fig, ax = plt.subplots(self.d + 1, 1)
        fig.set_size_inches(5, (self.d + 1) * 2)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)

        xmin = min(map(lambda b: min(map(lambda x: x[0], b)), self.bcodes))
        xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), self.bcodes))
        xspan = xmax - xmin
        xmin, xmax = xmin - 0.12 * xspan, xmax + 0.12 * xspan
        # print(f'xmin {xmin} xmax {xmax}')

        imgsize = self.persistence_images[0].shape[0]
        gap = (xmax - xmin) / imgsize

        xticks = np.linspace(0, xmax, 5)[:-1]
        xtickpos = [(x - xmin) // gap for x in xticks]

        vmax = max(map(lambda img: np.max(img), self.persistence_images))
        vmin = min(map(lambda img: np.min(img), self.persistence_images))
        vrange = max(vmax, -vmin)

        for i in range(self.d + 1):
            if not same_range:
                vmax = np.max(self.persistence_images[i])
                vmin = np.min(self.persistence_images[i])
                vrange = max(vmax, -vmin)

            axi = ax[self.d + 1 - i - 1]
            im = axi.imshow(self.persistence_images[i], origin='lower', vmin=-vrange, vmax=vrange, cmap='Spectral_r')# Create axes for colorbar
            
            axi.set_xticks(xtickpos, [f'{x:.1f}' for x in xticks], fontsize=9)
            axi.set_yticks(xtickpos, [f'{x:.1f}' for x in xticks], fontsize=9)

            # Add colorbar
            divider = make_axes_locatable(axi)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)

            if same_range:
                cbar.set_ticks([])

            axi.text(imgsize * 0.95, imgsize * 0.95, labels[i], horizontalalignment='right', verticalalignment='top')

        plt.get_current_fig_manager().set_window_title('Image')
        if show:
            plt.show()
        plt.savefig('image.svg')

    def domain_vertices(self, A, b):
        res = []
        if self.d == 2:
            for i in range(A.shape[0]):
                for j in range(i + 1, A.shape[0]):
                    _A = A[(i,j),:]
                    _b = np.array([b[i], b[j]])
                    if np.linalg.matrix_rank(_A) == self.d:
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
                        if np.linalg.matrix_rank(_A) == self.d:
                            v = np.linalg.solve(_A, _b)
                            c = A @ v / b
                            if c.max() < 1 + 1e-9:
                                res.append(v)
        return np.array(res)    

    def draw_polytope(self, A, b, ax, color='k', lw=1, ls='-', alpha=1, fill_color=False):
        domain_vertices = self.domain_vertices(A, b)
        hull = ConvexHull(domain_vertices)

        if A.shape[1] == 2:
            for simplex in hull.simplices:
                ax.plot(domain_vertices[simplex, 0], domain_vertices[simplex, 1], color=color, lw=lw, ls=ls, alpha=alpha)
            if fill_color:
                hull_polygon = Polygon(domain_vertices[hull.vertices], alpha=1, facecolor='#FBE5D6', 
                                    edgecolor='None', linewidth=2, label='Convex Hull', zorder=-1)
                ax.add_patch(hull_polygon)
        else:
            for simplex in hull.simplices:
                ax.plot(domain_vertices[simplex, 0], domain_vertices[simplex, 1], domain_vertices[simplex, 2], color=color, lw=lw, ls=ls, alpha=alpha)
                ax.plot(domain_vertices[[simplex[-1], simplex[0]], 0], domain_vertices[[simplex[-1], simplex[0]], 1], domain_vertices[[simplex[-1], simplex[0]], 2], color=color, lw=lw, ls=ls, alpha=alpha)
                ax.plot_trisurf(*domain_vertices[simplex].T, linewidth=0, color=color, antialiased=True, alpha=0.1)

    def draw_unit_cell(self, basis, ax, color=green):
        cell_vertices = []
        for mask in range(1<<self.d):
            v = np.zeros(self.d)
            for i in range(self.d):
                if mask & (1 << i):
                    v += basis[:,i]
            cell_vertices.append(v)
        cell_vertices = np.array(cell_vertices)
        edges = []
        for i in range(1<<self.d):
            for j in range(i+1, 1<<self.d):
                if (i^j).bit_count() == 1:
                    edges.append((i, j))
        for e in edges:
            ax.plot(*cell_vertices[e,:].T, color=color, lw=1)
    
    def plot_delaunay(self, show=True, animation_gif=None):
        if not hasattr(self, 'points'):
            raise Exception('No input points')
        if not hasattr(self, 'V'):
            self.V = periodica.reduced_basis(self.U)
        
        A, b =  periodica.dirichlet_domain(self.V)
        canonical_points = periodica.canonical_points(A, b, self.points)
        P, I, S = periodica.points_in_3x_domain(self.V, A, b, canonical_points)
        delaunay_edges = periodica.delaunay_skeleton(P)

        fig = plt.figure()

        if self.d == 2:
            ax = fig.add_subplot()

            # self.draw_unit_cell(self.V[:,:-1], ax, green)
            ax.arrow(0, 0, self.V[0,0], self.V[1,0], color=green, width=0.01, head_width=0.04)
            ax.arrow(0, 0, self.V[1,0], self.V[1,1], color=green, width=0.01, head_width=0.04)

            self.draw_polytope(A, b, ax, lw=1, alpha=1, ls='-', fill_color='b')
            self.draw_polytope(A, b * 3, ax, lw=0.75, ls='-', alpha=1)

            ax.scatter(*P[:,self.n:], color='k', s=5, zorder=1)
            ax.scatter(*canonical_points, color='k', s=5)
            
            for s, t in delaunay_edges:
                arc = False
                if s < self.n or t < self.n:
                    arc = True
                    if s > t:
                        s, t = t, s
                    if t >= self.n and s > I[t]:
                        arc = False
                color = '#0000FE' if arc else 'k'
                alpha = 0.8 if arc else 0.2
                lw = 1.5 if arc else 1
                ax.plot(*P[:,(s,t)], lw=lw, color=color, alpha=alpha, zorder=0)

            ax.set_aspect(1)
        
        else:
            ax = fig.add_subplot(projection='3d')

            self.draw_polytope(A, b * 3, ax, lw=1, ls='-', alpha=0.5)

            ax.scatter(*P[:,self.n:], color=blue, s=5)
            ax.scatter(*canonical_points, color=red, s=5)

            for s, t in delaunay_edges:
                arc = False
                if s < self.n or t < self.n:
                    arc = True
                    if s > t:
                        s, t = t, s
                    if t >= self.n and s > I[t]:
                        arc = False
                if not arc:
                    continue
                color = red if arc else blue
                alpha = 0.5 if arc else 0.2
                ax.plot(*P[:,(s,t)], lw=1, color=color, alpha=alpha)
            
            limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
            ax.set_box_aspect(np.ptp(limits, axis = 1))
        
            if animation_gif:
                gif = animation.FuncAnimation(fig, lambda x: ax.view_init(azim=x), frames=np.arange(0, 362, 2), interval=100)
                gif.save(animation_gif, dpi=80, writer='imagemagick')
        
        ax.set_xticks([])
        ax.set_yticks([])

        plt.get_current_fig_manager().set_window_title('Delaunay')
        if show:
            plt.show()
        plt.savefig('delaunay.svg')



periodic = Periodic()

# periodic.load_quotient_complex('examples/example_2d_1.txt')
# periodic.load_quotient_complex('examples/example_2d_2.txt')
# periodic.load_quotient_complex('examples/example_3d_1.txt')
periodic.generate_random_input(n=3, d=2)

# periodic.merge_tree()
# periodic.print_merge_tree()
periodic.plot_delaunay(show=False)

# periodic.images(size=50)
# periodic.plot_barcodes(show=False)
# periodic.plot_diagram(show=False)
# periodic.plot_images(same_range=True, show=False)
