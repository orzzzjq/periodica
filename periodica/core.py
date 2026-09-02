"""High-level Python API built on top of the native _periodica extension."""

try:
    from . import _periodica
except ImportError:
    import _periodica
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import Polygon
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

from functools import wraps
from time import perf_counter

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print(f'[TIME] {f.__name__} took: {te-ts:.3f} sec')
        return result
    return wrap

red = "#CD0000"
blue = "#7E7EFF"
green = "#30B830"

class Periodica:
    def set_geometry(self, INPUT):
        self.d = INPUT['d']
        self.U = INPUT['U']
        self.n_points = INPUT['n_points']
        self.points = INPUT['points']
        if 'weights' in INPUT.keys():
            self.weights = INPUT['weights']

    def set_weights(self, weights):
        self.weights = np.array(weights)

    @timing
    def periodic_delaunay(self):
        if not hasattr(self, 'points'):
            raise Exception('No input points')
        weights = getattr(self, 'weights', np.zeros(self.points.shape[1], dtype=float))
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != self.points.shape[1]:
            raise Exception('weights length must equal number of points')
        self.weights = weights
        self.V = _periodica.reduced_basis(self.U)    # reduced basis
        self.quotient_arcs, self.quotient_arc_filtration, self.quotient_arc_shift, kept = \
            _periodica.periodic_delaunay(self.U, self.points, self.weights)
        # Points hidden by larger weights are absent from the weighted triangulation
        # and contribute nothing to the filtration; the backend drops them and
        # remaps the arc endpoints to the surviving points.
        self.kept_points = np.asarray(kept, dtype=int).reshape(-1)
        self.hidden_points = sorted(set(range(self.n_points)) - set(self.kept_points.tolist()))
        if self.hidden_points:
            print(f'[WARNING] {len(self.hidden_points)} hidden point(s) dropped from weighted Delaunay: {self.hidden_points}')
        self.n_quotient_vertices = len(self.kept_points)
        # Power-distance scale: a vertex enters the filtration when its ball
        # {x : ||x - p_i||^2 - w_i <= f} appears, i.e. at f = -w_i.
        self.quotient_vertex_filtration = -self.weights[self.kept_points]

    @timing
    def periodic_voronoi(self):
        if not hasattr(self, 'points'):
            raise Exception('No input points')
        weights = getattr(self, 'weights', np.zeros(self.points.shape[1], dtype=float))
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != self.points.shape[1]:
            raise Exception('weights length must equal number of points')
        self.weights = weights
        self.V = _periodica.reduced_basis(self.U)    # reduced basis
        _, self.quotient_arcs, self.quotient_vertex_filtration, self.quotient_arc_filtration, self.quotient_arc_shift = \
            _periodica.periodic_voronoi(self.U, self.points, self.weights)
        self.n_quotient_vertices = _.shape[1]

    def quotient_complex(self, complex_type='delaunay'):
        if complex_type == 'delaunay':
            self.periodic_delaunay()
        elif complex_type == 'voronoi':
            self.periodic_voronoi()
        else:
            raise Exception(f'Does not support complex type {complex_type}')
        
        # print(f'vertex filtration:\n{self.quotient_vertex_filtration}')
        # print(f'arcs:\n')
        # for i in range(len(self.quotient_arcs)):
        #     print(f'{self.quotient_arcs[i]} {self.quotient_arc_filtration[i]:.6f} {self.quotient_arc_shift[:,i]}')
        # print(f'arcs:\n{self.quotient_arcs}')
        # print(f'arc filtration:\n{self.quotient_arc_filtration}')
        # print(f'arc shift:\n{self.quotient_arc_shift}')

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
            self.n_quotient_vertices = int(f.readline())
            self.quotient_vertex_filtration = []
            for i in range(self.n_quotient_vertices):
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
            self.tree = _periodica.merge_tree(self.n_quotient_vertices, self.d, self.V, self.quotient_arcs, self.quotient_arc_filtration, self.quotient_arc_shift, self.quotient_vertex_filtration)
        else:
            self.tree = _periodica.merge_tree(self.n_quotient_vertices, self.d, self.V, self.quotient_arcs, self.quotient_arc_filtration, self.quotient_arc_shift)
        return self.tree

    def print_merge_tree(self):
        if not hasattr(self, 'tree'):
            self.merge_tree()
        _periodica.print_merge_tree(self.tree)

    def barcodes(self):
        if not hasattr(self, 'tree'):
            self.merge_tree()
        self.bcodes = _periodica.barcode(self.d, self.tree)
        return self.bcodes
    
    def images(self, size=100):
        if not hasattr(self, 'bcodes'):
            self.barcodes()

        inf = np.inf
        xmin = min(map(lambda b: min(map(lambda x: x[0], b)), self.bcodes))
        xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), self.bcodes))
        xspan = xmax - xmin
        xmin, xmax = xmin - 0.12 * xspan, xmax + 0.12 * xspan
        
        self.persistence_images = []
        for i in range(self.d + 1):
            self.persistence_images.append(_periodica.image(self.bcodes[i], size, xmin, xmax))
        return self.persistence_images

    def plot_barcodes(self, show=True, ax=None):
        if not hasattr(self, 'bcodes'):
            self.barcodes()
        
        inf = np.inf
        sep = 1
        labels = [r'$\cdot R^0$', r'$\cdot 2 R^1$', r'$\cdot\pi R^2$', r'$\cdot \frac{4\pi}{3}R^3$']

        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(self.d + 1, 1)
            fig.set_size_inches(5, (self.d + 1) * 2)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)

        ax = np.atleast_1d(ax)
        if ax.shape[0] != self.d + 1:
            raise ValueError(f'ax must contain {self.d + 1} axes')

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

        if own_figure:
            plt.get_current_fig_manager().set_window_title('Barcode')
            if show:
                plt.show()
            plt.savefig('barcode.svg')

    def plot_diagram(self, show=True, ax=None):
        if not hasattr(self, 'bcodes'):
            self.barcodes()

        inf = np.inf
        labels = [r'$\cdot R^0$', r'$\cdot 2 R^1$', r'$\cdot\pi R^2$', r'$\cdot \frac{4\pi}{3}R^3$']

        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(self.d + 1, 1)
            fig.set_size_inches(5, (self.d + 1) * 2)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)

        ax = np.atleast_1d(ax)
        if ax.shape[0] != self.d + 1:
            raise ValueError(f'ax must contain {self.d + 1} axes')

        xmin = min(map(lambda b: min(map(lambda x: x[0], b)), self.bcodes))
        xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), self.bcodes))
        xspan = xmax - xmin
        xmin, xmax = xmin - 0.12 * xspan, xmax + 0.12 * xspan
        xticks = np.linspace(xmin, xmax, 5)[:-1]
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

        if own_figure:
            plt.get_current_fig_manager().set_window_title('Diagram')
            if show:
                plt.show()
            plt.savefig('diagram.svg')
            
    def plot_images(self, same_range=True, show=True, ax=None):
        if not hasattr(self, 'persistence_images'):
            self.images()

        inf = np.inf
        labels = [r'$\cdot R^0$', r'$\cdot 2 R^1$', r'$\cdot\pi R^2$', r'$\cdot \frac{4\pi}{3}R^3$']

        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(self.d + 1, 1)
            fig.set_size_inches(5, (self.d + 1) * 2)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)

        ax = np.atleast_1d(ax)
        if ax.shape[0] != self.d + 1:
            raise ValueError(f'ax must contain {self.d + 1} axes')

        xmin = min(map(lambda b: min(map(lambda x: x[0], b)), self.bcodes))
        xmax = max(map(lambda b: max(map(lambda x: x[1] if x[1] < inf else x[0], b)), self.bcodes))
        xspan = xmax - xmin
        xmin, xmax = xmin - 0.12 * xspan, xmax + 0.12 * xspan
        # print(f'xmin {xmin} xmax {xmax}')

        imgsize = self.persistence_images[0].shape[0]
        gap = (xmax - xmin) / imgsize

        xticks = np.linspace(xmin, xmax, 5)[:-1]
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

        if own_figure:
            plt.get_current_fig_manager().set_window_title('Image')
            if show:
                plt.show()
            plt.savefig('image.svg')

    def plot_all_descriptors(self, same_range=True, show=True):
        fig, ax = plt.subplots(self.d + 1, 3, squeeze=False, gridspec_kw={'width_ratios': [2, 1, 1]})
        fig.set_size_inches(15, (self.d + 1) * 2)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2, wspace=0.12)

        self.plot_barcodes(show=False, ax=ax[:, 0])
        self.plot_diagram(show=False, ax=ax[:, 1])
        self.plot_images(same_range=same_range, show=False, ax=ax[:, 2])

        ax[0, 0].set_title('Barcode')
        ax[0, 1].set_title('Diagram')
        ax[0, 2].set_title('Image')
        plt.get_current_fig_manager().set_window_title('All descriptors')
        if show:
            plt.show()
        plt.savefig('all_descriptors.svg')

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
            # Fill triangular facets without drawing edges, then draw only
            # non-coplanar facet intersections (true polytope edges).
            ax.add_collection3d(
                Poly3DCollection(
                    [domain_vertices[simplex] for simplex in hull.simplices],
                    linewidths=0,
                    facecolors=color,
                    edgecolors='none',
                    alpha=0.1,
                )
            )

            edge_to_facets = {}
            for facet_idx, simplex in enumerate(hull.simplices):
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    edge = tuple(sorted((simplex[i], simplex[j])))
                    edge_to_facets.setdefault(edge, []).append(facet_idx)

            for edge, facets in edge_to_facets.items():
                draw_edge = False
                if len(facets) <= 1:
                    draw_edge = True
                else:
                    ref_eq = hull.equations[facets[0]]
                    draw_edge = any(
                        not np.allclose(ref_eq, hull.equations[f], atol=1e-9, rtol=1e-6)
                        for f in facets[1:]
                    )

                if draw_edge:
                    edge_pts = domain_vertices[list(edge)]
                    ax.plot(
                        edge_pts[:, 0],
                        edge_pts[:, 1],
                        edge_pts[:, 2],
                        color=color,
                        lw=lw,
                        ls=ls,
                        alpha=alpha,
                    )

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
    
    def plot_delaunay(self, show=True, animation_gif=None, ax=None, slidebar=False):
        if not hasattr(self, 'points'):
            raise Exception('No input points')
        if not hasattr(self, 'V'):
            self.V = _periodica.reduced_basis(self.U)
        if not hasattr(self, 'weights'):
            self.weights = np.zeros(self.n_points)
        
        A, b =  _periodica.dirichlet_domain(self.V)
        P, I, __ = _periodica.points_in_3x_domain(self.V, A, b, self.points)
        P, delaunay_edges = _periodica.full_delaunay(self.U, self.points, self.weights)
        self.periodic_delaunay()

        show_slidebar = slidebar and not ax
        inf = np.inf
        # Slider range on the power-distance scale: from the earliest vertex
        # birth (-max weight) to the largest finite arc filtration value.
        max_filtration = max(map(lambda x: x if x < inf else -inf, self.quotient_arc_filtration))
        min_filtration = min(0.0, float(np.min(self.quotient_vertex_filtration)))

        if not ax:
            fig = plt.figure()

        if self.d == 2:
            if not ax:
                ax = fig.add_subplot()

            # self.draw_unit_cell(self.V[:,:-1], ax, green)
            ax.arrow(0, 0, self.V[0,0], self.V[1,0], color=green, width=0.01, head_width=0.04)
            ax.arrow(0, 0, self.V[0,1], self.V[1,1], color=green, width=0.01, head_width=0.04)

            self.draw_polytope(A, b, ax, lw=1, alpha=1, ls='-', fill_color='b')
            self.draw_polytope(A, b * 3, ax, lw=0.75, ls='-', alpha=1)

            # ax.scatter(*P, color='k', s=5, zorder=1)
            # ax.scatter(*P[:,self.n:], color='k', s=5, zorder=1)
            # ax.scatter(*canonical_points, color='k', s=5)
            
            # color = '#0000FE' if arc else 'k'
            # alpha = 0.8 if arc else 0.2
            # lw = 1.5 if arc else 1
            
            # Plot full Delaunay skeleton
            for s, t in delaunay_edges:
                ax.plot(*P[:,(s,t)], '--', lw=1, color='k', alpha=0.8, zorder=0)

            # Highlight the periodic Delaunay edges
            # (arc endpoints index the kept points; P uses original indexing)
            for i in range(len(self.quotient_arcs)):
                s, t = self.quotient_arcs[i]
                sP = P[:,self.kept_points[s]]
                tP = P[:,self.kept_points[t]] + self.V[:,:-1] @ self.quotient_arc_shift[:,i]
                ax.plot([sP[0], tP[0]], [sP[1], tP[1]], lw=1.5, color='#0000FE', zorder=1)

            ax.set_aspect(1)

            if show_slidebar:
                host_fig = ax.figure
                host_fig.subplots_adjust(bottom=0.2)
                slider_ax = host_fig.add_axes([0.2, 0.08, 0.6, 0.04])
                filtration_slider = Slider(slider_ax, 'f', min_filtration, max_filtration, valinit=0.0)
                # The sublevel set of the power distance at f is the union of
                # balls of radius sqrt(f + w_i); the ball for point i only
                # exists once f >= -w_i.
                circles = []
                for i in range(P.shape[1]):
                    circle = Circle((P[0, i], P[1, i]), radius=np.sqrt(self.weights[I[i]]), fill=True, color='#aaaaaa', alpha=1, zorder=0.5)
                    ax.add_patch(circle)
                    circles.append(circle)

                def update_filtration(f):
                    for i in range(len(circles)):
                        circles[i].set_radius(np.sqrt(max(f + self.weights[I[i]], 0.0)))
                    ax.figure.canvas.draw_idle()

                filtration_slider.on_changed(update_filtration)
                self._delaunay_filtration_slider = filtration_slider
        
        else:
            if not ax:
                ax = fig.add_subplot(projection='3d')

            self.draw_polytope(A, b * 3, ax, lw=1, ls='-', alpha=0.5)

            ax.scatter(*P, color='b', s=5)
            # ax.scatter(*canonical_points, color=red, s=5)

            # Highlight the periodic Delaunay edges
            # (arc endpoints index the kept points; P uses original indexing)
            for i in range(len(self.quotient_arcs)):
                s, t = self.quotient_arcs[i]
                sP = P[:,self.kept_points[s]]
                tP = P[:,self.kept_points[t]] + self.V[:,:-1] @ self.quotient_arc_shift[:,i]
                ax.plot([sP[0], tP[0]], [sP[1], tP[1]], [sP[2], tP[2]], lw=1.5, color='#0000FE', zorder=1)
            
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
        # plt.savefig('delaunay.svg')

    def plot_voronoi(self, show=True, animation_gif=None, use_circumcenter=False, ax=None, slidebar=False):
        if not hasattr(self, 'points'):
            raise Exception('No input points')
        if not hasattr(self, 'V'):
            self.V = _periodica.reduced_basis(self.U)
        if not hasattr(self, 'weights'):
            self.weights = np.zeros(self.n_points)
        
        if not ax:
            fig = plt.figure()
        
        A, b =  _periodica.dirichlet_domain(self.V)
        P, I, __ = _periodica.points_in_3x_domain(self.V, A, b, self.points)

        voronoi_points, voronoi_edges = _periodica.full_voronoi(
            self.U, self.points, self.weights, use_circumcenter
        )
        canonical_voronoi_points, periodic_voronoi_edges, point_filtrations, edge_filtrations, shift_vectors = _periodica.periodic_voronoi(
            self.U, self.points, self.weights, use_circumcenter
        )

        # Voronoi point filtrations are negated power values; the largest
        # power value over all Voronoi vertices bounds the Delaunay slider.
        max_filtration = -np.min(point_filtrations)
        min_filtration = min(0.0, -float(np.max(self.weights)))

        if self.d == 2:
            if not ax:
                ax = fig.add_subplot()

            # self.draw_unit_cell(self.V[:,:-1], ax, green)
            ax.arrow(0, 0, self.V[0,0], self.V[1,0], color=green, width=0.01, head_width=0.04, zorder=0)
            ax.arrow(0, 0, self.V[0,1], self.V[1,1], color=green, width=0.01, head_width=0.04, zorder=0)

            self.draw_polytope(A, b, ax, lw=1, alpha=1, ls='-', fill_color='b')
            self.draw_polytope(A, b * 3, ax, lw=0.75, ls='-', alpha=1)
            
            self.plot_delaunay(ax=ax, show=False)

            limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xy'])

            ax.scatter(*canonical_voronoi_points, color='r')
            for s, t in voronoi_edges:
                ax.plot(*voronoi_points[:,(s,t)], '-.', lw=1, color='r', zorder=1)
            
            # for s, t in periodic_voronoi_edges:
            #     ax.plot(*voronoi_points[:,(s,t)], lw=1.5, color='r', zorder=2)

            for i in range(periodic_voronoi_edges.shape[0]):
                s, t = periodic_voronoi_edges[i]
                sP = canonical_voronoi_points[:,s]
                tP = canonical_voronoi_points[:,t] + self.V[:,:-1] @ shift_vectors[:,i]
                ax.plot([sP[0], tP[0]], [sP[1], tP[1]], lw=1.5, color='r', zorder=2)

            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_aspect(1)

            if slidebar:
                host_fig = ax.figure
                host_fig.subplots_adjust(bottom=0.2)
                slider_ax = host_fig.add_axes([0.2, 0.08, 0.6, 0.04])
                filtration_slider = Slider(slider_ax, 'f', min_filtration, max_filtration, valinit=0.0)
                # Same power-scale balls as plot_delaunay: radius sqrt(f + w_i).
                circles = []
                for i in range(P.shape[1]):
                    circle = Circle((P[0, i], P[1, i]), radius=np.sqrt(self.weights[I[i]]), fill=True, color='#aaaaaa', alpha=1, zorder=0.5)
                    ax.add_patch(circle)
                    circles.append(circle)

                def update_filtration(f):
                    for i in range(len(circles)):
                        circles[i].set_radius(np.sqrt(max(f + self.weights[I[i]], 0.0)))
                    ax.figure.canvas.draw_idle()

                filtration_slider.on_changed(update_filtration)
                self._voronoi_filtration_slider = filtration_slider
        
        else:
            if not ax:
                ax = fig.add_subplot(projection='3d')

            self.draw_polytope(A, b * 3, ax, lw=1, ls='-', alpha=0.5)

            self.plot_delaunay(ax=ax, show=False)

            limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])

            # ax.scatter(*voronoi_points, color='r')
            # for s, t in periodic_voronoi_edges:
            #     ax.plot(*voronoi_points[:,(s,t)], color='r', zorder=1)
            
            for s,t in voronoi_edges:
                ax.plot(*voronoi_points[:,(s,t)], '-.', lw=1, color='k', zorder=1)

            # print(f'Number of canonical simplices: {canonical_voronoi_points.shape[1]}')
            ax.scatter(*canonical_voronoi_points, color='g')
            for i in range(periodic_voronoi_edges.shape[0]):
                s, t = periodic_voronoi_edges[i]
                sP = canonical_voronoi_points[:,s]
                tP = canonical_voronoi_points[:,t] + self.V[:,:-1] @ shift_vectors[:,i]

                # print(f'{s}->{t} shift {shift_vectors[:,i]} sP {canonical_voronoi_points[:,s]} tP {canonical_voronoi_points[:,t]}+{self.V[:,:-1] @ shift_vectors[:,i]}')
                ax.plot([sP[0], tP[0]], [sP[1], tP[1]], [sP[2], tP[2]], lw=1.5, color='r', zorder=2)

            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_box_aspect(np.ptp(limits, axis = 1))
            
            if animation_gif:
                gif = animation.FuncAnimation(fig, lambda x: ax.view_init(azim=x), frames=np.arange(0, 362, 2), interval=100)
                gif.save(animation_gif, dpi=80, writer='imagemagick')
        
        ax.set_xticks([])
        ax.set_yticks([])

        plt.get_current_fig_manager().set_window_title('')
        if show:
            plt.show()
        # plt.savefig('delaunay.svg')

    def plot_geometry(self, TYPE, show=False, slidebar=False, use_circumcenter=True):
        if TYPE == 'delaunay':
            self.plot_delaunay(show=show, slidebar=slidebar)
        else:
            self.plot_voronoi(show=show, slidebar=slidebar, use_circumcenter=use_circumcenter)
