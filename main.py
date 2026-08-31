from periodica.core import Periodica
import numpy as np

def run_2d_quotient_example_1():
    periodica = Periodica()
    periodica.load_quotient_complex('examples/example_2d_1.txt')
    periodica.print_merge_tree()
    periodica.plot_barcodes(show=False)
    periodica.plot_diagram(show=False)
    periodica.plot_images(same_range=True, show=True)

def run_2d_quotient_example_2():
    periodica = Periodica()
    periodica.load_quotient_complex('examples/example_2d_2.txt')
    periodica.print_merge_tree()
    periodica.plot_barcodes(show=False)
    periodica.plot_diagram(show=False)
    periodica.plot_images(same_range=True, show=True)

def run_3d_quotient_example():
    periodica = Periodica()
    periodica.load_quotient_complex('examples/example_3d_1.txt')
    periodica.print_merge_tree()
    periodica.plot_barcodes(show=False)
    periodica.plot_diagram(show=False)
    periodica.plot_images(same_range=True, show=True)

def run_example(INPUT, TYPE='delaunay', show_geometry=True):
    print(f'Quotient complex type: {TYPE}')
    periodica = Periodica()
    periodica.set_geometry(INPUT)
    periodica.quotient_complex(TYPE)
    periodica.merge_tree()
    periodica.print_merge_tree()
    periodica.plot_all_descriptors(show=(not show_geometry), same_range=False)
    if show_geometry:
        if TYPE != 'delaunay':
            periodica.plot_geometry(TYPE, show=False, slidebar=True, use_circumcenter=False)
        periodica.plot_geometry(TYPE, show=True, slidebar=True, )


EXAMPLES = {
"2d" : {
    "square" : {
        "d" : 2,
        "U" : np.eye(2),
        "n_points" : 1,
        "points" : np.array([
            # [0, 0],
            [0.5, 0.5],
            # [0.5, 0],
            # [0, 0.5]
        ]).T
    },
    "weighted" : {
        "d" : 2,
        "U" : np.eye(2),
        "n_points" : 2,
        "points" : np.array([
            # [0, 0],
            [0.5, 0.5],
            # [0.5, 0],
            [0.3, 0.5]
        ]).T,
        "weights" : np.array([
            0.1, 0 #, 0, 0
        ])
    },
    "hexagon" : {
        "d" : 2,
        "U" : np.array([
            [1, np.cos(np.pi/3)],
            [0, np.sin(np.pi/3)]
        ]),
        "n_points" : 1,
        "points" : np.array([
            [0, 0],
        ]).T
    },
    "tunnel" : {
        "d" : 2,
        "U" : np.eye(2),
        "n_points" : 4,
        "points" : np.array([
            [0, 0],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.75, 0.75],
        ]).T
    },
    "cycle" : {
        "d" : 2,
        "U" : np.eye(2),
        "n_points" : 7,
        "points" : np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.15, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.85, 0.85],
            [0.0, 0.5],
        ]).T
    }
},
"3d" : {
    "cube" : {
        "d" : 3,
        "U" : np.eye(3),
        "n_points" : 1,
        "points" : np.array([
            [0.5, 0.5, 0.5],
        ]).T
    },
    "diamond" : {
        "d" : 3,
        "U" : np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]),
        "n_points" : 2,
        "points" : np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ]).T
    },
    "line" : {
        "d" : 3,
        "U" : np.eye(3),
        "n_points" : 4,
        "points" : np.array([
            [0,0,z] for z in [0, 0.25, 0.5, 0.75]]).T
    },
    "tunnel" : {
        "d" : 3,
        "U" : np.eye(3),
        "n_points" : 70,
        "points" : np.array([
            [x,y,z] for x,y in [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.15, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.85, 0.85],
            [0.0, 0.5],
        ] for z in np.linspace(0,1,10,endpoint=False)]).T
    }
}
}

if __name__ == "__main__":
    # run_2d_quotient_example_1()
    # run_2d_quotient_example_2()
    # run_3d_quotient_example()
    
    # run_example(EXAMPLES['2d']['square'])
    # run_example(EXAMPLES['2d']['square'], 'voronoi')
    # run_example(EXAMPLES['2d']['hexagon'])
    # run_example(EXAMPLES['2d']['hexagon'], 'voronoi')
    # run_example(EXAMPLES['2d']['tunnel'])
    # run_example(EXAMPLES['2d']['tunnel'], 'voronoi')
    # run_example(EXAMPLES['2d']['cycle'])
    # run_example(EXAMPLES['2d']['cycle'], 'voronoi')
    run_example(EXAMPLES['2d']['weighted'])
    # run_example(EXAMPLES['2d']['weighted'], 'voronoi')

    # run_example(EXAMPLES['3d']['cube'])
    # run_example(EXAMPLES['3d']['cube'], 'voronoi')
    # run_example(EXAMPLES['3d']['diamond'])
    # run_example(EXAMPLES['3d']['diamond'], 'voronoi')
    # run_example(EXAMPLES['3d']['line'])
    # run_example(EXAMPLES['3d']['line'], 'voronoi')
    # run_example(EXAMPLES['3d']['tunnel'], show_geometry=True)
    # run_example(EXAMPLES['3d']['tunnel'], 'voronoi', show_geometry=False)

    # N = 50
    # input = {"d" : 3, "U" : np.eye(3)}
    # input['n_points'] = N
    # input['points'] = np.random.random((3, N))
    # # print(input['points'])
    # # run_example(input, 'delaunay', show_geometry=False)
    # print(f'{N} points in unit cell')
    # run_example(input, 'voronoi', show_geometry=False)
