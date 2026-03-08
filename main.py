from periodica.core import Periodica

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

def run_2d_delaunay_example():
    periodica = Periodica()
    periodica.generate_random_points(n=4, d=2, seed=3)
    periodica.plot_delaunay(show=True)
    periodica.print_merge_tree()
    # periodica.plot_barcodes(show=False)
    # periodica.plot_diagram(show=False)
    # periodica.plot_images(same_range=True, show=True)
    
def run_2d_voronoi_example():
    periodica = Periodica()
    periodica.generate_random_points(n=1, d=2, seed=0)
    periodica.plot_voronoi(show=True, use_circumcenter=True)
    # periodica.print_merge_tree()
    
def run_3d_voronoi_example():
    periodica = Periodica()
    periodica.generate_random_points(n=1, d=3, seed=1)
    periodica.plot_voronoi(show=True, use_circumcenter=True)
    # periodica.print_merge_tree()

def run_2d_grid_example():
    periodica = Periodica()
    periodica.generate_grid_points(d=2, k=3)
    periodica.plot_delaunay(show=True)

def run_2d_weighted_grid_example():
    periodica = Periodica()
    periodica.generate_grid_points(d=2, k=3)
    weights = [0] * periodica.n
    weights[0] = 0.1
    periodica.set_weights(weights)
    periodica.plot_delaunay(show=True)

def run_2d_weighted_delaunay_example():
    periodica = Periodica()
    periodica.generate_random_points(n=10, d=2)
    periodica.plot_delaunay(show=True)
    periodica.print_merge_tree()
    # periodica.plot_barcodes(show=False)
    # periodica.plot_diagram(show=False)
    # periodica.plot_images(same_range=True, show=True)

def run_3d_delaunay_example():
    periodica = Periodica()
    periodica.generate_random_points(n=4, d=3)
    periodica.plot_delaunay(show=False)
    periodica.print_merge_tree()
    periodica.plot_barcodes(show=False)
    periodica.plot_diagram(show=False)
    periodica.plot_images(same_range=True, show=True)

def run_3d_weighted_delaunay_example():
    periodica = Periodica()
    periodica.generate_random_points(n=4, d=3)
    periodica.plot_delaunay(show=False)
    periodica.print_merge_tree()
    periodica.plot_barcodes(show=False)
    periodica.plot_diagram(show=False)
    periodica.plot_images(same_range=True, show=True)

if __name__ == "__main__":
    # run_2d_quotient_example_1()
    # run_2d_quotient_example_2()
    # run_3d_quotient_example()
    
    # run_2d_delaunay_example()
    # run_3d_delaunay_example()

    # run_2d_weighted_delaunay_example()
    # run_3d_weighted_delaunay_example()

    # run_2d_grid_example()
    # run_2d_weighted_grid_example()

    run_2d_voronoi_example()
    # run_3d_voronoi_example()
