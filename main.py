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
    periodica.generate_random_input(n=3, d=2)
    periodica.plot_delaunay(show=False)
    periodica.print_merge_tree()
    periodica.plot_barcodes(show=False)
    periodica.plot_diagram(show=False)
    periodica.plot_images(same_range=True, show=True)

def run_3d_delaunay_example():
    periodica = Periodica()
    periodica.generate_random_input(n=4, d=3)
    periodica.plot_delaunay(show=False)
    periodica.print_merge_tree()
    periodica.plot_barcodes(show=False)
    periodica.plot_diagram(show=False)
    periodica.plot_images(same_range=True, show=True)

if __name__ == "__main__":
    # run_2d_quotient_example_1()
    # run_2d_quotient_example_2()
    # run_3d_quotient_example()
    
    run_2d_delaunay_example()
    # run_3d_delaunay_example()
