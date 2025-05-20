#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "delaunay.h"
#include "merge_tree.h"

namespace py = pybind11;

PYBIND11_MODULE(periodica, m) {
   m.def("delaunay_skeleton", &DELAUNAY::DelaunaySkeleton, 
        "Compute 1-skeleton of 2D & 3D Delaunay triangulations",
        py::arg("points"));
   m.def("euclidean_mst", &DELAUNAY::EuclideanMST, 
        "Compute 2D & 3D Euclidean minimum spanning trees",
        py::arg("points"));
   m.def("reduced_basis", &DELAUNAY::reducedBasis, 
        "Reduce Lattice basis into reduced basis",
        py::arg("U"));
   m.def("dirichlet_domain", &DELAUNAY::DirichletDomain, 
        "Compute Dirichlet domain of a lattice, input should be a 2x2 or 3x3 matrix representing the lattice basis",
        py::arg("U"));
   m.def("canonical_points", &DELAUNAY::canonicalPoints, 
        "Compute the canonical copy of points in the Dirichlet domain",
        py::arg("A"), py::arg("b"), py::arg("points"));
   m.def("points_in_3x_domain", &DELAUNAY::pointsIn3xDomain, 
        "Compute the periodic points in the 3X Dirichlet domain",
        py::arg("V"), py::arg("A"), py::arg("b"), py::arg("canonical_points"));
   m.def("periodic_delaunay", &DELAUNAY::periodicDelaunay, 
        "Compute quotient complex from periodic Delaunay complex",
        py::arg("U"), py::arg("points"));
   m.def("merge_tree", &PMT::mergeTree, "Compute periodic merge tree from a quotient complex",
        py::arg("n"), py::arg("d"), py::arg("V"),
        py::arg("arcs"), py::arg("arc_filtration"), py::arg("arc_shift"),
        py::arg("vertex_filtration") = Eigen::VectorXd(0));
   m.def("print_merge_tree", &PMT::printMergeTree, "Print periodic merge tree in a nice format",
        py::arg("tree"));
   m.def("barcode", &PMT::barcode, "Compute periodic barcode from merge tree",
        py::arg("d"), py::arg("tree"));
}
