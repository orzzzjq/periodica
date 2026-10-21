#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "delaunay.h"
#include "merge_tree.h"
#include "persistence_image.h"

namespace py = pybind11;

PYBIND11_MODULE(_periodica, m) {
   m.def("delaunay_skeleton", py::overload_cast<const Eigen::MatrixXd&>(&DELAUNAY::DelaunaySkeleton), 
        "Compute 1-skeleton of 2D & 3D Delaunay triangulations",
        py::arg("points"));
   m.def("weighted_delaunay_skeleton", py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&>(&DELAUNAY::DelaunaySkeleton),
        "Compute 1-skeleton of 2D & 3D weighted Delaunay triangulations",
        py::arg("points"), py::arg("weights"));
   m.def("euclidean_mst", &DELAUNAY::EuclideanMST, 
        "Compute 2D & 3D Euclidean minimum spanning trees",
        py::arg("points"));
   m.def("reduced_basis", &DELAUNAY::reducedBasis,
        "Reduce Lattice basis into reduced basis",
        py::arg("U"));
   m.def("reduced_basis_coeffs", &DELAUNAY::reducedBasisCoeffs,
        "Reduce lattice basis into an obtuse superbase; returns (V, T) with "
        "V = U * T, T integer, and the columns of both summing to zero",
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
        "Compute quotient complex from periodic Delaunay complex. Returns (edges, filtration, shift, kept), "
        "where kept holds the original indices of the points that appear in the weighted triangulation; "
        "points hidden by larger weights are dropped and edge endpoints are remapped accordingly",
        py::arg("U"), py::arg("points"), py::arg("weights"));
   m.def("periodic_voronoi", &DELAUNAY::periodicVoronoi, 
        "Compute quotient complex from periodic Voronoi complex",
        py::arg("U"), py::arg("points"), py::arg("weights"), py::arg("useCircumCenter") = false);
   m.def("full_voronoi", &DELAUNAY::fullVoronoiSkeleton, 
        "Compute the dual complex of the Delaunay of the points in the 3X Dirichlet domain",
        py::arg("U"), py::arg("points"), py::arg("weights"), py::arg("useCircumCenter") = false);
   m.def("full_delaunay", &DELAUNAY::fullDelaunaySkeleton, 
        "Compute the Delaunay complex of the points in the 3X Dirichlet domain",
        py::arg("U"), py::arg("points"), py::arg("weights"));
   py::class_<PMT::MergeTree>(m, "MergeTree",
        "Periodic merge tree: per quotient vertex a beam of time-sorted "
        "(time, coeff, exponent, child) events; indexable and iterable like "
        "a list of beams, to_list() gives that list")
        .def_readonly("d", &PMT::MergeTree::d)
        .def_readonly("n", &PMT::MergeTree::n)
        .def("__len__", &PMT::MergeTree::size)
        .def("__getitem__", [](const PMT::MergeTree& t, int i) {
            if (i < 0) i += t.n;
            if (i < 0 || i >= t.n) throw py::index_error();
            std::vector<PMT::Event> epoch;
            t.beam(i, epoch);
            py::list out;
            for (const auto& e : epoch) out.append(py::make_tuple(e.time, e.ratio, e.exponent, e.child));
            return out;
        })
        .def("__iter__", [](const PMT::MergeTree& t) { return py::iter(py::cast(t.toList())); })
        .def("to_list", &PMT::MergeTree::toList, "The tree as a list of beams, each a list of (time, coeff, exponent, child)")
        .def_static("from_list", &PMT::MergeTree::fromList, py::arg("tree"))
        .def("__repr__", [](const PMT::MergeTree& t) {
            return "<periodica.MergeTree d=" + std::to_string(t.d) + " n=" + std::to_string(t.n) +
                   " events=" + std::to_string(t.events.size()) + ">";
        });
   m.def("merge_tree", &PMT::mergeTree, "Compute periodic merge tree from a quotient complex",
        py::arg("n"), py::arg("d"), py::arg("V"),
        py::arg("arcs"), py::arg("arc_filtration"), py::arg("arc_shift"),
        py::arg("vertex_filtration") = Eigen::VectorXd(0));
   m.def("grid_merge_tree", &PMT::gridMergeTree,
        "Periodic merge tree of the lower-star filtration of a periodic grid: f is the "
        "field flattened in C order over shape, directions (columns) the grid steps "
        "joining neighbors; no arc list is materialized",
        py::arg("d"), py::arg("U"), py::arg("shape"), py::arg("f"), py::arg("directions"));
   m.def("print_merge_tree", &PMT::printMergeTree, "Print periodic merge tree in a nice format",
        py::arg("tree"));
   m.def("print_merge_tree", [](const PMT::MergeTree::List& tree) { PMT::printMergeTree(PMT::MergeTree::fromList(tree)); },
        py::arg("tree"));
   m.def("barcode", &PMT::barcode,
        "Compute periodic barcode from merge tree; drop_zero_length leaves out bars with death == birth",
        py::arg("d"), py::arg("tree"), py::arg("drop_zero_length") = false);
   m.def("barcode", [](int d, const PMT::MergeTree::List& tree, bool drop) { return PMT::barcode(d, PMT::MergeTree::fromList(tree), drop); },
        py::arg("d"), py::arg("tree"), py::arg("drop_zero_length") = false);
   m.def("image", &persistenceImage, "Compute persistence image from barcode",
        py::arg("barcode"), py::arg("size"), py::arg("min"), py::arg("max"));
}
