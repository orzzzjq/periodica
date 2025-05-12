#include "debug.h"
//#include "delaunay.h"
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>

#include <CGAL/squared_distance_2.h> //for 2D functions
#include <CGAL/squared_distance_3.h> //for 3D functions

#include <CGAL/Epeck_d.h>
#include <CGAL/Random.h>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include <Eigen/Dense>

#include <vector>

using K2 = CGAL::Epeck_d< CGAL::Dimension_tag<2> >;
using Point2 = K2::Point_d;

using K3 = CGAL::Epeck_d< CGAL::Dimension_tag<3> >;
using Point3 = K3::Point_d;

using VertexId = boost::property<boost::vertex_index_t, int>;
using EdgeWeight = boost::property<boost::edge_weight_t, double>;

using Graph = boost::adjacency_list<boost::vecS, boost::vecS,
                                    boost::undirectedS, VertexId, EdgeWeight>;

using VertexDescriptor = boost::graph_traits<Graph>::vertex_descriptor;
using EdgeDescriptor = boost::graph_traits<Graph>::edge_descriptor;

using VertexIterator = boost::graph_traits<Graph>::vertex_iterator;
using EdgeIterator = boost::graph_traits<Graph>::edge_iterator;

using namespace std;

Eigen::MatrixXi DelaunaySkeleton(const Eigen::MatrixXd& points) {
    if (points.rows() != 2 && points.rows() != 3) {
        throw std::invalid_argument("Input must be a 2D NumPy array of shape (2, n) or (3, n)");
    }

    int n = points.cols(), d = points.rows();

    Gudhi::Simplex_tree complex;

    if (d == 2) {
        vector<double> coord(2);
        vector<Point2> p;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                coord[j] = points(j, i);
            }
            p.push_back(Point2(coord[0], coord[1]));
        }
        Gudhi::alpha_complex::Alpha_complex<K2> alphaComplex(p);
        alphaComplex.create_complex(complex, INFINITY, false, true);
    }
    else {
        vector<double> coord(3);
        vector<Point3> p;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                coord[j] = points(j, i);
            }
            p.push_back(Point3(coord[0], coord[1], coord[2]));
        }
        Gudhi::alpha_complex::Alpha_complex<K3> alphaComplex(p);
        alphaComplex.create_complex(complex, INFINITY, false, true);
    }

    vector<vector<int>> edges;

    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            edges.push_back(id);
        }
    }

    Eigen::MatrixXi result(size(edges), 2);

    for (size_t i = 0; i < size(edges); ++i) {
        for (size_t j = 0; j < 2; ++j) {
            result(i, j) = edges[i][j];
        }
    }

    return result;
}

Eigen::MatrixXi EuclideanMST(const Eigen::MatrixXd& points) {
    if (points.rows() != 2 && points.rows() != 3) {
        throw std::invalid_argument("Input must be a 2D NumPy array of shape (2, n) or (3, n)");
    }

    int n = points.cols(), d = points.rows();

    Gudhi::Simplex_tree complex;

    if (d == 2) {
        vector<double> coord(2);
        vector<Point2> p;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                coord[j] = points(j, i);
            }
            p.push_back(Point2(coord[0], coord[1]));
        }
        Gudhi::alpha_complex::Alpha_complex<K2> alphaComplex(p);
        alphaComplex.create_complex(complex, INFINITY, false, true);
    }
    else {
        vector<double> coord(3);
        vector<Point3> p;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                coord[j] = points(j, i);
            }
            p.push_back(Point3(coord[0], coord[1], coord[2]));
        }
        Gudhi::alpha_complex::Alpha_complex<K3> alphaComplex(p);
        alphaComplex.create_complex(complex, INFINITY, false, true);
    }

    Graph graph;

    for (int i = 0; i < n; ++i) {
        add_vertex(i, graph);
    }

    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            int s = id[0], t = id[1];
            double sq_dist = 0;
            if (d == 2) {
                double dx = points(0, s) - points(0, t), dy = points(1, s) - points(1, t);
                sq_dist = dx * dx + dy * dy;
            }
            else {
                double dx = points(0, s) - points(0, t), dy = points(1, s) - points(1, t), dz = points(2, s) - points(2, t);
                sq_dist = dx * dx + dy * dy + dz * dz;
            }
            add_edge(s, t, sq_dist, graph);
        }
    }
    
    std::list<EdgeDescriptor> mst;
    boost::kruskal_minimum_spanning_tree(graph, std::back_inserter(mst));

    vector<vector<int>> edges;

    for (EdgeDescriptor e : mst)
    {
       VertexDescriptor s = source(e, graph);
       VertexDescriptor t = target(e, graph);
       edges.push_back(vector<int>{static_cast<int>(s), static_cast<int>(t)});
    }

    Eigen::MatrixXi result(size(edges), 2);

    for (size_t i = 0; i < size(edges); ++i) {
        for (size_t j = 0; j < 2; ++j) {
            result(i, j) = edges[i][j];
        }
    }

    return result;
}

double LatticeDiameter(const Eigen::MatrixXd& U) {
    if (U.rows() != U.cols() || (U.cols() != 2 && U.cols() != 3)) {
        throw std::invalid_argument("Input must be a 2x2 or 3x3 matrix");
    }

    int d = U.rows();
    int mask = 0;
    double max_dist = 0, dist;
    for (; mask < (1 << d); ++mask) {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(d);
        for (int i = 0; i < d; ++i) {
            if (mask & (1 << i)) { // add
                v += U.col(i);
            }
            else { // minus
                v -= U.col(i);
            }
        }
        dist = v.norm();
        if (dist > max_dist) max_dist = dist;
    }
    
    return max_dist;
}

void testLatticeDiameter() {
    Eigen::MatrixXd U(2, 2);

    U(0, 0) = 1, U(0, 1) = 0;
    U(1, 0) = 0, U(1, 1) = 1;

    cout << LatticeDiameter(U) << "\n";
}

int main() 
{
    testLatticeDiameter();
}

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(periodica, m) {
   m.def("delaunay_skeleton", &DelaunaySkeleton, "Compute 1-skeleton of 2D & 3D Delaunay triangulations",
         py::arg("points"));
   m.def("euclidean_mst", &EuclideanMST, "Compute 2D & 3D Euclidean minimum spanning trees",
         py::arg("points"));
   m.def("lattice_diameter", &LatticeDiameter, "Compute diameter of a unit cell in the lattice, input should be a 2x2 or 3x3 matrix representing the lattice basis",
         py::arg("U"));
}
