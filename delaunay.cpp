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

Eigen::MatrixXd reducedBasis(const Eigen::MatrixXd& U) {
    if (U.rows() != U.cols() || (U.cols() != 2 && U.cols() != 3)) {
        throw std::invalid_argument("Input must be a 2x2 or 3x3 matrix");
    }

    int d = U.rows();
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(d, d + 1);
    for (int i = 0; i < d; ++i) {
        V.col(i) = U.col(i);
        V.col(d) -= U.col(i);
    }

    bool reduced = 1;
    int i, j, h, k;
    vector<vector<int>> id;

    if (d == 2) {
        id.push_back({0, 1, 2});
        id.push_back({0, 2, 1});
        id.push_back({1, 2, 0});
        for (auto c : id) {
            i = c[0], j = c[1], h = c[2];
            if (V.col(i).dot(V.col(j)) > 0) {
                reduced = 0;
                break;
            }
        }
        while (!reduced) {
            V.col(h) += V.col(i);
            V.col(i) *= -1;
            reduced = 1;
            for (auto c : id) {
                i = c[0], j = c[1], h = c[2];
                if (V.col(i).dot(V.col(j)) > 0) {
                    reduced = 0;
                    break;
                }
            }
        }
    }
    else {
        id.push_back({0, 1, 2, 3});
        id.push_back({0, 2, 1, 3});
        id.push_back({0, 3, 1, 2});
        id.push_back({1, 2, 0, 3});
        id.push_back({1, 3, 0, 2});
        id.push_back({2, 3, 0, 1});
        for (auto c : id) {
            i = c[0], j = c[1], h = c[2], k = c[3];
            if (V.col(i).dot(V.col(j)) > 0) {
                reduced = 0;
                break;
            }
        }
        while (!reduced) {
            V.col(h) += V.col(i);
            V.col(k) += V.col(i);
            V.col(i) *= -1;
            reduced = 1;
            for (auto c : id) {
                i = c[0], j = c[1], h = c[2], k = c[3];
                if (V.col(i).dot(V.col(j)) > 0) {
                    reduced = 0;
                    break;
                }
            }
        }
    }

    return V;
}

// void generateBasis(const Eigen::MatrixXd& U, vector<Eigen::VectorXd>& V, int d, vector<int>& coeff) {
//     if (size(coeff) == d) {
//         Eigen::VectorXd v = Eigen::VectorXd::Zero(d);
//         bool allZero = 1;
//         for (int i = 0; i < d; ++i) {
//             if (coeff[i] != 0) {
//                 allZero = 0;
//                 v += coeff[i] * U.col(i);
//             }
//         }
//         if (!allZero) V.push_back(v);
//         return;
//     }
//     for (int c = -1; c <= 1; ++c) {
//         coeff.push_back(c);
//         generateBasis(U, V, d, coeff);
//         coeff.pop_back();
//     }
// }

pair<Eigen::MatrixXd, Eigen::VectorXd> DirichletDomain(const Eigen::MatrixXd& U) {
    if (U.rows() != U.cols() || (U.cols() != 2 && U.cols() != 3)) {
        throw std::invalid_argument("Input must be a 2x2 or 3x3 matrix");
    }

    int d = U.rows();

    Eigen::MatrixXd V = reducedBasis(U);

    vector<Eigen::VectorXd> F; // face normals
    
    for (int mask = 1; mask < (1 << d); ++mask) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(d);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(d);
        for (int i = 0; i < d; ++i) {
            if (mask & (1 << i)) { // plus minus
                u += V.col(i);
                v -= V.col(i);
            }
        }
        F.push_back(u);
        F.push_back(v);
    }

    Eigen::MatrixXd A(size(F), d);
    Eigen::VectorXd b(size(F));
    for (int i = 0; i < size(F); ++i) {
        A.row(i) = F[i];
        b(i) = F[i].norm() * F[i].norm() / 2;
    }

    return {A, b};
}

Eigen::MatrixXd canonicalPoints(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& points) {
    if (A.cols() != points.rows() || (A.cols() != 2 && A.cols() != 3)) {
        throw std::invalid_argument("Invalid input dimension");
    }

    vector<Eigen::VectorXd> cpoints;
    Eigen::VectorXd p, v;
    for (int i = 0; i < points.cols(); ++i) {
        p = points.col(i);
        int j = 0;
        while (j < A.rows()) {
            v = A.row(j);
            double scal = p.dot(v) / (2 * b(j));
            if (scal > (0.5 + 1e-9)) { // shift
                p -= v * floor(scal + 0.5);
                j = 0;
            }
            else {
                ++j;
            }
        }
        cpoints.push_back(p);
    }

    Eigen::MatrixXd result(points.rows(), size(cpoints));
    for (int i = 0; i < size(cpoints); ++i) {
        result.col(i) = cpoints[i];
    }

    return result;
}

pair<Eigen::MatrixXd, Eigen::MatrixXi> domainX3Points(const Eigen::MatrixXd V, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::MatrixXd& canonical_points) {
    if (A.cols() != canonical_points.rows() || (A.cols() != 2 && A.cols() != 3)) {
        throw std::invalid_argument("Invalid input dimension");
    }

    vector<Eigen::VectorXd> points;
    vector<Eigen::VectorXi> shifts;

    auto dfs = [&points, &shifts](const Eigen::MatrixXd V, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, int d,
        const Eigen::MatrixXd& canonical_points, vector<int>& coeff, auto&& dfs) -> void {
        if (size(coeff) == d) {
            bool allZero = 1;
            Eigen::VectorXi s = Eigen::VectorXi::Zero(d);
            Eigen::VectorXd v = Eigen::VectorXd::Zero(d);
            for (int i = 0; i < d; ++i) {
                if (coeff[i]) {
                    allZero = 0;
                    s(i) = coeff[i];
                    v += V.col(i) * coeff[i];
                }
            }
            if (allZero) return;
            Eigen::VectorXd p;
            for (int i = 0; i < canonical_points.cols(); ++i) {
                p = canonical_points.col(i) + v;
                Eigen::VectorXd c = A * p - b * 3;
                bool inside = 1;
                for (int j = 0; j < c.size(); ++j) {
                    if (c(j) > 1e-9) {
                        inside = 0;
                        break;
                    }
                }
                if (inside) {
                    points.push_back(p);
                    shifts.push_back(s);
                }
            }
            return;
        }
        for (int x = -3; x <= 3; ++x) {
            coeff.push_back(x);
            dfs(V, A, b, d, canonical_points, coeff, dfs);
            coeff.pop_back();
        }
    };

    int d = V.rows();

    for (int i = 0; i < canonical_points.cols(); ++i) {
        points.push_back(canonical_points.col(i));
        shifts.push_back(Eigen::VectorXi::Zero(d));
    }

    vector<int> coeff;
    dfs(V, A, b, d, canonical_points, coeff, dfs);

    Eigen::MatrixXd P(d, size(points));
    Eigen::MatrixXi S(d, size(shifts));
    for (int i = 0; i < size(points); ++i) {
        P.col(i) = points[i];
        S.col(i) = shifts[i];
    }

    return {P, S};
}

void testLatticeDiameter() {
    Eigen::MatrixXd U(2, 2);

    U(0, 0) = 1, U(0, 1) = 0;
    U(1, 0) = 0, U(1, 1) = 1;

    cout << LatticeDiameter(U) << "\n";
}

void testDirichletDomain2D() {
    Eigen::MatrixXd U(2, 2);

    U(0, 0) = 1, U(0, 1) = 0;
    U(1, 0) = 0, U(1, 1) = 1;

    auto D = DirichletDomain(U);

    cout << D.first << "\n";
    cout << D.second << "\n";
}

void testDirichletDomain3D() {
    Eigen::MatrixXd U = Eigen::MatrixXd::Identity(3, 3);

    auto D = DirichletDomain(U);

    cout << D.first << "\n";
    cout << D.second << "\n";
}

int main() 
{ 
    testDirichletDomain3D();
    // testLatticeDiameter();
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
   m.def("reduced_basis", &reducedBasis, "Reduce Lattice basis into reduced basis",
         py::arg("U"));
   m.def("dirichlet_domain", &DirichletDomain, "Compute Dirichlet domain of a lattice, input should be a 2x2 or 3x3 matrix representing the lattice basis",
         py::arg("U"));
   m.def("canonical_points", &canonicalPoints, "Compute the canonical copy of points in the Dirichlet domain",
         py::arg("A"), py::arg("b"), py::arg("points"));
   m.def("domain_x3_points", &domainX3Points, "Compute the periodic points in the 3X Dirichlet domain",
         py::arg("V"), py::arg("A"), py::arg("b"), py::arg("canonical_points"));
}
