#pragma once
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>

#include <CGAL/squared_distance_2.h> //for 2D functions
#include <CGAL/squared_distance_3.h> //for 3D functions

#include <CGAL/Epeck_d.h>
#include <CGAL/Random.h>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include <Eigen/Dense>

#include <vector>


namespace DELAUNAY {

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

Eigen::MatrixXi DelaunaySkeleton(const Eigen::MatrixXd& points);
Eigen::MatrixXi EuclideanMST(const Eigen::MatrixXd& points);
Eigen::MatrixXd reducedBasis(const Eigen::MatrixXd& U);
std::pair<Eigen::MatrixXd, Eigen::VectorXd> DirichletDomain(const Eigen::MatrixXd& V);

Eigen::MatrixXd canonicalPoints(
    const Eigen::MatrixXd& A, 
    const Eigen::VectorXd& b, 
    const Eigen::MatrixXd& points);

std::tuple<Eigen::MatrixXd, Eigen::VectorXi, Eigen::MatrixXi> pointsIn3xDomain(
    const Eigen::MatrixXd V, 
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b, 
    const Eigen::MatrixXd& canonical_points);

std::tuple<Eigen::MatrixXi, Eigen::VectorXd, Eigen::MatrixXi> periodicDelaunay(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXd& points);

}
