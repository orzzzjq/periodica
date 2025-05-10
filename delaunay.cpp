#include "debug.h"
//#include "delaunay.h"
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>

#include <CGAL/squared_distance_2.h> //for 2D functions
#include <CGAL/squared_distance_3.h> //for 3D functions

#include <CGAL/Epeck_d.h>
#include <CGAL/Random.h>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

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

void testGudhiAlpha2D() {
    int n = 10000;
    std::vector<Point2> points;

    // ---------- generate random points ----------
    CGAL::Random rd;
    for (int i = 0; i < n; ++i) {
        points.push_back(Point2(rd.get_double(), rd.get_double()));
    }

    // ---------- compute delaunay ----------
    recordStart();

    Gudhi::alpha_complex::Alpha_complex<K2> alphaComplex(points);

    recordStop("2D Delaunay triangulation");

    // ---------- create simplical complex ----------
    recordStart();

    Gudhi::Simplex_tree complex;
    alphaComplex.create_complex(complex, INFINITY, false, true);

    recordStop("Create complex");

    // ---------- compute filtration values ----------
    recordStart();

    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            auto dx = points[id[0]][0] - points[id[1]][0], dy = points[id[0]][1] - points[id[1]][1];
            auto sq_dist = (dx * dx + dy * dy);
            complex.assign_filtration(simplex, to_double(sq_dist));
        }
        else {
            complex.assign_filtration(simplex, 0);
        }
    }

    recordStop("Assign weights");

    // ---------- construct weighted graph ----------
    recordStart();

    Graph graph;

    for (int i = 0; i < size(points); ++i) {
        add_vertex(i, graph);
    }
    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            add_edge(id[0], id[1], complex.filtration(simplex), graph);
        }
    }

    recordStop("2D Graph");

    // ---------- compute minimum spanning tree ----------
    recordStart();

    std::list<EdgeDescriptor> mst;
    boost::kruskal_minimum_spanning_tree(graph, std::back_inserter(mst));

    recordStop("Minimum spanning tree");

    // ---------- debugging ----------
    
    //cout << "Number of 1D faces in complex: " << complex.num_simplices_by_dimension()[1] << "\n";
    //cout << "Number of edges in graph: " << num_edges(graph) << "\n";

    //cout << "Graph edges:\n";
    //for (auto simplex : complex.skeleton_simplex_range(1)) {
    //    if (complex.dimension(simplex) == 0) continue;
    //    cout << "\t";
    //    for (auto v : complex.simplex_vertex_range(simplex)) {
    //        cout << v << " ";
    //    }
    //    cout << ": " << complex.filtration(simplex) << "\n";
    //}

    //cout << "Number of edges in MST: " << mst.size() << "\n";
    //cout << "MST edges:\n";
    //for (EdgeDescriptor ed : mst)
    //{
    //    VertexDescriptor s = source(ed, graph);
    //    VertexDescriptor t = target(ed, graph);
    //    std::cout << "\t" << s << "--" << t << "\n";
    //}
}

void testGudhiAlpha3D() {
    int n = 5;
    std::vector<Point3> points;

    // ---------- generate random points ----------
    CGAL::Random rd;
    for (int i = 0; i < n; ++i) {
        points.push_back(Point3(rd.get_double(), rd.get_double(), rd.get_double()));
    }

    // ---------- compute delaunay ----------
    recordStart();

    Gudhi::alpha_complex::Alpha_complex<K3> alphaComplex(points);

    recordStop("3D Delaunay triangulation");

    // ---------- create simplical complex ----------
    recordStart();

    Gudhi::Simplex_tree complex;
    alphaComplex.create_complex(complex, INFINITY, false, true);

    recordStop("Create complex");

    // ---------- compute filtration values ----------
    recordStart();

    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            auto dx = points[id[0]][0] - points[id[1]][0], dy = points[id[0]][1] - points[id[1]][1];
            auto sq_dist = (dx * dx + dy * dy);
            complex.assign_filtration(simplex, to_double(sq_dist));
        }
        else {
            complex.assign_filtration(simplex, 0);
        }
    }

    recordStop("Assign weights");

    // ---------- construct weighted graph ----------
    recordStart();

    Graph graph;

    for (int i = 0; i < size(points); ++i) {
        add_vertex(i, graph);
    }
    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            add_edge(id[0], id[1], complex.filtration(simplex), graph);
        }
    }

    recordStop("3D Graph");

    // ---------- compute minimum spanning tree ----------
    recordStart();

    std::list<EdgeDescriptor> mst;
    boost::kruskal_minimum_spanning_tree(graph, std::back_inserter(mst));

    recordStop("Minimum spanning tree");

    // ---------- debugging ----------

    cout << "Number of 1D faces in complex: " << complex.num_simplices_by_dimension()[1] << "\n";
    cout << "Number of edges in graph: " << num_edges(graph) << "\n";

    cout << "Graph edges:\n";
    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex) == 0) continue;
        cout << "\t";
        for (auto v : complex.simplex_vertex_range(simplex)) {
            cout << v << " ";
        }
        cout << ": " << complex.filtration(simplex) << "\n";
    }

    cout << "Number of edges in MST: " << mst.size() << "\n";
    cout << "MST edges:\n";
    for (EdgeDescriptor ed : mst)
    {
        VertexDescriptor s = source(ed, graph);
        VertexDescriptor t = target(ed, graph);
        std::cout << "\t" << s << "--" << t << "\n";
    }
}

int main() 
{
    //testGudhiAlpha2D();
    testGudhiAlpha3D();
}
