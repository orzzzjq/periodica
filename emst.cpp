#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

//#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include <CGAL/Random.h>

#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <map>
#include <chrono>

#define recordtime

#ifdef recordtime
auto start = std::chrono::high_resolution_clock::now();
auto stop = std::chrono::high_resolution_clock::now();
#define recordStart() start = std::chrono::high_resolution_clock::now();
#define recordStop(x) stop = std::chrono::high_resolution_clock::now(); fprintf(stderr, "%s: %dms\n", (x), std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
#else
#define recordStart() ;
#define recordStop(x) ;
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef K::Point_3                                                  Point;

typedef CGAL::Delaunay_triangulation_3<K>                           Triangulation;

using VertexIDPorperty = boost::property<boost::vertex_index_t, int>;
using EdgeWeight = boost::property<boost::edge_weight_t, double>;

typedef boost::adjacency_list<boost::vecS, boost::vecS,
    boost::undirectedS, VertexIDPorperty, EdgeWeight>               Graph;

typedef boost::graph_traits<Graph>::vertex_descriptor               vertex_descriptor;
typedef boost::graph_traits<Graph>::vertex_iterator                 vertex_iterator;
typedef boost::graph_traits<Graph>::edge_descriptor                 edge_descriptor;

typedef std::unordered_map<Triangulation::Vertex_handle, int>       VertexHandleIndexMap;
typedef std::unordered_map<vertex_descriptor, int>                  VertexDescriptorIndexMap;

typedef boost::associative_property_map<VertexDescriptorIndexMap>   VertexIdPropertyMap;

std::vector<Point> points;
using namespace std;

int main(int argc, char* argv[])
{
    int n = 100000;

    // generate random points
    CGAL::Random rd;
    for (int i = 0; i < n; ++i) {
        points.push_back(Point(rd.get_double(), rd.get_double(), rd.get_double()));
        //cout << i << ": (" << points.back() << ")\n";
    }

    recordStart();
    
    //Triangulation tr(begin(points), end(points));
    VertexHandleIndexMap vertex_handle_id;
    VertexDescriptorIndexMap vertex_descriptor_id;

    Triangulation tr;
    Graph g;
    
    int id = 0;
    for (auto& p : points) {
        Triangulation::Vertex_handle vh = tr.insert(p);
        vertex_handle_id[vh] = id;
        vertex_descriptor_id[add_vertex(id, g)] = id++;
    }

    recordStop("Delaunay triangulation");
    //printf("Is valid: %d\n", tr.is_valid());

    recordStart();

    auto sqr = [](double x) { return x * x; };

    for (Triangulation::Finite_edges_iterator it = tr.finite_edges_begin(); it != tr.finite_edges_end(); ++it) {
        int s = vertex_handle_id[it->first->vertex(it->second)];
        int t = vertex_handle_id[it->first->vertex(it->third)];
        double sqr_distance = sqr(points[s].x() - points[t].x()) + 
            sqr(points[s].y() - points[t].y()) + sqr(points[s].z() - points[t].z());
        //printf("insert %d %d: %.4f\n", s, t, sqr_distance);
        add_edge(s, t, sqr_distance, g);
    }

    //auto vertex_idMap = get(boost::vertex_index, g);
    //boost::graph_traits <Graph>::vertex_iterator i, end;
    //boost::graph_traits <Graph>::adjacency_iterator ai, a_end;

    //for (boost::tie(i, end) = vertices(g); i != end; ++i) {
    //    std::cout << vertex_idMap[*i] << ": ";

    //    for (boost::tie(ai, a_end) = adjacent_vertices(*i, g); ai != a_end; ++ai) {
    //        std::cout << vertex_idMap[*ai];
    //        if (boost::next(ai) != a_end)
    //            std::cout << ", ";
    //    }
    //    std::cout << std::endl;
    //}

    // Euclidean MST
    std::list<edge_descriptor> mst;
    boost::kruskal_minimum_spanning_tree(g, std::back_inserter(mst));

    recordStop("Euclidean MST");

    printf("Number of edges in MST: %d\n", mst.size());

    std::cout << "The edges of the Euclidean minimum spanning tree:" << std::endl;
    for (edge_descriptor ed : mst)
    {
        vertex_descriptor s = source(ed, g);
        vertex_descriptor t = target(ed, g);
        std::cout << "[ " << vertex_descriptor_id[s] << "  |  " << vertex_descriptor_id[t] << " ] " << std::endl;
    }

    return EXIT_SUCCESS;
}