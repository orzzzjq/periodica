#include "debug.h"
#include "delaunay.h"

#include <CGAL/Random.h>

using namespace std;


void testDel2D() {
    int n = 4;
    std::vector<Point2> points;

    // generate random points
    CGAL::Random rd;
    for (int i = 0; i < n; ++i) {
        points.push_back(Point2(rd.get_double(), rd.get_double()));
    }

    // compute delaunay
    recordStart();

    Triangulation2 triangulation;
    VertexHandleIdMap2 vertexHandleId;

    delaunay(points, triangulation, vertexHandleId);

    recordStop("2D Delaunay triangulation");


    recordStart();

    Graph graph;
    VertexDescriptorIdMap vertexDescriptorId;

    delaunay2DToGraph(points, triangulation, vertexHandleId, graph, vertexDescriptorId);

    for (auto e = edges(graph).first; e != edges(graph).second; ++e) {
        std::cout << source(*e, graph) << "--" << target(*e, graph) << endl;
    }

    recordStop("2D Graph");
}

void testDel3D() {
    int n = 4;

    std::vector<Point3> points;
    // generate random points
    CGAL::Random rd;
    for (int i = 0; i < n; ++i) {
        points.push_back(Point3(rd.get_double(), rd.get_double(), rd.get_double()));
    }

    recordStart();

    Triangulation3 triangulation;
    VertexHandleIdMap3 vertexHandleId;

    delaunay(points, triangulation, vertexHandleId);

    recordStop("3D Delaunay triangulation");

    recordStart();

    Graph graph;
    VertexDescriptorIdMap vertexDescriptorId;

    delaunay3DToGraph(points, triangulation, vertexHandleId, graph, vertexDescriptorId);

    for (auto e = edges(graph).first; e != edges(graph).second; ++e) {
        std::cout << source(*e, graph) << "--" << target(*e, graph) << endl;
    }

    recordStop("3D Graph");
}

int xmain(int argc, char* argv[])
{
    testDel2D();
    testDel3D();
    //testDel3D();
    //int n = 1000;

    //std::vector<Point3> points;
    //// generate random points
    //CGAL::Random rd;
    //for (int i = 0; i < n; ++i) {
    //    points.push_back(Point3(rd.get_double(), rd.get_double(), rd.get_double()));
    //    //cout << i << ": (" << points.back() << ")\n";
    //}

    //recordStart();
    //
    ////Triangulation3 tr(begin(points), end(points));
    //VertexHandleIdMap vertex_handle_id;
    //VertexDescriptorIdMap vertex_descriptor_id;

    //Triangulation3 tr;
    //Graph g;
    //
    //int id = 0;
    //for (auto& p : points) {
    //    Triangulation3::Vertex_handle vh = tr.insert(p);
    //    vertex_handle_id[vh] = id;
    //    vertex_descriptor_id[add_vertex(id, g)] = id++;
    //}

    //recordStop("Delaunay triangulation");
    //////printf("Is valid: %d\n", tr.is_valid());

    ////recordStart();

    //auto sqr = [](double x) { return x * x; };

    //for (Triangulation3::Finite_edges_iterator it = tr.finite_edges_begin(); it != tr.finite_edges_end(); ++it) {
    //    int s = vertex_handle_id[it->first->vertex(it->second)];
    //    int t = vertex_handle_id[it->first->vertex(it->third)];
    //    double sqr_distance = sqr(points[s].x() - points[t].x()) + 
    //        sqr(points[s].y() - points[t].y()) + sqr(points[s].z() - points[t].z());
    //    //printf("insert %d %d: %.4f\n", s, t, sqr_distance);
    //    add_edge(s, t, sqr_distance, g);
    //}

    ////auto vertex_idMap = get(boost::vertex_index, g);
    ////boost::graph_traits <Graph>::vertex_iterator i, end;
    ////boost::graph_traits <Graph>::adjacency_iterator ai, a_end;

    ////for (boost::tie(i, end) = vertices(g); i != end; ++i) {
    ////    std::cout << vertex_idMap[*i] << ": ";

    ////    for (boost::tie(ai, a_end) = adjacent_vertices(*i, g); ai != a_end; ++ai) {
    ////        std::cout << vertex_idMap[*ai];
    ////        if (boost::next(ai) != a_end)
    ////            std::cout << ", ";
    ////    }
    ////    std::cout << std::endl;
    ////}

    //// Euclidean MST
    //std::list<edge_descriptor> mst;
    //boost::kruskal_minimum_spanning_tree(g, std::back_inserter(mst));

    //recordStop("Euclidean MST");

    //printf("Number of edges in MST: %d\n", mst.size());

    //std::cout << "The edges of the Euclidean minimum spanning tree:" << std::endl;
    //for (edge_descriptor ed : mst)
    //{
    //    vertex_descriptor s = source(ed, g);
    //    vertex_descriptor t = target(ed, g);
    //    std::cout << "[ " << vertex_descriptor_id[s] << "  |  " << vertex_descriptor_id[t] << " ] " << std::endl;
    //}

    return EXIT_SUCCESS;
}