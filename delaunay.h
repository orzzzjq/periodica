#pragma once
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/squared_distance_2.h>
#include <CGAL/squared_distance_3.h>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>

#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <vector>

using Kernel = CGAL::Epeck_d< CGAL::Dimension_tag<2> >;
using Point = Kernel::Point_d;
using Vector_of_points = std::vector<Point>;

// General kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;

// 2D
typedef K::Point_2                                                  Point2;
typedef CGAL::Delaunay_triangulation_2<K>                           Triangulation2;
typedef std::unordered_map<Triangulation2::Vertex_handle, int>      VertexHandleIdMap2;

// 3D
typedef K::Point_3                                                  Point3;
typedef CGAL::Delaunay_triangulation_3<K>                           Triangulation3;
typedef std::unordered_map<Triangulation3::Vertex_handle, int>      VertexHandleIdMap3;

// Weighted Graph
using VertexId = boost::property<boost::vertex_index_t, int>;
using EdgeWeight = boost::property<boost::edge_weight_t, double>;

typedef boost::adjacency_list<boost::vecS, boost::vecS,
    boost::undirectedS, VertexId, EdgeWeight>                       Graph;

typedef boost::graph_traits<Graph>::vertex_descriptor               VertexDescriptor;
typedef boost::graph_traits<Graph>::vertex_iterator                 VertexIterator;
typedef boost::graph_traits<Graph>::edge_descriptor                 EdgeDescriptor;
typedef boost::graph_traits<Graph>::edge_iterator                   EdgeIterator;

typedef std::unordered_map<VertexDescriptor, int>                   VertexDescriptorIdMap;


template <typename Point, typename Triangulation, typename VertexHandleIdMap>
inline void delaunay(
    const std::vector<Point>& points, 
    Triangulation& triangulation, 
    VertexHandleIdMap& vertexHandleId
) {
    vertexHandleId.clear();
    for (auto& p : points) {
        auto vh = triangulation.insert(p);
        vertexHandleId[vh] = vertexHandleId.size();
    }
}

inline void delaunay2DToGraph(
    const std::vector<Point2>& points, 
    const Triangulation2& triangulation, 
    VertexHandleIdMap2& vertexHandleId, 
    Graph& graph, 
    VertexDescriptorIdMap& vertexDescriptorId
) {
    for (auto it = triangulation.finite_edges_begin(); it != triangulation.finite_edges_end(); ++it) {
        int s = vertexHandleId[it->first->vertex((it->second + 1) % 3)];
        int t = vertexHandleId[it->first->vertex((it->second + 2) % 3)];
        if (vertexDescriptorId.find(s) == vertexDescriptorId.end()) {
            vertexDescriptorId[add_vertex(s, graph)] = s;
        }
        if (vertexDescriptorId.find(t) == vertexDescriptorId.end()) {
            vertexDescriptorId[add_vertex(t, graph)] = t;
        }
        double sq_distance = CGAL::squared_distance(points[s], points[t]);
        add_edge(s, t, sq_distance, graph);
    }
}

inline void delaunay3DToGraph(
    const std::vector<Point3>& points,
    const Triangulation3& triangulation,
    VertexHandleIdMap3& vertexHandleId,
    Graph& graph,
    VertexDescriptorIdMap& vertexDescriptorId
) {
    for (auto it = triangulation.finite_edges_begin(); it != triangulation.finite_edges_end(); ++it) {
        int s = vertexHandleId[it->first->vertex(it->second)];
        int t = vertexHandleId[it->first->vertex(it->third)];
        if (vertexDescriptorId.find(s) == vertexDescriptorId.end()) {
            vertexDescriptorId[add_vertex(s, graph)] = s;
        }
        if (vertexDescriptorId.find(t) == vertexDescriptorId.end()) {
            vertexDescriptorId[add_vertex(t, graph)] = t;
        }
        double sq_distance = CGAL::squared_distance(points[s], points[t]);
        add_edge(s, t, sq_distance, graph);
    }
}

//void delaunay3(const std::vector<Point3>& points, Triangulation3& triangulation, VertexHandleIdMap3& vertexHandleId);
