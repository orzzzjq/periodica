#include "auxiliary.h"
#include "delaunay.h"

#include <CGAL/number_utils.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace DELAUNAY {
using namespace std;

// Compute the 1-skeleton of the Delaunay triangulation
// Input:
//  Points: MatrixXd(d, N)
// Output:
//  Delaunay edges: MatrixXi(M, 2)
//  * Here M is the number of Delaunay edges
Eigen::MatrixXi DelaunaySkeleton(
    const Eigen::MatrixXd& points
) {
    if (points.rows() != 2 && points.rows() != 3) {
        throw std::invalid_argument("Input must be a 2D NumPy array of shape (2, n) or (3, n)");
    }

    int n = points.cols(), d = points.rows();

    Gudhi::Simplex_tree<> complex;

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

// Compute the weighted Delaunay triangulation
// Input:
//  Points: MatrixXd(d, N)
//  Weights: VectorXd(N)
// Output:
//  Delaunay complex: Gudhi::Simplex_tree<>
Gudhi::Simplex_tree<> DelaunayComplex(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& weights
) {
    if (points.rows() != 2 && points.rows() != 3) {
        throw std::invalid_argument("Input must be a 2D NumPy array of shape (2, n) or (3, n)");
    }

    int n = points.cols(), d = points.rows();
    if (weights.size() != n) {
        throw std::invalid_argument("weights size must be equal to the number of points");
    }

    Gudhi::Simplex_tree<> complex;

    if (d == 2) {
        vector<double> coord(2);
        vector<Point2> p;
        vector<double> w;
        p.reserve(n);
        w.reserve(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                coord[j] = points(j, i);
            }
            p.push_back(Point2(coord[0], coord[1]));
            w.push_back(weights(i));
        }
        Gudhi::alpha_complex::Alpha_complex<K2, true> alphaComplex(p, w);
        alphaComplex.create_complex(complex, INFINITY, false, false);
    }
    else {
        vector<double> coord(3);
        vector<Point3> p;
        vector<double> w;
        p.reserve(n);
        w.reserve(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                coord[j] = points(j, i);
            }
            p.push_back(Point3(coord[0], coord[1], coord[2]));
            w.push_back(weights(i));
        }
        Gudhi::alpha_complex::Alpha_complex<K3, true> alphaComplex(p, w);
        alphaComplex.create_complex(complex, INFINITY, false, false);
    }

    return complex;
}

// Compute the 1-skeleton of the weighted Delaunay triangulation
// Input:
//  Points: MatrixXd(d, N)
//  Weights: VectorXd(N)
// Output:
//  Delaunay edges: MatrixXi(M, 2)
//  * Here M is the number of Delaunay edges
Eigen::MatrixXi DelaunaySkeleton(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& weights
) {

    Gudhi::Simplex_tree<> complex = DelaunayComplex(points, weights);

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



// Compute the Euclidean MST of a point set
Eigen::MatrixXi EuclideanMST(
    const Eigen::MatrixXd& points
) {
    if (points.rows() != 2 && points.rows() != 3) {
        throw std::invalid_argument("Input must be a 2D NumPy array of shape (2, n) or (3, n)");
    }

    int n = points.cols(), d = points.rows();

    Gudhi::Simplex_tree<> complex;

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

// Compute diameter of unit cell of the lattice
double LatticeDiameter(
    const Eigen::MatrixXd& U
) {
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

// Compute reduced basis
// Input:
//  Original basis U: MatrixXd(d, d)
// Output:
//  Reduced basis V: MatrixXd(d, d)
Eigen::MatrixXd reducedBasis(
    const Eigen::MatrixXd& U
) {
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
            V.col(2) = -V.col(0) -V.col(1);
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

// Compute Dirichlet domain from reduced basis
// Input:
//  Reduced basis V: MatrixXd(d, d + 1)
// Ouput:
//  Ax <= b
//  Coefficient matrix A: MatrixXd(m, d)
//  Right-hand side b: VectorXd(m)
pair<Eigen::MatrixXd, Eigen::VectorXd> DirichletDomain(
    const Eigen::MatrixXd& V    // reduced basis
) {
    if (V.rows() != V.cols() - 1 || (V.rows() != 2 && V.rows() != 3)) {
        throw std::invalid_argument("Input must be a reduced basis with dimension (2, 3) or (3, 4)");
    }

    int d = V.rows();

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

// Compute canonical points in the Dirichlet domain
// Input:
//  Dirichlet domain parameterized by Ax <= b, where A is MatrixXd(m, d), b is VectorXd(m)
//  Points in original unit cell: MatrixXd(d, n)
// Output:
//  Canonical points: MatrixXd(d, n)
Eigen::MatrixXd canonicalPoints(
    const Eigen::MatrixXd& A, 
    const Eigen::VectorXd& b, 
    const Eigen::MatrixXd& points
) {
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

// Compute periodic copies of the canonical points in the 3x Dirichlet domain
// Input:
//  Reduced basis V: MatrixXd(d, d + 1)
//  Dirichlet domain parameterized by Ax <= b, where A is MatrixXd(m, d), b is VectorXd(m)
//  Canonical points: MatrixXd(d, n)
// Output:
//  Points in the 3x domain: MatrixXd(d, N)
//  Indices of the canonical copy: VectorXi(N)
//  Shift vectors of the point: MatrixXi(d, N)
//  * Here N is the number of point in the 3x domain
std::tuple<Eigen::MatrixXd, Eigen::VectorXi, Eigen::MatrixXi> pointsIn3xDomain(
    const Eigen::MatrixXd V, 
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b, 
    const Eigen::MatrixXd& canonical_points
) {
    if (A.cols() != canonical_points.rows() || (A.cols() != 2 && A.cols() != 3)) {
        throw std::invalid_argument("Invalid input dimension");
    }

    vector<Eigen::VectorXd> points;
    vector<int> indices;
    vector<Eigen::VectorXi> shifts;

    auto dfs = [&points, &indices, &shifts](const Eigen::MatrixXd V, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, int d,
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
                Eigen::VectorXd c = A * p;
                bool inside = 1;
                for (int j = 0; j < c.size(); ++j) {
                    if (c(j) / (b(j) * 3) > 1 + 1e-9) {
                        inside = 0;
                        break;
                    }
                }
                if (inside) {
                    points.push_back(p);
                    indices.push_back(i);
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
        indices.push_back(i);
        shifts.push_back(Eigen::VectorXi::Zero(d));
    }

    vector<int> coeff;
    dfs(V, A, b, d, canonical_points, coeff, dfs);

    Eigen::MatrixXd P(d, size(points));
    Eigen::VectorXi I(size(points));
    Eigen::MatrixXi S(d, size(shifts));
    for (int i = 0; i < size(points); ++i) {
        P.col(i) = points[i];
        I(i) = indices[i];
        S.col(i) = shifts[i];
    }

    return {P, I, S};
}

// Compute quotient complex from periodic Delaunay complex
// Output:
//  Delaunay edges: MatrixXi(m, 2)
//  Filtration values: VectorXd(m)
//  Shift vectors: MatrixXi(d, m)
std::tuple<Eigen::MatrixXi, Eigen::VectorXd, Eigen::MatrixXi> periodicDelaunay(
    const Eigen::MatrixXd& U,       // lattice basis
    const Eigen::MatrixXd& points,  // points in unit cell
    const Eigen::VectorXd& weights  // weights of points in unit cell
) {
    if (U.cols() != U.rows() || U.rows() != points.rows()) {
        throw std::invalid_argument("Invalid input");
    }
    if (weights.size() != points.cols()) {
        throw std::invalid_argument("weights size must be equal to the number of points");
    }

    int d = points.rows(), n = points.cols();
    
    // Reduced basis
    auto V = reducedBasis(U);
    
    // Dirichlet domain
    auto [A, b] = DirichletDomain(V);

    // Canonical points in the Dirichlet domain
    auto canonical_points = canonicalPoints(A, b, points);

    // Points in the 3x Dirichlet domain, together with original index and shift vectors
    auto [working_points, I, S] = pointsIn3xDomain(V, A, b, canonical_points);
    Eigen::VectorXd working_weights(working_points.cols());
    for (int i = 0; i < working_points.cols(); ++i) {
        working_weights(i) = weights(I(i));
    }

    // Weighted Delaunay complex from points in the 3x domain
    Gudhi::Simplex_tree<> complex = DelaunayComplex(working_points, working_weights);

    vector<vector<int>> delaunay_edges;
    vector<double> e_filtrations;
    // vector<double> v_filtrations(n, 0);

    // for (auto s : complex.skeleton_simplex_range(0)) {
    //     for (auto v : complex.simplex_vertex_range(s)) {
    //         int id = static_cast<int>(v);
    //         if (id < n) {
    //             v_filtrations[id] = (complex.filtration(s));
    //         }
    //         // printf("[debug] %d, ", id);
    //     }
    //     // printf("\n");
    // }

    // printf("[debug] ");
    // for (auto x : v_filtrations) {
    //     printf("%f, ", x);
    // }
    // printf("\n");

    for (auto simplex : complex.skeleton_simplex_range(1)) {
        if (complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            delaunay_edges.push_back(id);
            e_filtrations.push_back(sqrt(complex.filtration(simplex)));
        }
    }

    // Filter the periodic edges (have at least one end point in the 1x domain
    vector<pair<int,int>> quotient_edges;
    vector<double> quotient_filtrations;
    unordered_set<string> shift_set;
    for (int i = 0; i < delaunay_edges.size(); ++i) {
        int s = delaunay_edges[i][0], t = delaunay_edges[i][1];
        if (s < n || t < n) {
            if (s > t) swap(s, t); // let the first point be the one with smaller index
            if (t >= n && s > I(t)) continue;
            if (s == I(t)) { // If it's a self-loop, check if the opposite direction is already inserted
                string shift_key, opposite_key;
                shift_key.reserve(static_cast<size_t>(d + 1) * 8);
                opposite_key.reserve(static_cast<size_t>(d + 1) * 8);
                shift_key += to_string(s);
                opposite_key += to_string(s);
                for (int j = 0; j < d; ++j) {
                    shift_key.push_back(',');
                    opposite_key.push_back(',');
                    shift_key += to_string(S(j, t));
                    opposite_key += to_string(-S(j, t));
                }
                if (shift_set.find(opposite_key) != shift_set.end()) {
                    continue;
                }
                shift_set.insert(shift_key);
            }
            quotient_edges.push_back({s, t});
            quotient_filtrations.push_back(e_filtrations[i]);
        }
    }

    // Get the results
    int M = size(quotient_edges);
    Eigen::MatrixXi edges(M, 2);
    Eigen::VectorXd filtration(M);
    Eigen::MatrixXi shift(d, M);

    for (int i = 0; i < M; ++i) {
        // edge with original index
        auto [s, t] = quotient_edges[i];
        edges(i, 0) = I(s);
        edges(i, 1) = I(t);
        
        filtration(i) = quotient_filtrations[i];

        // shift vector
        shift.col(i) = S.col(t);
    }

    return {edges, filtration, shift};
}

Eigen::VectorXd circumCenter(const vector<Eigen::VectorXd>& vertices) {
    if (vertices.empty()) {
        throw std::invalid_argument("vertices must be non-empty");
    }

    int d = static_cast<int>(vertices[0].size());
    if (d != 2 && d != 3) {
        throw std::invalid_argument("Only 2D and 3D circumcenters are supported");
    }
    if (static_cast<int>(vertices.size()) != d + 1) {
        throw std::invalid_argument("A full simplex must have d+1 vertices");
    }
    for (const auto& v : vertices) {
        if (static_cast<int>(v.size()) != d) {
            throw std::invalid_argument("Inconsistent vertex dimensions");
        }
    }

    Eigen::VectorXd result = Eigen::VectorXd::Zero(d);
    if (d == 2) {
        K2 kernel;
        vector<Point2> pts;
        pts.reserve(3);
        for (const auto& v : vertices) {
            pts.emplace_back(v(0), v(1));
        }
        Point2 center = kernel.construct_circumcenter_d_object()(pts.begin(), pts.end());
        auto coord = kernel.compute_coordinate_d_object();
        for (int j = 0; j < d; ++j) {
            result(j) = CGAL::to_double(coord(center, j));
        }
    } else {
        K3 kernel;
        vector<Point3> pts;
        pts.reserve(4);
        for (const auto& v : vertices) {
            pts.emplace_back(v(0), v(1), v(2));
        }
        Point3 center = kernel.construct_circumcenter_d_object()(pts.begin(), pts.end());
        auto coord = kernel.compute_coordinate_d_object();
        for (int j = 0; j < d; ++j) {
            result(j) = CGAL::to_double(coord(center, j));
        }
    }

    return result;
}

struct simplexVertex {
    vector<double> p;
    int id;
    bool operator<(const simplexVertex& other) const {
        if (p[0] != other.p[0]) return p[0] < other.p[0];
        if (p[1] != other.p[1]) return p[1] < other.p[1];
        return p[2] < other.p[2];
    }
};

// Compute the periodic Voronoi complex
// Output:
//  Voronoi points: MatrixXd(d, l)
//  Voronoi edges: MatrixXi(m, 2)
//  Point filtration values
//  Edge filtration values
//  Edge (arc) shift vectors
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXi> periodicVoronoi(
    const Eigen::MatrixXd& U,       // lattice basis
    const Eigen::MatrixXd& points,  // points in unit cell
    const Eigen::VectorXd& weights, // weights of points in unit cell
    bool useCircumCenter
) {
    if (U.cols() != U.rows() || U.rows() != points.rows()) {
        throw std::invalid_argument("Invalid input");
    }
    if (weights.size() != points.cols()) {
        throw std::invalid_argument("weights size must be equal to the number of points");
    }

    int d = points.rows(), n = points.cols();
    
    // Reduced basis
    auto V = reducedBasis(U);
    
    // Dirichlet domain
    auto [A, b] = DirichletDomain(V);

    // Canonical points in the Dirichlet domain
    auto canonical_points = canonicalPoints(A, b, points);

    // Points in the 3x Dirichlet domain, together with original index and shift vectors
    auto [working_points, I, S] = pointsIn3xDomain(V, A, b, canonical_points);
    Eigen::VectorXd working_weights(working_points.cols());
    for (int i = 0; i < working_points.cols(); ++i) {
        working_weights(i) = weights(I(i));
    }

    Gudhi::Simplex_tree<> delaunay_complex = DelaunayComplex(working_points, working_weights);

    // Construct vertex information (canonical index, shift vector)
    vector<vector<int>> v_info;
    for (int i = 0; i < working_points.cols(); ++i) {
        vector<int> info;
        info.push_back(I(i)); // canonical index
        for (int j = 0; j < d; ++j) { // shift vector
            info.push_back(S.col(i)(j));
        }
        v_info.push_back(info);
    }

    // Get d-dimensional simplices
    unordered_map<string, int> s_hashmap;
    vector<vector<int>> s_vertices;
    vector<Eigen::VectorXi> s_shifts;
    vector<bool> is_canonical;
    vector<Eigen::VectorXd> voronoi_points;
    vector<double> voronoi_point_filtrations;
    
    // Map vertices information of a simplex to a string
    auto str = [](const vector<vector<int>>& s_info) {
        string key;
        key.reserve(s_info.size() * 10);
        for (auto row : s_info) {
            for (auto x : row) {
                key += to_string(x);
                key += ",";
            }
        }
        return key;
    };

    // Compute necessary information of the d-simplices
    int s_id = -1; // simplex index
    for (auto simplex : delaunay_complex.skeleton_simplex_range(d)) {
        if (delaunay_complex.dimension(simplex) != d) continue;

        s_id += 1;
        vector<int> s_verts; // the indices of the vertices of the simplex
        vector<vector<int>> s_info; // the array of v_info of the vertices
        for (auto v : delaunay_complex.simplex_vertex_range(simplex)) {
            int vi = static_cast<int>(v);
            s_verts.push_back(vi);
            s_info.push_back(v_info[vi]);
        }

        // Sort the v_info in lexicographical order
        sort(s_info.begin(), s_info.end());

        // Get the shift vector of the representative and check if it's canonical
        bool is_zero = true;
        Eigen::VectorXi s_shift(d);
        for (int i = 0; i < d; ++i) {
            s_shift(i) = s_info[0][i + 1];
            if (s_shift(i) != 0) is_zero = false;
        }

        is_canonical.push_back(is_zero);
        s_shifts.push_back(s_shift);        
        s_hashmap.emplace(str(s_info), s_id);
        s_vertices.push_back(s_verts);
        voronoi_point_filtrations.push_back(-sqrt(delaunay_complex.filtration(simplex)));
        
        // Compute the center of the simplex, for visualization purpose
        Eigen::VectorXd center = Eigen::VectorXd::Zero(d);
        if (useCircumCenter) {
            vector<Eigen::VectorXd> s_points;
            s_points.reserve(s_verts.size());
            for (int vi : s_verts) {
                s_points.push_back(working_points.col(vi));
            }
            center = circumCenter(s_points);
        } else {
            for (int vi : s_verts) {
                center += working_points.col(vi);
            }
            center /= static_cast<double>(s_verts.size());
        }
        voronoi_points.push_back(std::move(center));
    }

    // Find the canonical copy of the simplices
    vector<int> s_I(is_canonical.size()); // the canonical index of the simplex
    map<int, int> canonical_id;
    vector<int> canonical_voro_points;
    // vector<double> voronoi_point_filtrations;
    for (int i = 0; i < s_id + 1; ++i) {
        if (is_canonical[i]) {
            canonical_id[i] = canonical_voro_points.size();
            canonical_voro_points.push_back(i);
            // voronoi_point_filtrations.push_back(s_filtrations[i]);
        }
        vector<vector<int>> s_info;
        for (auto vi : s_vertices[i]) {
            s_info.push_back(v_info[vi]);
            for (int j = 0; j < d; ++j) {
                s_info.back()[j + 1] -= s_shifts[i](j);
            }
        }
        sort(s_info.begin(), s_info.end());
        string key = str(s_info);
        if (s_hashmap.find(key) != s_hashmap.end()) {
            s_I[i] = s_hashmap[key];
        } else {
            s_I[i] = -1;
        }
    }

    // Voronoi edges connect adjacent d-simplices that share a (d-1)-face.
    vector<pair<int,int>> voronoi_edges;
    vector<double> voronoi_edge_filtrations;
    for (auto simplex : delaunay_complex.skeleton_simplex_range(d-1)) {
        if (delaunay_complex.dimension(simplex) != d-1) continue;

        vector<int> cofaces;
        for (auto coface : delaunay_complex.cofaces_simplex_range(simplex, 1)) {
            // coface is a d-simplex handle
            vector<vector<int>> s_info; // the array of v_info of the vertices
            for (auto v : delaunay_complex.simplex_vertex_range(coface)) {
                int vi = static_cast<int>(v);
                s_info.push_back(v_info[vi]);
            }
            sort(s_info.begin(), s_info.end());
            cofaces.push_back(s_hashmap[str(s_info)]);
        }

        // The simplex should have two cofaces
        if (cofaces.size() == 2) {
            voronoi_edges.push_back({cofaces[0], cofaces[1]});
            voronoi_edge_filtrations.push_back(-sqrt(delaunay_complex.filtration(simplex)));
        }
    }

    // Re-label the canonical voronoi points
    for (int i = 0; i < is_canonical.size(); ++i) {
        if (s_I[i] != -1) {
            s_I[i] = canonical_id[s_I[i]];
        }
    }

    // Get the periodic voronoi edges
    vector<int> periodic_voro_edges;
    for (int i = 0; i < voronoi_edges.size(); ++i) {
        auto [s,t] = voronoi_edges[i];
        if (s_I[s] == -1 || s_I[t] == -1) continue;
        if (is_canonical[s] || is_canonical[t]) {
            periodic_voro_edges.push_back(i);
        }
    }

    // Organize the results
    int L = size(canonical_voro_points);
    int M = size(periodic_voro_edges);
    Eigen::MatrixXd res_voronoi_points(d, L);
    Eigen::MatrixXi res_voronoi_edges(M, 2);
    Eigen::VectorXd res_point_filtrations(L);
    Eigen::VectorXd res_edge_filtrations(M);
    Eigen::MatrixXi res_edge_shifts(d, M);

    for (int i = 0; i < L; ++i) {
        res_voronoi_points.col(i) = voronoi_points[canonical_voro_points[i]];
        res_point_filtrations(i) = voronoi_point_filtrations[canonical_voro_points[i]];
    }

    for (int i = 0; i < M; ++i) {
        auto [s, t] = voronoi_edges[periodic_voro_edges[i]];
        if (!is_canonical[s]) swap(s,t);
        res_voronoi_edges(i, 0) = s_I[s];
        res_voronoi_edges(i, 1) = s_I[t];
        res_edge_filtrations(i) = voronoi_edge_filtrations[periodic_voro_edges[i]];
        res_edge_shifts.col(i) = s_shifts[t];
    }

    return {res_voronoi_points, res_voronoi_edges, res_point_filtrations, res_edge_filtrations, res_edge_shifts};
}


// Compute the Voronoi 1-skeleton of the points in 3x Direchlet region
// Output:
//  Voronoi points: MatrixXd(d, l)
//  Voronoi edges: MatrixXi(m, 2)
//  **todo Filtration values: VectorXd(m)
//  **todo Shift vectors: MatrixXi(d, m)
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> fullVoronoiSkeleton(
    const Eigen::MatrixXd& U,       // lattice basis
    const Eigen::MatrixXd& points,  // points in unit cell
    const Eigen::VectorXd& weights, // weights of points in unit cell
    bool useCircumCenter
) {
    if (U.cols() != U.rows() || U.rows() != points.rows()) {
        throw std::invalid_argument("Invalid input");
    }
    if (weights.size() != points.cols()) {
        throw std::invalid_argument("weights size must be equal to the number of points");
    }

    int d = points.rows(), n = points.cols();
    
    // Reduced basis
    auto V = reducedBasis(U);
    
    // Dirichlet domain
    auto [A, b] = DirichletDomain(V);

    // Canonical points in the Dirichlet domain
    auto canonical_points = canonicalPoints(A, b, points);

    // Points in the 3x Dirichlet domain, together with original index and shift vectors
    auto [working_points, I, S] = pointsIn3xDomain(V, A, b, canonical_points);
    Eigen::VectorXd working_weights(working_points.cols());
    for (int i = 0; i < working_points.cols(); ++i) {
        working_weights(i) = weights(I(i));
    }

    Gudhi::Simplex_tree<> delaunay_complex = DelaunayComplex(working_points, working_weights);

    // Get d-dimensional simplices
    unordered_map<int64_t, int> simplex_id_map;
    vector<vector<int>> simplex_vertices;
    vector<Eigen::VectorXd> voronoi_points;
    vector<bool> simplex_in_1x_domain;
    
    auto simplexKey = [](const vector<int>& vertices) {
        int64_t key = 0;
        for (size_t i = 0; i < vertices.size(); ++i) {
            key |= vertices[i];
            key <<= 20;
        }
        return key;
    };

    for (auto simplex : delaunay_complex.skeleton_simplex_range(d)) {
        if (delaunay_complex.dimension(simplex) != d) continue;

        vector<int> vertices;
        bool in_1x_domain = false;
        for (auto v : delaunay_complex.simplex_vertex_range(simplex)) {
            int vi = static_cast<int>(v);
            vertices.push_back(vi);
            if (vi < n) in_1x_domain = true;
        }

        sort(vertices.begin(), vertices.end());
        Eigen::VectorXd center = Eigen::VectorXd::Zero(d);
        if (useCircumCenter) {
            vector<Eigen::VectorXd> simplex_points;
            simplex_points.reserve(vertices.size());
            for (int vi : vertices) {
                simplex_points.push_back(working_points.col(vi));
            }
            center = circumCenter(simplex_points);
        } else {
            for (int vi : vertices) {
                center += working_points.col(vi);
            }
            center /= static_cast<double>(vertices.size());
        }

        int id = static_cast<int>(simplex_vertices.size());
        simplex_id_map.emplace(simplexKey(vertices), id);
        simplex_vertices.push_back(std::move(vertices));
        voronoi_points.push_back(std::move(center));
        simplex_in_1x_domain.push_back(in_1x_domain);
    }

    // Voronoi edges connect adjacent d-simplices that share a (d-1)-face.
    vector<pair<int,int>> voronoi_edges;
    for (auto simplex : delaunay_complex.skeleton_simplex_range(d-1)) {
        if (delaunay_complex.dimension(simplex) != d-1) continue;

        vector<int> cofaces;
        for (auto coface : delaunay_complex.cofaces_simplex_range(simplex, 1)) {
            // coface is a d-simplex handle
            std::vector<int> verts;
            for (auto v : delaunay_complex.simplex_vertex_range(coface)) {
                verts.push_back(static_cast<int>(v));
            }
            sort(verts.begin(), verts.end());
            cofaces.push_back(simplex_id_map[simplexKey(verts)]);
        }

        if (cofaces.size() == 2) {
            // The simplex has two cofaces
            voronoi_edges.push_back({cofaces[0], cofaces[1]});
        }
    }

    // Get the results
    int L = size(voronoi_points);
    int M = size(voronoi_edges);
    Eigen::MatrixXd v_points(d, L);
    Eigen::MatrixXi v_edges(M, 2);
    // Eigen::VectorXd filtration(M);
    // Eigen::MatrixXi shift(d, M);

    for (int i = 0; i < L; ++i) {
        v_points.col(i) = voronoi_points[i];
    }

    for (int i = 0; i < M; ++i) {
        // Voronoi edge between dual points of d-simplices.
        auto [s, t] = voronoi_edges[i];
        v_edges(i, 0) = s;
        v_edges(i, 1) = t;
    }

    return {v_points, v_edges};
}


// Compute the Delaunay 1-skeleton of the points in 3x Direchlet region
// Output:
//  Delaunay points: MatrixXd(d, l)
//  Delaunay edges: MatrixXi(m, 2)
//  **todo Filtration values: VectorXd(m)
//  **todo Shift vectors: MatrixXi(d, m)
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> fullDelaunaySkeleton(
    const Eigen::MatrixXd& U,       // lattice basis
    const Eigen::MatrixXd& points,  // points in unit cell
    const Eigen::VectorXd& weights  // weights of points in unit cell
) {
    if (U.cols() != U.rows() || U.rows() != points.rows()) {
        throw std::invalid_argument("Invalid input");
    }
    if (weights.size() != points.cols()) {
        throw std::invalid_argument("weights size must be equal to the number of points");
    }

    int d = points.rows(), n = points.cols();
    
    // Reduced basis
    auto V = reducedBasis(U);
    
    // Dirichlet domain
    auto [A, b] = DirichletDomain(V);

    // Canonical points in the Dirichlet domain
    auto canonical_points = canonicalPoints(A, b, points);

    // Points in the 3x Dirichlet domain, together with original index and shift vectors
    auto [working_points, I, S] = pointsIn3xDomain(V, A, b, canonical_points);
    Eigen::VectorXd working_weights(working_points.cols());
    for (int i = 0; i < working_points.cols(); ++i) {
        working_weights(i) = weights(I(i));
    }

    Gudhi::Simplex_tree<> delaunay_complex = DelaunayComplex(working_points, working_weights);

    // Delaunay edges
    vector<pair<int,int>> delaunay_edges;
    for (auto simplex : delaunay_complex.skeleton_simplex_range(1)) {
        if (delaunay_complex.dimension(simplex)) {
            vector<int> id;
            for (auto v : delaunay_complex.simplex_vertex_range(simplex)) {
                id.push_back(int(v));
            }
            delaunay_edges.push_back({id[0], id[1]});
        }
    }

    Eigen::MatrixXi del_edges(size(delaunay_edges), 2);

    for (size_t i = 0; i < size(delaunay_edges); ++i) {
        auto [s, t] = delaunay_edges[i];
        del_edges(i, 0) = s;
        del_edges(i, 1) = t;
    }

    return {working_points, del_edges};
}

} // End of namespace DELAUNAY
