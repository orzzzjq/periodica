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
    auto delaunay_edges = DelaunaySkeleton(working_points, working_weights);

    // Filter the periodic edges (have at least one end point in the 1x domain
    vector<pair<int,int>> quotient_edges;
    unordered_set<string> shift_set;
    for (int i = 0; i < delaunay_edges.rows(); ++i) {
        int s = delaunay_edges(i, 0), t = delaunay_edges(i, 1);
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
        
        // filtration value
        double sq_dist = 0;
        if (d == 2) {
            double dx = working_points(0, s) - working_points(0, t), dy = working_points(1, s) - working_points(1, t);
            sq_dist = dx * dx + dy * dy;
        }
        else {
            double dx = working_points(0, s) - working_points(0, t), dy = working_points(1, s) - working_points(1, t), dz = working_points(2, s) - working_points(2, t);
            sq_dist = dx * dx + dy * dy + dz * dz;
        }
        filtration(i) = sqrt(sq_dist);

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
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::VectorXd, Eigen::VectorXd> periodicVoronoi(
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
    vector<Eigen::VectorXi> simplex_shifts;
    vector<Eigen::VectorXd> voronoi_points;
    vector<double> v_point_filtrations;
    vector<bool> canonical_simplex;
    
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
        vector<simplexVertex> s_vertices;
        for (auto v : delaunay_complex.simplex_vertex_range(simplex)) {
            int vi = static_cast<int>(v);
            vertices.push_back(vi);
            vector<double> p;
            for (int i = 0; i < d; ++i) {
                p.push_back(working_points(i, vi));
            }
            s_vertices.push_back(simplexVertex(p, vi));
        }

        simplexVertex s_repr = s_vertices[0];
        for (int i = 1; i < s_vertices.size(); ++i) {
            if (s_vertices[i] < s_repr) {
                s_repr = s_vertices[i];
            }
        }

        int s_repr_id = s_repr.id;
        canonical_simplex.push_back(s_repr_id < n);
        simplex_shifts.push_back(S.col(s_repr_id));

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
        v_point_filtrations.push_back(-sqrt(delaunay_complex.filtration(simplex)));
        printf("%f\n", delaunay_complex.filtration(simplex));
    }

    // Voronoi edges connect adjacent d-simplices that share a (d-1)-face.
    vector<pair<int,int>> voronoi_edges;
    vector<double> v_edge_filtrations;
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
            v_edge_filtrations.push_back(-sqrt(delaunay_complex.filtration(simplex)));
        }
    }

    vector<int> canonical_v_points;
    vector<double> _p_filtrations;
    for (int i = 0; i < canonical_simplex.size(); ++i) {
        if (canonical_simplex[i]) {
            canonical_v_points.push_back(i);
            _p_filtrations.push_back(v_point_filtrations[i]);
        }
    }

    vector<pair<int,int>> periodic_v_edges;
    vector<double> _e_filtrations;
    for (int i = 0; i < voronoi_edges.size(); ++i) {
        auto [s,t] = voronoi_edges[i];
        if (canonical_simplex[s] || canonical_simplex[t]) {
            if (canonical_simplex[t] && !canonical_simplex[s]) {
                swap(s, t);
            }
            periodic_v_edges.push_back({s,t});
            _e_filtrations.push_back(v_edge_filtrations[i]);
        }
    }

    // Get the results
    int L = size(canonical_v_points);
    int M = size(periodic_v_edges);
    Eigen::MatrixXd v_points(d, L);
    Eigen::MatrixXi v_edges(M, 2);
    Eigen::VectorXd p_filtrations(L);
    Eigen::VectorXd e_filtrations(M);
    // Eigen::MatrixXi shift(d, M);

    for (int i = 0; i < L; ++i) {
        v_points.col(i) = voronoi_points[canonical_v_points[i]];
        p_filtrations(i) = _p_filtrations[i];
    }

    for (int i = 0; i < M; ++i) {
        // Voronoi edge between dual points of d-simplices.
        auto [s, t] = periodic_v_edges[i];
        v_edges(i, 0) = s;
        v_edges(i, 1) = t;
        e_filtrations(i) = _e_filtrations[i];
    }

    return {v_points, v_edges, p_filtrations, e_filtrations};
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
