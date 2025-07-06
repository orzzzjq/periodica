#include "auxiliary.h"
#include "delaunay.h"

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

// Compute the Euclidean MST of a point set
Eigen::MatrixXi EuclideanMST(
    const Eigen::MatrixXd& points
) {
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
    const Eigen::MatrixXd& points   // points in unit cell
) {
    if (U.cols() != U.rows() || U.rows() != points.rows()) {
        throw std::invalid_argument("Invalid input");
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

    // Delaunay complex from point in the 3x domain
    auto delaunay_edges = DelaunaySkeleton(working_points);

    // Filter the periodic edges (have at least one end point in the 1x domain
    vector<pair<int,int>> quotient_edges;
    for (int i = 0; i < delaunay_edges.rows(); ++i) {
        int s = delaunay_edges(i, 0), t = delaunay_edges(i, 1);
        if (s < n || t < n) {
            if (s > t) swap(s, t); // let the first point be the one with smaller index
            if (t >= n && s > I(t)) continue;
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
} // End of namespace DELAUNAY
