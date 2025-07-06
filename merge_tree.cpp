#include "auxiliary.h"
#include "delaunay.h"
#include "merge_tree.h"

namespace PMT {
using namespace Eigen;

// Inverse of volume of input lattice, serve as multiplier for shadow monomial coefficient
double inputVolumeInv = 1; 

// The class for lattice operations
template <typename integer>
class Lattice {
	typedef Matrix<integer, Dynamic, Dynamic> Matrix;

private:
	Matrix V; // basis of the lattice

public:
	Lattice() {}
	Lattice(int dimension) { V = Matrix(dimension, 0); }
	Lattice(int dimension, int size) {
		V = Matrix(dimension, size); 
		V.setZero();
	}
	Lattice(const Matrix& V) : V(V) {}

	integer dimension() const { return V.rows(); }
	integer size() const { return V.cols(); }
	Matrix& basis() { return V; }

	integer operator()(int i, int j) const { return V(i, j); }
	integer& operator()(int i, int j) { return V(i, j); }

	// joint two lattices
	void operator+=(const Lattice& L) {
		if (this->dimension() != L.dimension()) {
			throw std::invalid_argument("lattices in different dimensions");
		}
		Matrix NewBasis(this->dimension(), this->size() + L.size());
		for (int i = 0; i < this->dimension(); ++i) {
			for (int j = 0; j < this->size(); ++j) {
				NewBasis(i, j) = V(i, j);
			}
			for (int j = 0; j < L.size(); ++j) {
				NewBasis(i, j + this->size()) = L(i, j);
			}
		}
		V = HermiteNormalForm(NewBasis);
	}

	// compute HNF of an integer matrix
	Matrix HermiteNormalForm(Matrix& M) {
		// column index, for implicit column exchange
		std::vector<int> colid(M.cols()); 
		for (int j = 0; j < M.cols(); ++j) colid[j] = j;
		
		// the reduction algorithm
		int rows = M.rows(), cols = M.cols();
		int i, j = 0, k, l;
		for (i = 0; i < rows; ++i) {
			// find the first l >= j such that M(i, l) != 0
			bool found = 0;
			for (l = j; l < cols; ++l) {
				if (M(i, colid[l]) != 0) {
					swap(colid[j], colid[l]);
					found = 1;
					break;
				}
			}
			if (!found) continue;
			// make sure M(i, j) > 0
			if (M(i, colid[j]) < 0) {
				for (l = i; l < rows; ++l) {
					M(l, colid[j]) = -M(l, colid[j]);
				}
			}
			for (k = j + 1; k < cols; ++k) {
				if (M(i, colid[k]) != 0) {
					// make sure M(i, k) > 0
					if (M(i, colid[k]) < 0) {
						for (l = i; l < rows; ++l) {
							M(l, colid[k]) = -M(l, colid[k]);
						}
					}
					// euclidean algorithm for gcd
					while (M(i, colid[j]) > 0 && M(i, colid[k]) > 0) {
						integer t = M(i, colid[j]) / M(i, colid[k]);
						if (t) { // substrace t * M(:, k) from M(:, j)
							for (l = i; l < rows; ++l) {
								M(l, colid[j]) -= M(l, colid[k]) * t;
							}
						}
						swap(colid[j], colid[k]);
					}
				}
			}
			// reduce columns precede to j
			for (k = 0; k < j; ++k) {
				integer t = M(i, colid[k]) / M(i, colid[j]);
				if (t) { // substrace t * M(:, j) from M(:, k)
					for (l = i; l < rows; ++l) {
						M(l, colid[k]) -= M(l, colid[j]) * t;
					}
				}
			}
			j += 1;
		}
		
		// collect the final result
		int finalsize = j;
		Matrix Reduced(rows, finalsize);
		for (i = 0; i < rows; ++i) {
			for (j = 0; j < finalsize; ++j) {
				Reduced(i, j) = M(i, colid[j]);
			}
		}
		return Reduced;
	}
};

// compute the volume of a lattice up to 3D
template <typename Matrix, typename Lattice>
double Volume(const Matrix& U, const Lattice& L) {
	if (L.size() == 0) { // vol_0
		return 1.0;
	}
	Matrix B(L.dimension(), L.size());
	for (int i = 0; i < L.dimension(); ++i) {
		for (int j = 0; j < L.size(); ++j) {
			B(i, j) = L(i, j) * 1.0;
		}
	}
	B = U * B;
	if (L.size() == 1) { // vol_1
		return B.norm();
	}
	if (L.size() == L.dimension()) { // full rank
		return abs(B.determinant());
	}
	// vol_2 of 3D lattice (cross product)
	double x, y, z;
	x = B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0);
	y = B(1, 0) * B(2, 1) - B(1, 1) * B(2, 0);
	z = B(2, 0) * B(0, 1) - B(2, 1) * B(0, 0);
	return sqrt(x * x + y * y + z * z);
}

const double pi = acos(-1.0);
std::vector<double> unitBallVolume = { 1, 2, 1, 4.0 / 3 };

class Event {
private:
	double _time; // time of the event
	int _child; // if it's a merger event, record the root of the child component. -1 for null.
	double _ratio; // vol_p / vol_d
	int _exponent; // exponent of the shadow monomial

public:
	Event() : _time(0), _child(-1), _ratio(1), _exponent(0) {}

	// notice that the second parameter is vol(periodic lattice) / vol(input lattice), instead of real coefficient
	Event(double time, int child, double ratio, int exponent)
		: _time(time), _child(child), _ratio(ratio), _exponent(exponent) {}

	double& time() { return _time; }
	int& child() { return _child; }
	double ratio() const { return _ratio; }
	int exponent() const { return _exponent; }

	bool operator==(const Event& other) const {
		if (_exponent != other.exponent()) return false;
		if (abs(_ratio - other.ratio()) < 1e-9) return true;
		return false;
	}

	bool operator>(const Event& other) const {
		if (_exponent > other.exponent()) return true;
		if (_ratio - other.ratio() >= 1e-9) return true;
		return false;
	}

	std::string toString() {
		char s[50];
		sprintf_s(s, "%.3f %sR^%d", _ratio * unitBallVolume[_exponent], (_exponent > 1 ? "pi" : ""), _exponent);
		return std::string(s);
	}
};

// the base class for 0- and 1-dimensional simplex (vertex & arc)
class Simplex {
private:
	int _p; // dimension of the simplex
	double _filtration; // filtration value

public:
	Simplex() : _p(0), _filtration(0) {}
	Simplex(int p, double f) : _p(p), _filtration(f) {}

	int order() { return _p; }
	double filtration() { return _filtration; }
};

template <typename integer>
class Arc : public Simplex {
	typedef Matrix<integer, Dynamic, 1> Vector;

private:
	int _source, _target; // index of source & target
	Vector _shift; // shift vector

public:
	Arc() : Simplex(1, 0), _source(0), _target(0) { _shift = Vector(0, 1); }
	// construct with dimensionality
	Arc(int s, int t, double f, int d)
		: Simplex(1, f), _source(s), _target(t) {
		_shift = Vector(d, 1);
		_shift.setZero();
	}
	// construct with nonzero shift vector
	Arc(int s, int t, double f, Vector v)
		: Simplex(1, f), _source(s), _target(t), _shift(v) {}

	int source() { return _source; }
	int target() { return _target; }
	Vector shift() const { return _shift; }
};


template <typename integer>
class Vertex : public Simplex {
	typedef Matrix<integer, Dynamic, 1> Vector;
	typedef Lattice<integer> Lattice;

private:
	int _root, _next; // indices of root, next. -1 for null.
	int _size; // size of subtree
	double _old; // birth time of the oldest child
	int _oldId; // index of the oldest child
	Vector _drift; // the drift vector
	Lattice _lattice; // the periodic lattice

public:
	Vertex() : Simplex(0, 0), _root(0), _next(-1), _size(1), _old(0.0), _oldId(0) {
		_drift = Vector(0, 1);
		_lattice = Lattice(_drift);
	}

	// construct with filtration value, dimensionality, and index
	Vertex(int id, double f, int d) : Simplex(0, f), _root(id), _next(-1), _size(1), _old(f), _oldId(id) {
		_drift = Vector(d, 1); _drift.setZero();
		_lattice = Lattice(d);
	}

	void driftAdd(const Vector& v) {
		_drift += v;
	}

	int& root() { return _root; }
	int& next() { return _next; }
	int& size() { return _size; }
	double& old() { return _old; }
	int& oldId() { return _oldId; }
	Vector& drift() { return _drift; }
	Lattice& lattice() { return _lattice; }
};

typedef int integer;

using namespace std;

struct filtrationComparator {
	bool operator()(Simplex& a, Simplex& b) const {
		return a.filtration() < b.filtration();
	}
};

template <typename Matrix, typename VertexList, typename ArcList, typename EventList>
void process(int d, Matrix& inputBasis, VertexList& vertices, ArcList& arcs, vector<EventList>& beams) {
	// record birth events for each vertex
	for (auto v : vertices) {
		beams[v.root()].push_back(Event(v.old(), -1, inputVolumeInv, d));
	}

	// sort the arcs in non-decreasing order
	std::sort(begin(arcs), end(arcs), filtrationComparator());

	myDebug("Sorted filtration:\n");
	for (auto a : arcs) {
		myDebug("%.3f ", a.filtration());
	}
	myDebug("\n");

	// union-find algorithm for PMT
	int x, y, r, s, z, last, p, rOldId, sOldId;
	double time = 0, vol_p;
	for (auto a : arcs) {
		time = a.filtration();
		x = a.source(), y = a.target();
		r = vertices[x].root(), s = vertices[y].root();
		rOldId = vertices[r].oldId(), sOldId = vertices[s].oldId();
		if (r == s) { // catenation
			if (beams[rOldId].back().ratio() == 1.0 && beams[rOldId].back().exponent() == 0) continue;
#ifdef debuging
			printf("\n** catenation %d -> %d\n", x, y);
#endif
			Lattice<integer> L(vertices[x].drift() + a.shift() - vertices[y].drift());
			vertices[r].lattice() += L;
			// compute shadow monomial, insert new event
			p = vertices[r].lattice().size();
			vol_p = Volume(inputBasis, vertices[r].lattice());
			beams[rOldId].push_back(Event(time, -1, vol_p * inputVolumeInv, d - p));
#ifdef debuging
			cout << "time: " << time << endl;
			cout << "v:\n" << L.basis() << endl;
			cout << "new lattice:\n" << vertices[r].lattice().basis() << endl;
			cout << "vol_p: " << vol_p << endl;
			cout << "monomial: " << beams[rOldId].back().toString() << endl;
#endif
		}
		else { // merger
#ifdef debuging
			printf("\n** merger %d -> %d\n", x, y);
#endif
			// record index of oldest vertex in both components
			if (vertices[r].old() > vertices[s].old()) {
				swap(rOldId, sOldId);
			}
			// make sure size(s) <= size(r), and merge s -> r
			if (vertices[r].size() < vertices[s].size()) {
				swap(r, s);
				swap(x, y);
			}
			if (vertices[r].old() > vertices[s].old()) {
				vertices[r].old() = vertices[s].old();
				// vertices[r].oldId() = vertices[s].oldId();
			}
			vertices[r].oldId() = rOldId;
			vertices[r].lattice() += vertices[s].lattice();
			Eigen::Matrix<integer, Dynamic, 1> v(vertices[x].drift() + a.shift() - vertices[y].drift());
			z = s;
			while (z != -1) {
				vertices[z].root() = r;
				vertices[z].drift() += v;
				last = z;
				z = vertices[z].next();
			}
			vertices[r].size() += vertices[s].size();
			vertices[last].next() = vertices[r].next();
			vertices[r].next() = s;
			// update shadow monomial, insert new event
			p = vertices[r].lattice().size();
			vol_p = Volume(inputBasis, vertices[r].lattice());
			beams[rOldId].push_back(Event(time, sOldId, vol_p * inputVolumeInv, d - p));
#ifdef debuging
			cout << "time: " << time << endl;
			cout << "v:\n" << v << endl;
			cout << "new lattice:\n" << vertices[r].lattice().basis() << endl;
			cout << "vol_p: " << vol_p << endl;
			cout << "merge " << vertices[s].oldId() << " -> " << vertices[r].oldId() << endl;
			cout << "monomial: " << beams[rOldId].back().toString() << endl;
#endif
		}
	}
}

// append last event to the beams that merged to other beams
template <typename EventList>
void addRightEnd(vector<EventList>& beams) {
	for (int i = 0; i < beams.size(); ++i) {
		for (auto event : beams[i]) {
			if (event.child() != -1 && event.child() != i) {
				beams[event.child()].push_back(event);
			}
		}
	}
}

// print periodic merge tree
template <typename EventList>
void printBeams(vector<EventList>& beams) {
	for (int i = 0; i < beams.size(); ++i) {
		printf("%d:\n", i);
		for (auto event : beams[i]) {
			printf(" time: %.3f, monomial: %s", event.time(), event.toString().c_str());
			if (event.child() != -1) {
				if (event.child() == i) printf(", dies");
				else printf(", child: %d", event.child());
			}
			printf("\n");
		}
	}
}

class Barcode {
private:
	double _birth, _death, _multiplicity;
public:
	Barcode() : _birth(std::numeric_limits<double>::lowest()), _death(std::numeric_limits<double>::max()), _multiplicity(0) {}
	Barcode(double birth, double death, double multiplicity)
		: _birth(birth), _death(death), _multiplicity(multiplicity) {}

	double birth() { return _birth; }
	double death() { return _death; }
	double multiplicity() { return _multiplicity; }

	std::string toString() {
		char s[100];
		if (_death == std::numeric_limits<double>::max()) {
			sprintf_s(s, "[%.3f, +inf) %.3f", _birth, _multiplicity);
		}
		else {
			sprintf_s(s, "[%.3f, %.3f] %.3f", _birth, _death, _multiplicity);
		}
		return std::string(s);
	}
};

// for sorting barcodes
struct barcodeComparator {
	bool operator()(Barcode& a, Barcode& b) {
		if (a.birth() < b.birth()) return true;
		else if (a.birth() == b.birth()) return a.death() < b.death();
		else return false;
	}
};

// print barcode in different eras
template <typename EventList>
void constructBarcodes(vector<EventList>& beams, vector<vector<Barcode>>& barcodes) {
	int i, j, k;
	double birth, death;
	constexpr double inf = std::numeric_limits<double>::max();
	for (i = 0; i < beams.size(); ++i) {
		EventList& epoch = beams[i];
		myDebug("%d: %d epoches\n", i, epoch.size());
		birth = epoch[0].time();
		for (j = 0; j < epoch.size(); ) {
			myDebug(" %d time: %.3f monomial: %s child: %d, ", j, epoch[j].time(), epoch[j].toString().c_str(), epoch[j].child());
			for (k = j + 1; k < epoch.size(); ++k) { // find the shadow monomial beam[k] > beam[j]
				if (epoch[j] > epoch[k]) {
					break;
				}
			}
			if (k == epoch.size()) {
				if (epoch[k - 1].child() != i) { 
					// never die
					myDebug("case 1 | ");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, inf, epoch[j].ratio()));
					myDebug("%dd %s", epoch[j].exponent(), barcodes[epoch[j].exponent()].back().toString().c_str());
				}
				else { 
					// merger with the same monomial
					myDebug("case 2 | ");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, epoch.back().time(), epoch[j].ratio()));
					myDebug("%dd %s", epoch[j].exponent(), barcodes[epoch[j].exponent()].back().toString().c_str());
				}
			}
			else {
				if (epoch[j].exponent() != epoch[k].exponent()) { 
					// different exponents, split into different dimensions
					myDebug("case 3 | ");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, epoch[k].time(), epoch[j].ratio()));
					myDebug("%dd %s, ", epoch[j].exponent(), barcodes[epoch[j].exponent()].back().toString().c_str());
					if (k == epoch.size() - 1 && epoch[k].child() == i) break;
					barcodes[epoch[k].exponent()].push_back(Barcode(birth, epoch[k].time(), -epoch[k].ratio()));
					myDebug("%dd %s", epoch[k].exponent(), barcodes[epoch[k].exponent()].back().toString().c_str());
				}
				else { 
					// same exponent, (epoch[j].ratio() - epoch[k].ratio()) components die
					myDebug("case 4 | ");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, epoch[k].time(), epoch[j].ratio() - epoch[k].ratio()));
					myDebug("%dd %s", epoch[j].exponent(), barcodes[epoch[j].exponent()].back().toString().c_str());
				}
			}
			j = k;
			myDebug("\n");
		}
	}
}

void printBarcodes(vector<vector<Barcode>>& barcodes) { 
	int i, j;
	for (i = barcodes.size() - 1; i >= 0; --i) {
		printf("%d-th:\n", i);
		sort(begin(barcodes[i]), end(barcodes[i]), barcodeComparator());
		for (j = 0; j < barcodes[i].size(); ++j) {
			printf(" %s\n", barcodes[i][j].toString().c_str());
		}
	}
}

// Compute periodic merge tree from a quotient complex
// Input:
//	Number of vertices: int
// 	Dimension: int
// 	Lattice basis: MatrixXd(d, d)
// 	Arcs: MatrixXi(M, 2)
//	Arc filtration value: VectorXd(M)
//	Arc shift vector: MatrixXd(d, M)
// 	(Optional) Vertex filtration value: VectorXd(N), 0 by default
// Output:
// 	Periodic merge tree: vector<vector<Event>>, where Event is a tuple containing
//	 Time: double -- the time that the event happends
//	 Coefficient: double -- coefficient of the shadow monomial
//	 Exponent: int -- exponent of the shadow monomial
// 	 Child: int -- root of the branch merged to current branch, -1 if it's not a merger
std::vector<std::vector<std::tuple<double, double, int, int>>> mergeTree (
	int n, 						// number of vertices
	int d, 						// dimension
	const Eigen::MatrixXd& V, 	// lattice basis
	const Eigen::MatrixXi& arcs, 
	const Eigen::VectorXd& arc_filtration,
	const Eigen::MatrixXi& arc_shift,
	Eigen::VectorXd& vertex_filtration
) {
	if (d != V.rows() || V.cols() < V.rows()) {
        throw std::invalid_argument("Invalid lattice basis");
	}
	if (d != arc_shift.rows()) {
        throw std::invalid_argument("Shift vectors does not match dimension");
	}
	if (arcs.rows() != arc_filtration.size()) {
		throw std::invalid_argument("Number of arcs does not match filtration values");
	}
	if (vertex_filtration.size() == 0) {
		vertex_filtration = Eigen::VectorXd::Zero(n);
	}
	else if (n != vertex_filtration.size()) {
		throw std::invalid_argument("Number of vertices does not match filtration values");
	}

	auto validFiltration = [&arcs, &arc_filtration, &vertex_filtration]() -> bool {
		for (int i = 0; i < arcs.rows(); ++i) {
			int s = arcs(i, 0), t = arcs(i, 1);
			if (arc_filtration(i) < vertex_filtration(s) || arc_filtration(i) < vertex_filtration(t)) {
				return false;
			}
		}
		return true;
	};

	if (!validFiltration()) {
		throw std::invalid_argument("Input is not a valid filtration");
	}

	using Vector = Matrix<integer, Dynamic, 1>;
	using Matrix = Matrix<double, Dynamic, Dynamic>;
	using Vertex = Vertex<integer>;
	using Arc = Arc<integer>;

	// Copy the lattice basis
	Matrix U(d, d);
	for (int i = 0; i < d; ++i) {
		for (int j = 0; j < d; ++j) {
			U(i, j) = V(i, j);
		}
	}

	// Compute volume of unit cell
	inputVolumeInv = 1.0 / abs(U.determinant());

	// Create vertices
	vector<Vertex> _vertices;
	for (int i = 0; i < n; ++i) {
		_vertices.push_back(Vertex(i, vertex_filtration(i), d));
	}
	
	// Create arcs
	int m = arcs.rows();
	vector<Arc> _arcs;
	for (int i = 0; i < m; ++i) {
		_arcs.push_back(Arc(arcs(i, 0), arcs(i, 1), arc_filtration(i), arc_shift.col(i)));
	}

	// Compute periodic merge tree
	vector<vector<Event>> beams(n);
	process(d, U, _vertices, _arcs, beams);
	addRightEnd(beams);

	// Get results
	std::vector<std::vector<std::tuple<double, double, int, int>>> result(n);
	for (int i = 0; i < n; ++i) {
		for (auto event : beams[i]) {
			result[i].push_back({event.time(), event.ratio(), event.exponent(), event.child()});
		}
	}

	return result;
}


// Print periodic merge tree in a nice format
// Input:
// 	Periodic merge tree as lists of events
void printMergeTree(
	const std::vector<std::vector<std::tuple<double, double, int, int>>>& tree
) {
	printf("** Periodic merge tree **\n");
	for (int i = 0; i < tree.size(); ++i) {
		printf("%d:\n", i);
		for (auto [time, coeff, exponent, child] : tree[i]) {
			printf(" time: %.3f, monomial: %s", time, Event(0, 0, coeff, exponent).toString().c_str());
			if (child != -1) {
				if (child == i) printf(" - merge ->");
				else printf(" <- child: %d", child);
			}
			printf("\n");
		}
	}
}

// Compute periodic barcode from merge tree
// Input:
// 	Dimension: d
//	Periodic merge tree as lists of events
// Output:
//	Lists of barcodes: vector<vector<Barcode>> of size d + 1, where Barcode is a tuple containing
//	 Birth time: double
//	 Death time: double
//	 Multiplicity: double
std::vector<std::vector<std::tuple<double, double, double>>> barcode(
	int d,
	const std::vector<std::vector<std::tuple<double, double, int, int>>>& tree
) {
	int n = size(tree);
	
	// From tuple to Event instance
	vector<vector<Event>> beams(n);
	for (int i = 0; i < n; ++i) {
		for (auto [time, coeff, exponent, child] : tree[i]) {
			beams[i].push_back(Event(time, child, coeff, exponent));
		}
	}

	// Construct barcodes
	vector<vector<Barcode>> barcodes(d + 1);
	constructBarcodes(beams, barcodes);

	// Sort barcodes
	for (int i = d; i >= 0; --i) {
		sort(begin(barcodes[i]), end(barcodes[i]), barcodeComparator());
	}

	// Convert result to tuple
	std::vector<std::vector<std::tuple<double, double, double>>> result(d + 1);
	for (int i = 0; i < d + 1; ++i) {
		for (auto b : barcodes[i]) {
			result[i].push_back({b.birth(), b.death(), b.multiplicity()});
		}
	}

	return result;
}

} // End of namespace PMT
