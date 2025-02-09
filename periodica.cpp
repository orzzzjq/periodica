#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

//#define debuging
//#define simpledfs

using namespace Eigen;

// inverse of volume of input lattice, serve as multiplier for shadow monomial coefficient
double inputVolumeInv = 1; 

// function for exchanging two indices
inline void exchange(int& x, int& y) { int z = x; x = y, y = z; };

// minimum of two double variables
inline double min(double x, double y) { return x < y ? x : y; };

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
					exchange(colid[j], colid[l]);
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
						exchange(colid[j], colid[k]);
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
	double _coefficient; // coefficient of the shadow monomial
	int _exponent; // exponent of the shadow monomial

public:
	Event() : _time(0), _child(-1), _coefficient(0), _exponent(0) {}

	// notice that the second parameter is vol(periodic lattice) / vol(input lattice), instead of real coefficient
	Event(double time, int child, double ratio, int exponent)
		: _time(time), _child(child), _coefficient(ratio * unitBallVolume[exponent]), _exponent(exponent) {}

	double time() { return _time; }
	int& child() { return _child; }
	double frequency() const { return _coefficient; }
	int dimension() const { return _exponent; }

	void setValue(double time, int child, double ratio, int exponenet) {
		_time = time;
		_child = child;
		_coefficient = ratio * unitBallVolume[exponenet];
		_exponent = exponenet;
	}

	std::string toString() {
		char s[50];
		sprintf_s(s, "%.3f %sR^%d", _coefficient, (_exponent > 1 ? "pi " : ""), _exponent);
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
	Arc(double f, int s, int t, int d)
		: Simplex(1, f), _source(s), _target(t) {
		_shift = Vector(d, 1);
		_shift.setZero();
	}
	// construct with nonzero shift vector
	Arc(double f, int s, int t, Vector v)
		: Simplex(1, f), _source(s), _target(t), _shift(v) {}

	int source() { return _source; }
	int target() { return _target; }
	Vector& shift() { return _shift; }
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

	// record the changes of shadow monomial (including the corresponding 
	// time and also the index to child component if it's a merger)
	std::vector<Event> _events; 

public:
	Vertex() : Simplex(0, 0), _root(0), _next(-1), _size(1), _old(0.0), _oldId(0) {
		_drift = Vector(0, 1);
		_lattice = Lattice(_drift);
		_events.clear();
	}

	// construct with filtration value, dimensionality, and index
	Vertex(double f, int d, int id) : Simplex(0, f), _root(id), _next(-1), _size(1), _old(f), _oldId(id) {
		_drift = Vector(d, 1); _drift.setZero();
		_lattice = Lattice(d);
		_events.clear();
		_events.push_back(Event(f, -1, inputVolumeInv, d)); // record birth of vertex
	}

	int& root() { return _root; }
	int& next() { return _next; }
	int& size() { return _size; }
	double& old() { return _old; }
	int& oldId() { return _oldId; }
	Vector& drift() { return _drift; }
	Lattice& lattice() { return _lattice; }
	std::vector<Event>& events() { return _events; }
};

struct comparator {
	bool operator()(Simplex& a, Simplex& b) const {
		return a.filtration() < b.filtration();
	}
};

typedef int integer;

using namespace std;

template <typename Matrix, typename VertexList, typename ArcList>
void process(int d, Matrix& inputBasis, VertexList& vertices, ArcList& arcs) {
	std::sort(begin(arcs), end(arcs), comparator());
	int x, y, r, s, z, last, p;
	double time = 0, vol_p;
	for (auto a : arcs) {
		time = a.filtration();
		x = a.source(), y = a.target();
		r = vertices[x].root(), s = vertices[y].root();
		if (r == s) { // catenation
			Lattice<integer> L(vertices[x].drift() + a.shift() - vertices[y].drift());
			vertices[r].lattice() += L;
			// compute shadow monomial, insert new event
			p = vertices[r].lattice().size();
			vol_p = Volume(inputBasis, vertices[r].lattice());
			vertices[r].events().push_back(Event(time, -1, vol_p * inputVolumeInv, d - p));
#ifdef debuging
			printf("\n** catenation %d -> %d\n", x, y);
			cout << "v:\n" << L.basis() << endl;
			cout << "new lattice:\n" << vertices[r].lattice().basis() << endl;
			cout << "vol_p: " << vol_p << endl;
			cout << "monomial: " << vertices[r].events().back().toString() << endl;
#endif
		}
		else { // merger
			// make sure size(s) <= size(r)
			if (vertices[r].size() < vertices[s].size()) {
				exchange(r, s);
				exchange(x, y);
			}
			if (vertices[r].old() > vertices[s].old()) {
				vertices[r].old() = vertices[s].old();
				vertices[r].oldId() = vertices[s].oldId();
			}
			vertices[r].lattice() += vertices[s].lattice();
			auto v = vertices[x].drift() + a.shift() - vertices[y].drift();
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
			vertices[r].events().push_back(Event(time, s, vol_p * inputVolumeInv, d - p));
#ifdef debuging
			printf("\n** merger %d -> %d\n", x, y);
			cout << "v:\n" << v << endl;
			cout << "new lattice:\n" << vertices[r].lattice().basis() << endl;
			cout << "vol_p: " << vol_p << endl;
			cout << "monomial: " << vertices[r].events().back().toString() << endl;
#endif
		}
	}
}

template <typename VertexList, typename EventList>
void dfs(VertexList& vertices, EventList& succEvents, int root) {
	int i, j;
	auto events = vertices[root].events();

	// find a index list with monotonically increasing birth time
	vector<pair<int, int>> older; // {index in events list, index in vertices list}
	for (i = events.size() - 1; i >= 0; --i) {
		if (events[i].child() != -1) {
			int c = events[i].child();
			// monotonic stack
			while (older.size() && (vertices[c].old() < vertices[older.back().second].old())) {
				older.pop_back();
			}
			older.push_back({ i, c });
		}
	}
	while (older.size() && (vertices[root].filtration() < vertices[older.back().second].old())) {
		older.pop_back();
	}
	older.push_back({ 0, root });

	// transplant the successor events to older child
	j = events.size() - 1;
	for (i = 0; i < older.size(); ++i) {
		for (; j >= older[i].first; --j) {
			succEvents.push_back(events[j]);
		}
		if (older[i].second == root) continue;
		else {
			if (i < older.size() - 2) {
				succEvents.back().child() = vertices[older[i + 1].second].oldId();
			}
			else {
				succEvents.back().child() = root;
			}
		}
		dfs(vertices, succEvents, older[i].second);
	}

	// print events
	printf("%d:\n", root);
	for (i = succEvents.size() - 1; i >= 0; --i) {
		if (i > 0 && succEvents[i].time() == succEvents[i - 1].time()) continue;
		auto event = succEvents[i];
		printf(" time: %.3f, monomial: %s", event.time(), event.toString().c_str());
		if (event.child() != -1) {
			printf(", child: %d", event.child());
		}
		printf("\n");
	}
	succEvents.clear();

	// handle unvisited children
	j = 0;
	for (i = events.size() - 1; i >= 0; --i) {
		if (events[i].child() != -1) {
			if (events[i].child() == older[j].second) {
				++j;
			}
			else {
				dfs(vertices, succEvents, events[i].child());
			}
		}
	}
}

template <typename VertexList>
void simpleDfs(VertexList& vertices, int root) {
	printf("%d:\n", root);
	auto events = vertices[root].events();
	for (int i = 0; i < events.size(); ++i) {
		if (i < events.size() - 1 && events[i].time() == events[i + 1].time()) continue;
		auto event = events[i];
		printf(" time: %.3f, monomial: %s", event.time(), event.toString().c_str());
		if (event.child() != -1) {
			printf(", child: %d", event.child());
		}
		printf("\n");
	}
#ifdef debuging
	// find a index list with monotonically increasing birth time
	vector<pair<int, int>> older; // {index in events list, index in vertices list}
	for (int i = events.size() - 1; i >= 0; --i) {
		if (events[i].child() != -1) {
			int c = events[i].child();
			// monotonic stack
			while (older.size() && (vertices[c].old() < vertices[older.back().second].old())) {
				older.pop_back();
			}
			older.push_back({ i, c });
		}
	}
	while (older.size() && (vertices[root].filtration() < vertices[older.back().second].old())) {
		older.pop_back();
	}
	older.push_back({ 0, root });
	printf("*older: ");
	for (auto x : older) printf("%d ", x.second);
	printf("\n");
#endif
	for (auto event : events) {
		if (event.child() != -1) {
			simpleDfs(vertices, event.child());
		}
	}
}

void run2DExample_1() {
	int d = 2;
	typedef Matrix<integer, Dynamic, 1> Vector;
	typedef Matrix<double, Dynamic, Dynamic> Matrix;
	typedef Vertex<integer> Vertex;
	typedef Arc<integer> Arc;

	Matrix U(d, d);
	U << 1, 0, 
		 0, 1;
	inputVolumeInv = 1.0 / abs(U.determinant());

	vector<Vertex> vertices;
	vertices.push_back(Vertex(1, d, 0));
	vertices.push_back(Vertex(3, d, 1));

	vector<Arc> arcs;
	arcs.push_back(Arc(5, 0, 1, d));
	Vector shift(d, 1);
	shift << 1, 0;
	arcs.push_back(Arc(9, 1, 0, shift));
	shift << 1, 1;
	arcs.push_back(Arc(7, 1, 0, shift));

	process(d, U, vertices, arcs);

	int root = vertices[0].root();
	printf("\nperiodic merge tree:\n");
#ifdef simpledfs
	simpleDfs(vertices, root);
#else
	vector<Event> succEvents(0);
	dfs(vertices, succEvents, root);
#endif
}

void run2DExample_2() {
	int d = 2;
	typedef Matrix<integer, Dynamic, 1> Vector;
	typedef Matrix<double, Dynamic, Dynamic> Matrix;
	typedef Vertex<integer> Vertex;
	typedef Arc<integer> Arc;

	Matrix U(d, d);
	U << 1, 0, 
		 0, 2;
	inputVolumeInv = 1.0 / abs(U.determinant());

	vector<Vertex> vertices;
	vertices.push_back(Vertex(1, d, 0));
	vertices.push_back(Vertex(3, d, 1));
	vertices.push_back(Vertex(1, d, 2));
	vertices.push_back(Vertex(3, d, 3));

	vector<Arc> arcs;
	arcs.push_back(Arc(5, 0, 1, d));
	arcs.push_back(Arc(9, 1, 2, d));
	arcs.push_back(Arc(5, 2, 3, d));
	Vector shift(d, 1);
	shift << 0, 1;
	arcs.push_back(Arc(7, 1, 2, shift));
	shift << 1, 1;
	arcs.push_back(Arc(7, 3, 0, shift));
	shift << 1, 0;
	arcs.push_back(Arc(9, 3, 0, shift));

	process(d, U, vertices, arcs);

	int root = vertices[0].root();
	printf("\nperiodic merge tree:\n");
#ifdef simpledfs
	simpleDfs(vertices, root);
#else
	vector<Event> succEvents(0);
	dfs(vertices, succEvents, root);
#endif
}

void run3DExample_1() {
	int d = 3;
	typedef Matrix<integer, Dynamic, 1> Vector;
	typedef Matrix<double, Dynamic, Dynamic> Matrix;
	typedef Vertex<integer> Vertex;
	typedef Arc<integer> Arc;

	Matrix U(d, d);
	U << 1, 0, 0,
		 0, 1, 0,
		 0, 0, 1;
	inputVolumeInv = 1.0 / abs(U.determinant());

	vector<Vertex> vertices;
	vertices.push_back(Vertex());
	vertices.push_back(Vertex(1, d, 1));
	vertices.push_back(Vertex(2, d, 2));
	vertices.push_back(Vertex(3, d, 3));
	vertices.push_back(Vertex(4, d, 4));
	vertices.push_back(Vertex(5, d, 5));

	vector<Arc> arcs;
	Vector shift(d, 1);
	arcs.push_back(Arc(13, 1, 2, d));
	shift << 1, 0, 0;
	arcs.push_back(Arc(12, 1, 2, shift));
	shift << 1, 1, 0;
	arcs.push_back(Arc(6, 2, 2, shift));
	arcs.push_back(Arc(10, 2, 4, d));
	arcs.push_back(Arc(7, 4, 3, d));
	shift << 1, 0, 0;
	arcs.push_back(Arc(8, 4, 5, shift));
	shift << 1, 0, 0;
	arcs.push_back(Arc(9, 5, 3, shift));
	shift << 0, 0, 1;
	arcs.push_back(Arc(11, 3, 3, shift));

	process(d, U, vertices, arcs);

	int root = vertices[1].root();
	printf("\nperiodic merge tree:\n");
#ifdef simpledfs
	simpleDfs(vertices, root);
#else
	vector<Event> succEvents(0);
	dfs(vertices, succEvents, root);
#endif
}

int main()
{
	//run2DExample_1();
	//run2DExample_2();
	run3DExample_1();
}
