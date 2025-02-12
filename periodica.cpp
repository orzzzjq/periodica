#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <Eigen/Dense>

//#define debuging
#define recordtime
//#define simpledfs

#ifdef recordtime
auto start = std::chrono::high_resolution_clock::now();
auto stop = std::chrono::high_resolution_clock::now();
#define recordStart() start = std::chrono::high_resolution_clock::now();
#define recordStop(x) stop = std::chrono::high_resolution_clock::now(); fprintf(stderr, "%s %dms ", (x), std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
#else
#define recordStart() ;
#define recordStop(x) ;
#endif

#ifdef debuging
#define debug(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__);
#else
#define debug(fmt, ...) ;
#endif

using namespace Eigen;

// inverse of volume of input lattice, serve as multiplier for shadow monomial coefficient
double inputVolumeInv = 1; 

// function for exchanging two indices
inline void swap(int& x, int& y) { int z = x; x = y, y = z; };

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
		sprintf_s(s, "%.3f %sR^%d", _ratio * unitBallVolume[_exponent], (_exponent > 1 ? "¦Ð" : ""), _exponent);
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
	Vertex(int id, double f, int d) : Simplex(0, f), _root(id), _next(-1), _size(1), _old(f), _oldId(id) {
		_drift = Vector(d, 1); _drift.setZero();
		_lattice = Lattice(d);
		_events.clear();
		_events.push_back(Event(f, -1, inputVolumeInv, d)); // record birth of vertex
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
	std::vector<Event>& events() { return _events; }
};

typedef int integer;

using namespace std;

struct filtrationComparator {
	bool operator()(Simplex& a, Simplex& b) const {
		return a.filtration() < b.filtration();
	}
};

template <typename Matrix, typename VertexList, typename ArcList>
void process(int d, Matrix& inputBasis, VertexList& vertices, ArcList& arcs) {
	std::sort(begin(arcs), end(arcs), filtrationComparator());

	int x, y, r, s, z, last, p;
	double time = 0, vol_p;
	for (auto a : arcs) {
		time = a.filtration();
		x = a.source(), y = a.target();
		r = vertices[x].root(), s = vertices[y].root();
		if (r == s) { // catenation
#ifdef debuging
			printf("\n** catenation %d -> %d\n", x, y);
#endif
			Lattice<integer> L(vertices[x].drift() + a.shift() - vertices[y].drift());
			vertices[r].lattice() += L;
			// compute shadow monomial, insert new event
			p = vertices[r].lattice().size();
			vol_p = Volume(inputBasis, vertices[r].lattice());
			vertices[r].events().push_back(Event(time, -1, vol_p * inputVolumeInv, d - p));
#ifdef debuging
			cout << "time: " << time << endl;
			cout << "v:\n" << L.basis() << endl;
			cout << "new lattice:\n" << vertices[r].lattice().basis() << endl;
			cout << "vol_p: " << vol_p << endl;
			cout << "monomial: " << vertices[r].events().back().toString() << endl;
#endif
		}
		else { // merger
#ifdef debuging
			printf("\n** merger %d -> %d\n", x, y);
#endif
			// make sure size(s) <= size(r)
			if (vertices[r].size() < vertices[s].size()) {
				swap(r, s);
				swap(x, y);
			}
			if (vertices[r].old() > vertices[s].old()) {
				vertices[r].old() = vertices[s].old();
				vertices[r].oldId() = vertices[s].oldId();
			}
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
			vertices[r].events().push_back(Event(time, s, vol_p * inputVolumeInv, d - p));
#ifdef debuging
			cout << "time: " << time << endl;
			cout << "v:\n" << v << endl;
			cout << "new lattice:\n" << vertices[r].lattice().basis() << endl;
			cout << "vol_p: " << vol_p << endl;
			cout << "monomial: " << vertices[r].events().back().toString() << endl;
#endif
		}
	}
}

template <typename VertexList, typename EventList>
void dfs(int root, VertexList& vertices, EventList& succEvents, vector<EventList>& beams) {
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
		dfs(older[i].second, vertices, succEvents, beams);
	}

	// construct beam
	for (i = succEvents.size() - 1; i >= 0; --i) {
		if (i > 0 && succEvents[i].time() == succEvents[i - 1].time()) continue;
		beams[root].push_back(succEvents[i]);
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
				dfs(events[i].child(), vertices, succEvents, beams);
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

// append last event to the beams that merged to other beams
template <typename EventList>
void addRightEnd(vector<EventList>& beams) {
	for (int i = 1; i < beams.size(); ++i) {
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
	for (int i = 1; i < beams.size(); ++i) {
		printf("%d:\n", i);
		for (auto event : beams[i]) {
			if (event.child() == i) continue;
			printf(" time: %.3f, monomial: %s", event.time(), event.toString().c_str());
			if (event.child() != -1) {
				printf(", child: %d", event.child());
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
void printBarcodes(int d, vector<EventList>& beams) {
	vector<vector<Barcode>> barcodes(d + 1);
	int i, j, k;
	double birth, death;
	const double inf = double(std::numeric_limits<double>::max());
	for (i = 1; i < beams.size(); ++i) {
		EventList& epoch = beams[i];
		debug("%d: %d epoches\n", i, epoch.size());
		birth = epoch[0].time();
		for (j = 0; j < epoch.size(); ) {
			debug(" %d time: %.3f monomial: %s child: %d, ", j, epoch[j].time(), epoch[j].toString().c_str(), epoch[j].child());
			for (k = j + 1; k < epoch.size(); ++k) { // find the shadow monomial beam[k] > beam[j]
				if (epoch[j] > epoch[k]) {
					break;
				}
			}
			if (k == epoch.size()) {
				if (j == epoch.size() - 1 && epoch[j].child() != i) { // never die
					debug("case 1");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, inf, epoch[j].ratio()));
				}
				else { // merger with the same monomial
					debug("case 2");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, epoch.back().time(), epoch[j].ratio()));
				}
			}
			else {
				if (epoch[j].exponent() != epoch[k].exponent()) { // different exponents, decompose into two
					debug("case 3");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, epoch[k].time(), epoch[j].ratio()));
					barcodes[epoch[k].exponent()].push_back(Barcode(birth, epoch[k].time(), -epoch[k].ratio()));
				}
				else { // same exponent, (epoch[j].ratio() - epoch[k].ratio()) components dies
					debug("case 4");
					barcodes[epoch[j].exponent()].push_back(Barcode(birth, epoch[k].time(), epoch[j].ratio() - epoch[k].ratio()));
				}
			}
			j = k;
			debug("\n");
		}
	}
	for (i = d; i >= 0; --i) {
		printf("%d-th:\n", i);
		sort(begin(barcodes[i]), end(barcodes[i]), barcodeComparator());
		for (j = 0; j < barcodes[i].size(); ++j) {
			if (j < barcodes[i].size() - 1 
				&& barcodes[i][j].birth() == barcodes[i][j+1].birth() 
				&& barcodes[i][j].death() == barcodes[i][j+1].death() 
				&& barcodes[i][j].multiplicity() == -barcodes[i][j + 1].multiplicity()) {
				j++;
				continue;
			}
			printf(" %s\n", barcodes[i][j].toString().c_str());
		}
	}
}

// run example from file
void runExample(const char *filename) {
	int d, n, m;
	typedef Matrix<integer, Dynamic, 1> Vector;
	typedef Matrix<double, Dynamic, Dynamic> Matrix;
	typedef Vertex<integer> Vertex;
	typedef Arc<integer> Arc;

	FILE* fp;
	fopen_s(&fp, filename, "r");
	if (!fp) {
		throw std::invalid_argument("cannot open file");
		return;
	}

	const int buffer = 1000;
	char s[buffer];

	// dimension
	fgets(s, buffer, fp);
	fgets(s, buffer, fp);
	sscanf_s(s, "%d", &d);
	printf("dimension: %d\n", d);

	// lattice basis
	fgets(s, buffer, fp);
	Matrix U(d, d);
	for (int i = 0; i < d; ++i) {
		for (int j = 0; j < d; ++j) {
			double x;
			fscanf_s(fp, "%lf ", &U(i,j));
		}
	}
	inputVolumeInv = 1.0 / abs(U.determinant());
	cout << "lattice:\n" << U << endl;
	printf("volume: %.3f\n", 1.0 / inputVolumeInv);

	// vertices
	fgets(s, buffer, fp);
	printf("vertices:\n");
	vector<Vertex> vertices(1);
	int id;
	double f;
	fscanf_s(fp, "%d", &n);
	for (int i = 0; i < n; ++i) {
		fscanf_s(fp, "%d %lf ", &id, &f);
		vertices.push_back(Vertex(id, f, d));
		printf("%d: %.3f\n", vertices.back().root(), vertices.back().filtration());
	}

	// arcs
	fgets(s, buffer, fp);
	printf("arcs:\n");
	vector<Arc> arcs;
	int source, target;
	Vector shift(d, 1);
	fscanf_s(fp, "%d", &m);
	for (int i = 0; i < m; ++i) {
		fscanf_s(fp, "%d %d %lf ", &source, &target, &f);
		for (int j = 0; j < d; ++j) {
			fscanf_s(fp, "%d ", &shift(j, 0));
		}
		arcs.push_back(Arc(source, target, f, shift));
		printf("%d->%d: %.3f (", arcs.back().source(), arcs.back().target(), arcs.back().filtration());
		for (int j = 0; j < d; ++j) {
			printf("%d%c", arcs.back().shift()(j, 0), ",)"[j==d-1]);
		}
		printf("\n");
	}

	// run algorithm
	process(d, U, vertices, arcs);

	int root = vertices[1].root();
	printf("\nperiodic merge tree:\n");
#ifdef simpledfs
	simpleDfs(vertices, root);
#else
	vector<Event> succEvents(0);
	vector<vector<Event>> beams(n + 1);
	dfs(root, vertices, succEvents, beams);
	printBeams(beams);

	printf("\nperiodic barcode:\n");
	addRightEnd(beams);
	printBarcodes(d, beams);
#endif
}

int main()
{
	//runExample("C:/_/Project/periodica/examples/example_2d_1.txt");
	//runExample("C:/_/Project/periodica/examples/example_2d_2.txt");
	recordStart();
	runExample("C:/_/Project/periodica/examples/example_3d_1.txt");
	recordStop("\nrunning time:");
}
