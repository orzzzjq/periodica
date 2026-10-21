#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include <Eigen/Dense>

namespace PMT {

// One event of a beam (the history of one component, identified by its
// oldest vertex): the shadow monomial ratio * R^exponent from `time` on.
// child == -1: plain monomial change; child == beam: the beam dies here;
// otherwise child is the beam that merged into this one.
struct Event {
	double time;
	double ratio;	// vol_p(periodic lattice) / vol(input lattice)
	int exponent;	// d - rank of the periodic lattice
	int child;
};

// Periodic merge tree: per vertex a beam of time-sorted events. Births are
// implicit (the vertex filtration value), the other events live in one flat
// array linked per beam; a beam that merged into another records the merger
// event as its last (death) event.
class MergeTree {
public:
	typedef std::vector<std::vector<std::tuple<double, double, int, int>>> List;

	MergeTree() : d(0), n(0), volumeInv(1) {}

	int d, n;
	double volumeInv;				// 1 / |det U|: ratio of a bounded component
	std::vector<double> birth;		// vertex filtration value = birth of beam i
	std::vector<Event> events;		// all non-birth events, in order of creation
	std::vector<int> next;			// next event of the same beam, -1 at the end
	std::vector<int> head, tail;	// first / last event of beam i, -1 if none
	std::vector<int> death;			// index of the merger event that ended beam i, -1 if alive
	std::vector<double> deathTime;	// its time (sequential copy, so the barcode scan stays cache-friendly)

	void init(int d, int n, double volumeInv, const double* f);
	void push(int beam, const Event& e);
	const Event& last(int beam) const;		// current monomial of beam i
	void beam(int i, std::vector<Event>& out) const;	// materialize beam i (birth first)
	int size() const { return n; }

	List toList() const;
	static MergeTree fromList(const List& tree);
};

MergeTree mergeTree (
	int n, 						// number of vertices
	int d, 						// dimension
	const Eigen::MatrixXd& V, 	// lattice basis
	const Eigen::MatrixXi& arcs, 
	const Eigen::VectorXd& arc_filtration,
	const Eigen::MatrixXi& arc_shift,
	Eigen::VectorXd& vertex_filtration
);

// Periodic merge tree of a lower-star filtration on a periodic grid without
// materializing the arcs: f is the field flattened in C order over `shape`
// (already negated for superlevel sets), every grid point is joined to its
// neighbors along the columns of `directions` (integer grid steps), and the
// arcs are processed in the order of their later endpoint.
MergeTree gridMergeTree (
	int d,
	const Eigen::MatrixXd& U,
	const std::vector<int>& shape,
	const Eigen::VectorXd& f,
	const Eigen::MatrixXi& directions
);

void printMergeTree(const MergeTree& tree);

// Signed barcode per exponent 0..d as (birth, death, multiplicity); with
// drop_zero_length the bars with death == birth (one per non-critical vertex
// in a grid) are left out.
std::vector<std::vector<std::tuple<double, double, double>>> barcode(int d, const MergeTree& tree, bool drop_zero_length = false);

}
