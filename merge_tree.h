#pragma once
#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Dense>

namespace PMT {
std::vector<std::vector<std::tuple<double, double, int, int>>> mergeTree (
	int n, 						// number of vertices
	int d, 						// dimension
	const Eigen::MatrixXd& V, 	// lattice basis
	const Eigen::MatrixXi& arcs, 
	const Eigen::VectorXd& arc_filtration,
	const Eigen::MatrixXi& arc_shift,
	Eigen::VectorXd& vertex_filtration
);

void printMergeTree(
	const std::vector<std::vector<std::tuple<double, double, int, int>>>& tree
);

std::vector<std::vector<std::tuple<double, double, double>>> barcode(
	int d,
	const std::vector<std::vector<std::tuple<double, double, int, int>>>& tree
);

}
