#pragma once
#include "Persistence_image_impl.h"
// #include <gudhi/common_persistence_representations.h>

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <cmath>

Eigen::MatrixXd persistenceImage(const std::vector<std::tuple<double, double, double>>& barcode, int size, double min, double max);
