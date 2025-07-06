#pragma once
#include "persistence_image_impl.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <cmath>

Eigen::MatrixXd persistenceImage(const std::vector<std::tuple<double, double, double>>& barcode, int size, double min, double max);
