#pragma once

#include <vector>

#include "antitopo_engine.h"
#include "vec.h"

std::vector<std::vector<size_t>>
accel_k_means(const std::vector<vec<float>>& vecs, size_t k,
							antitopo_engine_config conf);
