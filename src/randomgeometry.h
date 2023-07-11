#pragma once

#include <random>

#include "hyperplane.h"
#include "vec.h"
#include "vecset.h"

template <typename T> class arragement_generator {
	std::random_device rd;
	std::mt19937 gen;
	std::normal_distribution<> d;
	T eps;
	size_t affine_copies = 4;
	size_t num_orientations = 64;

public:
	arragement_generator() : rd(), gen(rd()), d(0, 1), eps(1e-7) {}

	vec<T> random_vec() {
		vec<T> res;
		do {
			for (size_t i = 0; i < res.size(); ++i)
				res[i] = d(gen);
		} while (res.norm2() < eps);
		return res;
	}
};
