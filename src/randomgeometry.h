#pragma once

#include <random>
#include <vector>

#include "hyperplane.h"
#include "vec.h"
#include "vecset.h"

template <typename T> struct arrangement {
	std::vector<hyperplane<T>> orientations;
	std::vector<std::vector<T>> distances;

	// lazy/simple implementation
	std::vector<unsigned short> compute_multiindex(const vec<T>& v) {
		std::vector<unsigned short> ans(orientations.size());
		for (size_t i = 0; i < orientations.size(); ++i) {
			T sd = orientations[i].signeddist(v);
			ans[i] = std::lower_bound(distances[i].begin(), distances[i].end(), sd) -
							 distances[i].begin();
		}
		return ans;
	}
};

template <typename T> struct vec_generator {
	std::random_device rd;
	std::mt19937 gen;
	std::normal_distribution<> d;
	T eps;

	vec_generator() : rd(), gen(rd()), d(0, 1), eps(1e-7) {}

	vec<T> random_vec() {
		vec<T> res;
		do {
			for (size_t i = 0; i < res.size(); ++i)
				res[i] = d(gen);
		} while (res.norm2() < eps);
		return res;
	}
};

template <typename T> class arragement_generator : public vec_generator<T> {
	using vec_generator<T>::random_vec;
	size_t affine_copies = 4;
	size_t num_orientations = 64;

public:
	arragement_generator() {} //  : rd(), gen(rd()), d(0, 1), eps(1e-7) {}

	arrangement<T> operator()(const vecset<T>& vs) {
		arrangement<T> res;
		for (size_t i = 0; i < num_orientations; ++i) {
			auto orientation = hyperplane(random_vec());
			std::vector<T> distance_samples;
			for (const vec<T>& v : vs.sample(affine_copies))
				distance_samples.emplace_back(orientation.signeddist(v));
			res.orientations.emplace_back(std::move(orientation));
			res.distances.emplace_back(std::move(distance_samples));
		}
		return res;
	}
};

// TODO actually implement ANN and partioning, as well as hierarchical
// partitioning
// TODO also turn multiindex into a bitset or something, and then hash it maybe
