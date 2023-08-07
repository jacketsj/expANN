#pragma once

#include <memory>
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
			// TODO this looks wrong
			ans[i] = std::lower_bound(distances[i].begin(), distances[i].end(), sd) -
							 distances[i].begin();
		}
		return ans;
	}
};

template <typename T> struct vec_generator {
	std::random_device rd;
	std::shared_ptr<std::mt19937> gen;
	std::normal_distribution<> d;
	T eps;
	size_t dim;

	vec_generator(size_t _dim)
			: rd(), gen(std::make_shared<std::mt19937>(rd())), d(0, 1), eps(1e-7),
				dim(_dim) {}

	vec_generator(size_t _dim, std::shared_ptr<std::mt19937> _gen)
			: rd(), gen(_gen), d(0, 1), eps(1e-7), dim(_dim) {}

	vec<T> random_vec() {
		vec<T> res(dim);
		do {
			for (size_t i = 0; i < res.size(); ++i)
				res[i] = d(*gen);
		} while (res.norm2() < eps);
		return res;
	}
};

template <typename T> class arragement_generator : public vec_generator<T> {
	using vec_generator<T>::random_vec;
	size_t affine_copies = 2;
	size_t num_orientations = 10;

public:
	// arragement_generator() {}
	arragement_generator(size_t _dim) : vec_generator<T>(_dim) {}
	arragement_generator(size_t _dim, size_t _affine_copies,
											 size_t _num_orientations)
			: vec_generator<T>(_dim), affine_copies(_affine_copies),
				num_orientations(_num_orientations) {
		this->dim = _dim;
	}
	arragement_generator(size_t _dim, size_t _affine_copies,
											 size_t _num_orientations,
											 std::shared_ptr<std::mt19937> _gen)
			: vec_generator<T>(_dim, _gen), affine_copies(_affine_copies),
				num_orientations(_num_orientations) {
		this->dim = _dim;
	}

	arrangement<T> operator()(vecset<T>& vs) {
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
