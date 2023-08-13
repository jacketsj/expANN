#pragma once

#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "hyperplane.h"
#include "small_vector.hpp"
#include "vec.h"
#include "vecset.h"

typedef gch::small_vector<unsigned short> arrangement_location;
// typedef std::vector<unsigned short> arrangement_location;

template <typename T> struct arrangement {
	std::vector<hyperplane<T>> orientations;
	std::vector<std::vector<T>> distances;

	// lazy/simple implementation
	arrangement_location compute_multiindex(const vec<T>& v) {
		arrangement_location ans(orientations.size());
		for (size_t i = 0; i < orientations.size(); ++i) {
			T sd = orientations[i].signeddist(v);
			ans[i] = std::lower_bound(distances[i].begin(), distances[i].end(), sd) -
							 distances[i].begin();
		}
		return ans;
	}

	// get neighbouring arrangement cells (for hamming distance graph)
	std::vector<arrangement_location> neighbours(arrangement_location loc) {
		std::vector<arrangement_location> ret;
		for (size_t i = 0; i < orientations.size(); ++i)
			for (size_t delta : {-1, 1})
				if (int(loc[i]) + delta >= 0 &&
						int(loc[i]) + delta <= distances[i].size()) {
					arrangement_location next = loc;
					next[i] += delta;
					ret.emplace_back(next);
				}
		return ret;
	}

	std::vector<arrangement_location> random_probes(arrangement_location loc,
																									size_t num_probes,
																									std::mt19937& gen) {
		// do a random traversal to get up to num_probes close cells
		// return value includes loc (not counted in the cell count)
		std::set<arrangement_location> visited;
		std::priority_queue<std::pair<size_t, arrangement_location>> to_visit;
		to_visit.emplace(0, loc);
		std::uniform_int_distribution<> distrib(0, std::numeric_limits<int>::max());
		while (!to_visit.empty() && visited.size() <= num_probes) {
			arrangement_location cur = to_visit.top().second;
			to_visit.pop();
			if (visited.contains(cur))
				continue;
			visited.emplace(cur);
			for (auto& neighbour : neighbours(cur))
				to_visit.emplace(distrib(gen), neighbour);
		}
		std::vector<arrangement_location> ret;
		for (auto& next : visited)
			ret.emplace_back(next);
		return ret;
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
		vec<T> res;
		res.set_dim(dim);
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
