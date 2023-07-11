#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "vec.h"

template <typename T> class vecset {
	std::random_device rd;
	std::mt19937 gen;
	// std::uniform_int_distribution<> d;
	std::vector<vec<T>> internal;

public:
	// vecset() : rd(), gen(rd()), d(0, 0), internal({0}) {}
	vecset(const std::vector<vec<T>>& _internal)
			: rd(), gen(rd),
				// d(0, _internal.size() - 1),
				internal(_internal) {}

	const std::vector<vec<T>> sample(size_t n) {
		assert(n <= internal.size());
		std::vector<vec<T>> res;
		std::sample(internal.begin(), internal.end(), std::back_inserter(res), n,
								gen);
		return res;
	}

	typedef typename std::vector<vec<T>>::const_iterator const_iterator;
	const_iterator begin() const { return internal.const_container.begin(); }
	const_iterator end() const { return internal.const_container.end(); }
};
