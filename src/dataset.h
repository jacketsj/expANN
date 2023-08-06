#pragma once

#include <vector>

#include "vec.h"

// TODO add a 'name' field to both dataset and test_dataset (maybe just the
// former?)
template <typename T, class Derived> struct dataset {
	size_t dim;
	size_t n; // number of vectors
	std::string name;
	vec<T> get_vec(size_t i) const {
		// assert(i < n);
		return static_cast<const Derived*>(this)->_get_vec(i);
	}
};

template <typename T, class DerivedTest, class Derived>
struct test_dataset : public Derived {
	size_t m; // number of queries
	size_t k; // size of query ans
	vec<T> get_query(size_t i) const {
		// assert(i < m);
		return static_cast<const DerivedTest*>(this)->_get_query(i);
	}
	std::vector<size_t> get_query_ans(size_t i) const {
		// assert(i < m);
		return static_cast<const DerivedTest*>(this)->_get_query_ans(i);
	}
};
