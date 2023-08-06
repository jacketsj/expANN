#pragma once

#include <vector>

#include "dataset.h"
#include "vec.h"

template <typename T>
struct in_memory_dataset : public dataset<T, in_memory_dataset<T>> {
	std::vector<vec<T>> all_vecs;
	vec<T> _get_vec(size_t i) const { return all_vecs[i]; };
};

template <typename T>
struct in_memory_test_dataset
		: public test_dataset<T, in_memory_test_dataset<T>, in_memory_dataset<T>> {
	std::vector<vec<T>> all_query_vecs;
	std::vector<std::vector<size_t>> all_query_ans;
	vec<T> _get_query(size_t i) const { return all_query_vecs[i]; }
	std::vector<size_t> _get_query_ans(size_t i) const {
		return all_query_ans[i];
	}
};
