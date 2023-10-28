#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "vec.h"

template <typename T> struct projection_filter {
	std::vector<size_t> global_indices;
	std::vector<Eigen::Matrix<float, 16, 1>> reduced_vecs;
	size_t offset;

	projection_filter() = default;

	projection_filter(const std::vector<vec<T>>& vecs,
										const std::vector<size_t>& _global_indices,
										std::mt19937& gen)
			: global_indices(_global_indices) {
		if (!vecs.empty()) {
			auto distribution =
					std::uniform_int_distribution<>(0, vecs[0].size() / 16 - 1);
			offset = 16 * distribution(gen);
		}
		for (size_t i = 0; i < vecs.size(); ++i) {
			reduced_vecs.emplace_back();
			for (size_t j = 0; j < 16; ++j) {
				reduced_vecs.back()[j] = vecs[i].at(j + offset);
			}
		}
	}

	std::vector<std::pair<T, size_t>> get_top_k_vectors(const vec<T>& query,
																											size_t k) {
		k = std::min(k, reduced_vecs.size());
		std::vector<std::pair<T, size_t>> ret_combined(reduced_vecs.size());
		for (size_t i = 0; i < reduced_vecs.size(); ++i) {
			ret_combined[i].second = global_indices[i];
		}
		__m512 reduced_q = _mm512_loadu_ps(&query.at(offset));
		constexpr size_t prefetch_distance = 12;
		for (size_t i = 0; i < std::min(prefetch_distance, reduced_vecs.size());
				 ++i) {
			_mm_prefetch(reduced_vecs[i].data(), _MM_HINT_T0);
		}
		size_t i = 0;
		for (; i + prefetch_distance < reduced_vecs.size(); ++i) {
			_mm_prefetch(reduced_vecs[i + prefetch_distance].data(), _MM_HINT_T0);
			__m512 a = _mm512_loadu_ps(reduced_vecs[i].data());
			__m512 diff = _mm512_sub_ps(a, reduced_q);
			__m512 squared_diff = _mm512_mul_ps(diff, diff);
			ret_combined[i].first = _mm512_reduce_add_ps(squared_diff);
		}
		for (; i < reduced_vecs.size(); ++i) {
			__m512 a = _mm512_loadu_ps(reduced_vecs[i].data());
			__m512 diff = _mm512_sub_ps(a, reduced_q);
			__m512 squared_diff = _mm512_mul_ps(diff, diff);
			ret_combined[i].first = _mm512_reduce_add_ps(squared_diff);
		}

		auto it = ret_combined.begin() + k;
		std::nth_element(ret_combined.begin(), it, ret_combined.end());
		ret_combined.resize(k);
		return ret_combined;
	}
};
