#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "vec.h"

struct product_quantizer {
	using Vector_t = vec<float>::Underlying;
	// TODO fix matrix type as well
	using Matrix_t = Eigen::MatrixXf;
	using codes_t = std::vector<uint8_t>;
	std::vector<Matrix_t> sub_centroids;
	size_t subvector_size;

	product_quantizer() = default;

	product_quantizer(const std::vector<Vector_t>& centroids,
										size_t subvector_size)
			: subvector_size(subvector_size) {
		size_t num_subvectors = centroids[0].size() / subvector_size;
		for (size_t i = 0; i < num_subvectors; ++i) {
			Matrix_t sc(centroids.size(), subvector_size);
			for (size_t j = 0; j < centroids.size(); ++j)
				sc.row(j) = centroids[j]
												.segment(i * subvector_size, subvector_size)
												.transpose();
			sub_centroids.push_back(std::move(sc));
		}
	}

	codes_t encode(const Vector_t& v) const {
		/*
		codes_t codes;
		for (const auto& sc : sub_centroids) {
			size_t min_idx = 0;
			for (size_t i = 1; i < sc.rows(); ++i)
				if ((v - sc.row(i)).squaredNorm() < (v - sc.row(min_idx)).squaredNorm())
					min_idx = i;
			codes.push_back(static_cast<uint8_t>(min_idx));
		}
		return codes;
		*/

		codes_t codes;
		for (const auto& sc : sub_centroids) {
			Eigen::VectorXf v_sub =
					v.segment(codes.size() * subvector_size, subvector_size);
			Eigen::VectorXf dists =
					(sc.rowwise() - v_sub.transpose()).rowwise().squaredNorm();
			ptrdiff_t min_idx;
			dists.minCoeff(&min_idx);
			codes.push_back(static_cast<uint8_t>(min_idx));
		}
		return codes;
	}

	std::vector<size_t> get_top_k_vectors(const Vector_t& query,
																				const std::vector<codes_t>& codes_list,
																				size_t k) const {
		std::vector<std::vector<float>> dist_table(
				sub_centroids.size(), std::vector<float>(sub_centroids[0].rows()));
		for (size_t i = 0; i < sub_centroids.size(); ++i)
			for (size_t j = 0; j < sub_centroids[i].rows(); ++j)
				dist_table[i][j] = (query.segment(i * subvector_size, subvector_size) -
														sub_centroids[i].row(j))
															 .squaredNorm();

		std::vector<std::pair<float, size_t>> distances(codes_list.size());
		for (size_t i = 0; i < codes_list.size(); ++i) {
			distances[i].second = i;
			for (size_t j = 0; j < codes_list[i].size(); ++j)
				distances[i].first += dist_table[j][codes_list[i][j]];
		}

		std::partial_sort(distances.begin(), distances.begin() + k,
											distances.end());

		std::vector<size_t> result(k);
		for (size_t i = 0; i < k; ++i)
			result[i] = distances[i].second;
		return result;
	}
};

template <typename T> struct pq_searcher {
	product_quantizer pq_;
	std::vector<std::vector<uint8_t>> codes_;

	pq_searcher() = default;
	pq_searcher(std::vector<vec<T>> centroids, int subvector_size,
							const std::vector<vec<T>>& vectors) {
		std::vector<typename vec<T>::Underlying> centroids_underlying;
		if (centroids.empty()) { // fill with origin if empty
			centroids.emplace_back(std::vector<T>(DIM, 0));
		}
		for (auto& v : centroids)
			centroids_underlying.emplace_back(v.get_underlying());
		pq_ = product_quantizer(centroids_underlying, subvector_size);
		for (const auto& v : vectors) {
			codes_.push_back(pq_.encode(v.get_underlying()));
		}
	}

	std::vector<size_t> get_top_k_vectors(const vec<T>& query, size_t k) const {
		return pq_.get_top_k_vectors(query.get_underlying(), codes_, k);
	}
};
