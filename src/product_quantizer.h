#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "vec.h"

struct product_quantizer {
	using Vector_t = vec<float>::Underlying;
	// TODO fix matrix type as well, plus the VectorXf in the searcher
	using Matrix_t = Eigen::MatrixXf;
	using codes_t = std::vector<uint8_t>;
	std::vector<Matrix_t> sub_centroids;
	size_t subvector_size;

	product_quantizer() = default;

	product_quantizer(const std::vector<Vector_t>& centroids,
										size_t subvector_size)
			: subvector_size(subvector_size) {
		size_t num_subvectors = centroids[0].size() / subvector_size;
		sub_centroids.resize(num_subvectors);
		for (size_t i = 0; i < num_subvectors; ++i) {
			sub_centroids[i] = Matrix_t(subvector_size, centroids.size());
			for (size_t j = 0; j < centroids.size(); ++j)
				sub_centroids[i].col(j) =
						centroids[j].segment(i * subvector_size, subvector_size);
		}
	}

	codes_t encode(const Vector_t& v) const {
		codes_t codes(sub_centroids.size());
		for (size_t i = 0; i < sub_centroids.size(); ++i) {
			Eigen::VectorXf subvector =
					v.segment(i * sub_centroids[i].rows(), sub_centroids[i].rows());
			Eigen::MatrixXf diffs = sub_centroids[i].colwise() - subvector;
			Eigen::VectorXf dists = diffs.colwise().squaredNorm();
			Eigen::VectorXf::Index minIndex;
			dists.minCoeff(&minIndex);
			codes[i] = static_cast<uint8_t>(minIndex);
		}
		return codes;
	}

	void get_top_k_vectors(const Vector_t& query,
												 std::vector<Eigen::VectorXf>& distance_tables,
												 std::vector<std::pair<float, size_t>> ret,
												 const std::vector<codes_t>& codes_list,
												 size_t k) const {
		k = std::min(k, codes_list.size());

		// Compute distance table
		// std::vector<Eigen::VectorXf> distance_tables(sub_centroids.size());
		for (size_t i = 0; i < sub_centroids.size(); ++i) {
			distance_tables[i] =
					(sub_centroids[i].colwise() -
					 query.segment(i * sub_centroids[i].rows(), sub_centroids[i].rows()))
							.colwise()
							.squaredNorm();
		}

		// Compute distances for each vector in codes_list
		for (size_t i = 0; i < codes_list.size(); ++i) {
			for (size_t j = 0; j < codes_list[i].size(); ++j) {
				ret[i].first += distance_tables[j](0, codes_list[i][j]);
			}
		}

		// Find the top-k closest vectors
		auto it = ret.begin() + k;
		std::nth_element(ret.begin(), it, ret.end());
		ret.resize(k);
		// std::vector<size_t> result(k);
		// for (size_t i = 0; i < k; ++i) {
		//	result[i] = distances[i].second;
		// }
		// return result;
	}
};

template <typename T> struct pq_searcher {
	product_quantizer pq_;
	std::vector<std::vector<uint8_t>> codes_;
	std::vector<Eigen::VectorXf> distance_tables;
	std::vector<std::pair<T, size_t>> stored;
	std::vector<std::pair<T, size_t>> ret;

	pq_searcher() = default;
	pq_searcher(std::vector<vec<T>> centroids, int subvector_size,
							const std::vector<size_t>& indices,
							const std::vector<vec<T>>& vectors) {
		std::vector<typename vec<T>::Underlying> centroids_underlying;
		if (centroids.empty()) // fill with origin if empty
			centroids.emplace_back(std::vector<T>(DIM, 0));
		for (auto& v : centroids)
			centroids_underlying.emplace_back(v.get_underlying());
		pq_ = product_quantizer(centroids_underlying, subvector_size);
		for (const auto& v : vectors)
			codes_.push_back(pq_.encode(v.get_underlying()));
		for (const auto& i : indices)
			stored.emplace_back(0.0, i);
		distance_tables = std::vector<Eigen::VectorXf>(pq_.sub_centroids.size());
	}

	// std::vector<size_t> get_top_k_vectors(const vec<T>& query, size_t k) const
	// { 	return pq_.get_top_k_vectors(query.get_underlying(), codes_, k);
	// }
	const std::vector<std::pair<T, size_t>>&
	get_top_k_vectors(const vec<T>& query, size_t k) {
		ret = stored;
		pq_.get_top_k_vectors(query.get_underlying(), distance_tables, ret, codes_,
													k);
		return ret;
	}
};
