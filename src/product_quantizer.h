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

	std::vector<size_t> get_top_k_vectors(const Vector_t& query,
																				const std::vector<codes_t>& codes_list,
																				size_t k) const {
		k = std::min(k, codes_list.size());

		// Compute distance table
		std::vector<Eigen::MatrixXf> distance_tables(sub_centroids.size());
		for (size_t i = 0; i < sub_centroids.size(); ++i) {
			distance_tables[i] =
					(sub_centroids[i].colwise() -
					 query.segment(i * sub_centroids[i].rows(), sub_centroids[i].rows()))
							.colwise()
							.squaredNorm();
		}

		// Compute distances for each vector in codes_list
		std::vector<std::pair<float, size_t>> distances(codes_list.size());
		for (size_t i = 0; i < codes_list.size(); ++i) {
			float dist = 0.0;
			for (size_t j = 0; j < codes_list[i].size(); ++j) {
				dist += distance_tables[j](0, codes_list[i][j]);
			}
			distances[i] = {dist, i};
		}

		// Find the top-k closest vectors
		std::partial_sort(distances.begin(), distances.begin() + k,
											distances.end());
		std::vector<size_t> result(k);
		for (size_t i = 0; i < k; ++i) {
			result[i] = distances[i].second;
		}
		return result;

		// no distance table version (works)
		/*
		std::vector<std::pair<float, size_t>> distances(codes_list.size());

		for (size_t i = 0; i < codes_list.size(); ++i) {
			float dist = 0.0f;
			for (size_t j = 0; j < codes_list[i].size(); ++j) {
				Eigen::VectorXf centroid = sub_centroids[j].col(codes_list[i][j]);
				dist += (query.segment(j * centroid.size(), centroid.size()) - centroid)
										.squaredNorm();
			}
			distances[i] = {dist, i};
		}

		std::partial_sort(distances.begin(), distances.begin() + k,
											distances.end());
		std::vector<size_t> result(k);
		for (size_t i = 0; i < k; ++i) {
			result[i] = distances[i].second;
		}
		return result;
		*/
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
