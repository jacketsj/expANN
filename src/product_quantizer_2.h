#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "vec.h"

struct product_quantizer_2 {
	using Vector_t = vec<float>::Underlying;
	// TODO fix matrix type as well, plus the VectorXf in the searcher
	using Matrix_t = Eigen::MatrixXf;
	using codes_t = std::vector<uint8_t>;
	std::vector<Matrix_t> sub_centroids;
	size_t subvector_size;

	product_quantizer_2() = default;

	product_quantizer_2(const std::vector<Vector_t>& centroids,
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

	std::vector<codes_t> encode_many(const std::vector<Vector_t>& vs) const {
		std::vector<codes_t> ret;
		for (auto& v : vs)
			ret.emplace_back(encode(v));
		return ret;
	}

	std::vector<Eigen::VectorXf>
	compute_distance_tables(const Vector_t& query) const {
		std::vector<Eigen::VectorXf> distance_tables(sub_centroids.size());
		for (size_t i = 0; i < sub_centroids.size(); ++i) {
			distance_tables[i] =
					(sub_centroids[i].colwise() -
					 query.segment(i * sub_centroids[i].rows(), sub_centroids[i].rows()))
							.colwise()
							.squaredNorm();
			return distance_tables;
		}
	}

	void compute_distances(
			const std::vector<Eigen::VectorXf>& distance_tables,
			const std::vector<std::reference_wrapper<codes_t>>& codes_list,
			std::vector<std::pair<float, size_t>> ret) const {
		// Compute distances for each vector in codes_list
		for (size_t i = 0; i < codes_list.size(); ++i) {
			for (size_t j = 0; j < codes_list[i].size(); ++j) {
				ret[i].first += distance_tables[j](0, codes_list[i].get()[j]);
			}
		}
	}
};

template <typename T> struct pq_searcher_2 {
	const product_quantizer_2& pq_;
	std::vector<Eigen::VectorXf> distance_tables;

	pq_searcher_2() = default;
	pq_searcher_2(const product_quantizer_2& pq, vec<T> query)
			: pq_(pq),
				distance_tables(pq.compute_distance_tables(query.get_underlying())) {}

	void compute_distances(
			const std::vector<std::reference_wrapper<codes_t>>& codes_list,
			std::vector<std::pair<float, size_t>>& ret) {
		pq_.compute_distances(distance_tables, codes_list, ret);
	}
};
