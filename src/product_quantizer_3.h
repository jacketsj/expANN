#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "vec.h"

struct product_quantizer_3 {
	using Vector_t = vec<float>::Underlying;
	using Matrix_t = Eigen::MatrixXf;
	using codes_t = std::vector<uint8_t>;
	std::vector<Matrix_t> sub_centroids;
	size_t subvector_size;
	std::vector<codes_t> codes_list;

	product_quantizer_3() = default;

	product_quantizer_3(const std::vector<Vector_t>& data_vecs,
											size_t num_centroids, size_t subvector_size)
			: subvector_size(subvector_size) {
		num_centroids = std::min(num_centroids, data_vecs.size());
		auto centroids = k_means_cluster(data_vecs, num_centroids);
		size_t num_subvectors = centroids[0].size() / subvector_size;
		sub_centroids.resize(num_subvectors);
		for (size_t i = 0; i < num_subvectors; ++i) {
			sub_centroids[i] = Matrix_t(subvector_size, centroids.size());
			for (size_t j = 0; j < centroids.size(); ++j)
				sub_centroids[i].col(j) =
						centroids[j].segment(i * subvector_size, subvector_size);
		}
		codes_list = encode_many(data_vecs);
	}

	std::vector<Vector_t>
	k_means_cluster(const std::vector<Vector_t>& data_vectors, size_t k) {
		std::vector<Vector_t> centroids(k);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, data_vectors.size() - 1);
		for (auto& centroid : centroids) {
			centroid = data_vectors[dis(gen)];
		}

		bool changed;
		std::vector<int> labels(data_vectors.size());
		do {
			changed = false;
			for (size_t i = 0; i < data_vectors.size(); ++i) {
				int closest = std::distance(
						centroids.begin(),
						std::min_element(centroids.begin(), centroids.end(),
														 [&](const Vector_t& a, const Vector_t& b) {
															 return (data_vectors[i] - a).squaredNorm() <
																			(data_vectors[i] - b).squaredNorm();
														 }));
				if (labels[i] != closest) {
					labels[i] = closest;
					changed = true;
				}
			}
			std::vector<int> counts(k, 0);
			std::fill(centroids.begin(), centroids.end(),
								Vector_t::Zero(data_vectors[0].size()));
			for (size_t i = 0; i < data_vectors.size(); ++i) {
				centroids[labels[i]] += data_vectors[i];
				counts[labels[i]]++;
			}
			for (size_t i = 0; i < k; ++i) {
				if (counts[i] > 0) {
					centroids[i] /= counts[i];
				}
			}
		} while (changed);

		return centroids;
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
		}
		return distance_tables;
	}

	void compute_distances_given_table(
			const std::vector<Eigen::VectorXf>& distance_tables,
			std::vector<float>& ret) const {
		// Compute distances for each vector in codes_list
		for (size_t i = 0; i < codes_list.size(); ++i) {
			for (size_t j = 0; j < codes_list[i].size(); ++j) {
				ret[i] += distance_tables[j](0, codes_list[i][j]);
			}
		}
	}

	std::vector<float> compute_distances(const vec<float>& query) const {
		std::vector<float> ret(codes_list.size());
		compute_distances_given_table(
				compute_distance_tables(query.get_underlying()), ret);
		return ret;
	}
};
