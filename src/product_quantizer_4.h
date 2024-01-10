#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "vec.h"

struct product_quantizer_4 {
	using Vector_t = vec<float>::Underlying; // Data vec dimension'd
	using Matrix_t = Eigen::MatrixXf;
	using codes_t = std::vector<uint8_t>;
	size_t subvector_size;
	std::vector<Matrix_t> centroids; // Centroids for each subvector
	std::vector<codes_t> codes_list; // Encoded vectors

	product_quantizer_4() = default;

	product_quantizer_4(const std::vector<Vector_t>& data_vecs,
											size_t num_centroids, size_t subvector_size)
			: subvector_size(subvector_size) {
		// Split vectors and perform k-means for each segment
		size_t num_subvectors = data_vecs[0].size() / subvector_size;
		for (size_t i = 0; i < num_subvectors; ++i) {
			std::vector<Vector_t> sub_data_vecs;
			for (const auto& v : data_vecs) {
				sub_data_vecs.push_back(v.segment(i * subvector_size, subvector_size));
			}
			centroids.push_back(k_means_cluster(sub_data_vecs, num_centroids));
		}

		// Encode all vectors
		for (const auto& v : data_vecs) {
			codes_list.push_back(encode(v));
		}
	}

	// Improved k-means++ initialization
	Matrix_t k_means_plus_plus_init(const std::vector<Vector_t>& data_vectors,
																	size_t k) {
		Matrix_t centers(subvector_size, k);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> distrib(0, data_vectors.size() - 1);

		centers.col(0) = data_vectors[distrib(gen)];
		for (size_t i = 1; i < k; ++i) {
			Eigen::VectorXf min_dists = Eigen::VectorXf::Constant(
					data_vectors.size(), std::numeric_limits<float>::max());
			for (size_t j = 0; j < data_vectors.size(); ++j) {
				for (size_t c = 0; c < i; ++c) {
					float dist = (data_vectors[j] - centers.col(c)).squaredNorm();
					min_dists(j) = std::min(min_dists(j), dist);
				}
			}
			float sum = min_dists.sum();
			std::uniform_real_distribution<> dist(0, sum);
			float rnd = dist(gen);
			for (size_t j = 0; j < data_vectors.size(); ++j) {
				rnd -= min_dists(j);
				if (rnd <= 0) {
					centers.col(i) = data_vectors[j];
					break;
				}
			}
		}
		return centers;
	}

	// Updated k-means function using k-means++ initialization
	Matrix_t k_means_cluster(const std::vector<Vector_t>& data_vectors,
													 size_t k) {
		Matrix_t centers = k_means_plus_plus_init(data_vectors, k);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> distrib(0, data_vectors.size() - 1);
		for (size_t i = 0; i < k; ++i) {
			centers.col(i) = data_vectors[distrib(gen)];
		}

		std::vector<int> labels(data_vectors.size());
		bool changed;
		int iters = 50;
		do {
			changed = false;

			// Assign labels
			for (size_t i = 0; i < data_vectors.size(); ++i) {
				int best_j = 0;
				float best_dist = (data_vectors[i] - centers.col(0)).squaredNorm();
				for (size_t j = 1; j < k; ++j) {
					float dist = (data_vectors[i] - centers.col(j)).squaredNorm();
					if (dist < best_dist) {
						best_j = j;
						best_dist = dist;
					}
				}
				if (labels[i] != best_j) {
					labels[i] = best_j;
					changed = true;
				}
			}

			// Update centroids
			centers = Matrix_t::Zero(subvector_size, k);
			std::vector<int> count(k, 0);
			for (size_t i = 0; i < data_vectors.size(); ++i) {
				centers.col(labels[i]) += data_vectors[i];
				++count[labels[i]];
			}
			for (size_t j = 0; j < k; ++j) {
				if (count[j] > 0) {
					centers.col(j) /= count[j];
				}
			}
		} while (changed && --iters > 0);

		return centers;
	}

	codes_t encode(const Vector_t& v) const {
		codes_t code;
		for (size_t i = 0; i < centroids.size(); ++i) {
			const auto& subvector = v.segment(i * subvector_size, subvector_size);
			float min_dist = (subvector - centroids[i].col(0)).squaredNorm();
			int min_idx = 0;
			for (int j = 1; j < centroids[i].cols(); ++j) {
				float dist = (subvector - centroids[i].col(j)).squaredNorm();
				if (dist < min_dist) {
					min_dist = dist;
					min_idx = j;
				}
			}
			code.push_back(static_cast<uint8_t>(min_idx));
		}
		return code;
	}

	std::vector<codes_t> encode_many(const std::vector<Vector_t>& vs) const {
		std::vector<codes_t> ret;
		for (const auto& v : vs)
			ret.emplace_back(encode(v));
		return ret;
	}

	std::vector<Eigen::VectorXf>
	compute_distance_tables(const Vector_t& query) const {
		std::vector<Eigen::VectorXf> distance_tables;
		for (size_t i = 0; i < centroids.size(); ++i) {
			const auto& subvector = query.segment(i * subvector_size, subvector_size);
			Eigen::VectorXf distances(centroids[i].cols());
			for (int j = 0; j < centroids[i].cols(); ++j) {
				distances(j) = (subvector - centroids[i].col(j)).norm();
			}
			distance_tables.push_back(distances);
		}
		return distance_tables;
	}

	void compute_distances_given_table(
			const std::vector<Eigen::VectorXf>& distance_tables,
			std::vector<float>& ret) const {
		for (size_t i = 0; i < codes_list.size(); ++i) {
			float distance = 0;
			for (size_t j = 0; j < distance_tables.size(); ++j) {
				distance += distance_tables[j][codes_list[i][j]];
			}
			ret[i] = distance;
		}
	}

	std::vector<float> compute_distances(const vec<float>& query) const {
		std::vector<float> ret(codes_list.size());
		compute_distances_given_table(
				compute_distance_tables(query.get_underlying()), ret);
		return ret;
	}
};
