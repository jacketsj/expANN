#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <vector>

#include "vec.h"

struct product_quantizer {
	using Vector_t = vec<float>::Underlying;
	// TODO fix matrix type as well
	using Matrix_t = Eigen::MatrixXf;
	Matrix_t centroids_;
	size_t subvector_size_, num_subvectors_;

	using codes_t = std::vector<uint8_t>;
	using approx_distance_table = std::vector<std::vector<double>>;

	product_quantizer() = default;
	product_quantizer(const std::vector<Vector_t>& centroids,
										size_t subvector_size)
			: subvector_size_(subvector_size),
				num_subvectors_(centroids.size() / subvector_size) {
		if (subvector_size_ <= 0 || centroids.size() % subvector_size_ != 0) {
			throw std::invalid_argument("Invalid subvector size");
		}
		if (subvector_size_ > 256) {
			throw std::invalid_argument("Subvector size must be less than or equal "
																	"to 256 for uint8_t encoding");
		}
		if (centroids.empty()) {
			throw std::invalid_argument("Centroids vector cannot be empty");
		}

		size_t dimension = centroids[0].size();
		centroids_ = Matrix_t(dimension, centroids.size());
		for (size_t i = 0; i < centroids.size(); ++i) {
			// if (centroids[i].size() != dimension) {
			// 	throw std::invalid_argument(
			// 			"All centroids must have the same dimension");
			// }
			centroids_.col(i) = centroids[i];
		}
	}

	codes_t encode(const Vector_t& vector) const {
		codes_t codes(num_subvectors_);
		for (size_t i = 0; i < num_subvectors_; ++i) {
			Vector_t subvector = vector.segment(i * subvector_size_, subvector_size_);
			codes[i] = static_cast<uint8_t>(
					(centroids_.middleRows(i * subvector_size_, subvector_size_)
							 .colwise() -
					 subvector)
							.rowwise()
							.squaredNorm()
							.minCoeff());
		}
		return codes;
	}

	Vector_t decode(const codes_t& codes) const {
		Vector_t reconstructed_vector(centroids_.cols());
		for (size_t i = 0; i < num_subvectors_; ++i) {
			reconstructed_vector.segment(i * subvector_size_, subvector_size_) =
					centroids_.row(codes[i]);
		}
		return reconstructed_vector;
	}

	approx_distance_table
	fill_approx_distance_table(const Vector_t& query_vector) const {
		approx_distance_table table(num_subvectors_,
																std::vector<double>(subvector_size_));
		for (size_t i = 0; i < num_subvectors_; ++i) {
			Vector_t subvector =
					query_vector.segment(i * subvector_size_, subvector_size_);
			for (size_t j = 0; j < subvector_size_; ++j) {
				Vector_t centroid = centroids_.row(i * subvector_size_ + j);
				table[i][j] = (centroid - subvector).squaredNorm();
			}
		}
		return table;
	}

	double compute_approx_distance(const approx_distance_table& table,
																 const std::vector<uint8_t>& codes) const {
		double distance = 0.0;
		for (size_t i = 0; i < num_subvectors_; ++i) {
			distance += table[i][codes[i]];
		}
		return distance;
	}

	std::vector<int>
	get_top_k_vectors(const Vector_t& query,
										const std::vector<std::vector<uint8_t>>& codes_list,
										size_t k) const {

		approx_distance_table query_dist_table = fill_approx_distance_table(query);

		// TODO slightly improve this, maybe just sort instead
		using distance_index_pair = std::pair<double, int>;
		std::priority_queue<distance_index_pair, std::vector<distance_index_pair>,
												std::greater<distance_index_pair>>
				min_heap;

		for (size_t i = 0; i < codes_list.size(); ++i) {
			double distance =
					compute_approx_distance(query_dist_table, codes_list[i]);
			if (min_heap.size() < k) {
				min_heap.emplace(distance, i);
			} else if (distance < min_heap.top().first) {
				min_heap.pop();
				min_heap.emplace(distance, i);
			}
		}

		std::vector<int> result;
		while (!min_heap.empty()) {
			result.push_back(min_heap.top().second);
			min_heap.pop();
		}
		std::reverse(result.begin(), result.end());
		return result;
	}
};

template <typename T> struct pq_searcher {
	product_quantizer pq_;
	std::vector<std::vector<uint8_t>> codes_;

	pq_searcher() = default;
	pq_searcher(const std::vector<vec<T>>& centroids, int subvector_size,
							const std::vector<vec<T>>& vectors) {
		std::vector<typename vec<T>::Underlying> centroids_underlying;
		for (auto& v : centroids)
			centroids_underlying.emplace_back(v.get_underlying());
		pq_ = product_quantizer(centroids_underlying, subvector_size);
		for (const auto& v : vectors) {
			codes_.push_back(pq_.encode(v.get_underlying()));
		}
	}

	std::vector<int> get_top_k_vectors(const vec<T>& query, int k) const {
		return pq_.get_top_k_vectors(query.get_underlying(), codes_, k);
	}
};
