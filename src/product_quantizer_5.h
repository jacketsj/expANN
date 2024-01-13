#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

template <typename T, size_t Dimension, size_t NumSubspaces,
					size_t NumCentroids>
struct product_quantizer {
	static_assert(NumCentroids <= 1 << (16 - 1),
								"NumCentroids must be 65536 or less.");
	static_assert(Dimension % NumSubspaces == 0,
								"NumSubspaces must divide Dimension.");

	static constexpr auto BitsPerCode = []() {
		for (size_t bits : {1, 2, 4, 8, 16})
			if (NumCentroids <= 1 << (bits - 1))
				return bits;
	};

	using FullVector_t = Eigen::Matrix<T, Dimension, 1>;
	using SubVector_t = Eigen::Matrix<T, Dimension / NumSubspaces, 1>;
	static constexpr auto PackedCodeSize = NumCentroids <= 256 ? 8 : 16;
	static constexpr auto CodesPerPackedCode = PackedCodeSize / BitsPerCode();
	static constexpr auto NumPackedCodes =
			(NumSubspaces + CodesPerPackedCode - 1) / CodesPerPackedCode;
	using PackedCodeType =
			std::conditional_t<PackedCodeSize == 8, uint8_t, uint16_t>;
	using CodesMatrix_t =
			Eigen::Matrix<PackedCodeType, NumPackedCodes, Eigen::Dynamic>;

	using CentroidMatrix_t = Eigen::Matrix<T, Dimension, NumCentroids>;
	using DistanceMatrix_t = Eigen::Matrix<T, NumSubspaces, NumCentroids>;

	CentroidMatrix_t centroids;
	CodesMatrix_t codes_matrix;

	product_quantizer(const std::vector<FullVector_t>& data_vecs) {
		assert(data_vecs.size() > 0 &&
					 data_vecs[0].size() == NumSubspaces * subvector_size);

		// TODO-A Initialize segments of centroids for each subspace
		// TODO-B Use k-means and k-means++

		// TODO-C Encoding the data vectors by choosing the closest segment
	}

	std::vector<T>
	apply_codes_to_table(const DistanceMatrix_t& distance_matrix) const {
		// TODO-D iterate through codes_matrix and compute approximate
	}

	DistanceMatrix_t compute_distance_table(const FullVector_t& query) const {
		// TODO-E compute a distance table
	}

	std::vector<T> compute_distances(const FullVector_t& query) const {
		return apply_codes_to_table(compute_distance_table(query));
	}
};
