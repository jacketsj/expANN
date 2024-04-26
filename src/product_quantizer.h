#pragma once

#include "clusterer.h"
#include "quantizer.h"
#include <iostream>
#include <optional>

class product_quantizer_scorer : public quantized_scorer {
	using fvec = vec<float>::Underlying;
	fvec query;

	static constexpr size_t NUM_SUBSPACES = 16;
	static constexpr size_t NUM_CENTROIDS = 16;

	using codes_t = std::array<uint8_t, NUM_SUBSPACES>;

	const std::vector<std::vector<codes_t>>& codes;
	const std::vector<std::array<std::array<Eigen::VectorXf, NUM_CENTROIDS>,
															 NUM_SUBSPACES>>& sub_centroids;

public:
	product_quantizer_scorer(
			const fvec& _query, const std::vector<std::vector<codes_t>>& _codes,
			const std::vector<std::array<std::array<Eigen::VectorXf, NUM_CENTROIDS>,
																	 NUM_SUBSPACES>>& _sub_centroids)
			: query(_query), codes(_codes), sub_centroids(_sub_centroids) {}

	virtual float score(size_t index) override { return 0; }
	virtual void prefetch(size_t index) override {}
	virtual void filter_by_score(size_t cur_vert,
															 const std::vector<size_t>& to_filter,
															 const std::vector<size_t>& to_filter_offsets,
															 std::vector<size_t>& filtered_out,
															 std::vector<float>& distances,
															 float cutoff) override {
		size_t subspace_size = query.size() / NUM_SUBSPACES;
		// Compute pq table
		std::array<std::array<float, NUM_CENTROIDS>, NUM_SUBSPACES> distance_table;
		for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace) {
			Eigen::VectorXf subquery =
					query.segment(cursubspace * subspace_size, subspace_size);
			for (size_t curcentroid = 0; curcentroid < NUM_CENTROIDS; ++curcentroid) {
				auto res =
						(sub_centroids[cur_vert][cursubspace][curcentroid] - subquery)
								.squaredNorm();
				distance_table[cursubspace][curcentroid] = res;
			}
		}
		// Use pq table to compute distances for to_filter_offsets
		for (size_t entry_index = 0; entry_index < to_filter_offsets.size();
				 ++entry_index) {
			float res = 0;
			for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace) {
				res += distance_table[cursubspace]
														 [codes[cur_vert][to_filter_offsets[entry_index]]
																	 [cursubspace]];
			}
			if (res < cutoff) {
				filtered_out.emplace_back(to_filter[entry_index]);
				distances.emplace_back(res);
			}
		}
	}
};

class product_quantizer : public quantizer {
	using fvec = vec<float>::Underlying;
	clusterer clust;

	static constexpr size_t NUM_SUBSPACES = 16;
	static constexpr size_t NUM_CENTROIDS = 16;

	using codes_t = std::array<uint8_t, NUM_SUBSPACES>;
	std::vector<std::vector<codes_t>> codes;
	std::vector<
			std::array<std::array<Eigen::VectorXf, NUM_CENTROIDS>, NUM_SUBSPACES>>
			sub_centroids;

	Eigen::VectorXf pad(const fvec& input) {
		size_t dimension = input.size();
		size_t padded_dimension =
				input.size() +
				(NUM_SUBSPACES - (input.size() % NUM_SUBSPACES)) % NUM_SUBSPACES;
		Eigen::VectorXf ret = Eigen::VectorXf::Map(input.data(), input.size());
		ret.conservativeResize(padded_dimension);
		ret.tail(padded_dimension - dimension).setZero();
		return ret;
	}

public:
	product_quantizer() = default;
	~product_quantizer() = default;

	virtual void build(const std::vector<fvec>& unquantized,
										 const std::vector<std::vector<size_t>>& adj) override {
		sub_centroids.resize(unquantized.size());
		codes.resize(unquantized.size());
		std::cout << "Running k-means on each neighbourhood" << std::endl;
		for (size_t base = 0; base < unquantized.size(); ++base) {
			if (base % 5000 == 0)
				std::cout << "base=" << base << std::endl;
			std::vector<std::vector<Eigen::VectorXf>> neighbours_per_sub(
					NUM_SUBSPACES);
			for (size_t neighbour_index : adj[base]) {
				auto padded_neighbour = pad(unquantized[neighbour_index]);
				size_t subspace_size = padded_neighbour.size() / NUM_SUBSPACES;
				for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES;
						 ++cursubspace) {
					neighbours_per_sub[cursubspace].emplace_back(padded_neighbour.segment(
							cursubspace * subspace_size, subspace_size));
				}
			}
			// sub dimension -> index(16) -> centroid for subdim
			codes[base].resize(adj[base].size());
			for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace) {
				std::vector<Eigen::VectorXf> current_centroids;
				std::vector<size_t> sub_labels;
				clust.k_means(neighbours_per_sub[cursubspace], NUM_CENTROIDS,
											sub_labels, current_centroids);
				for (size_t curcentroid = 0; curcentroid < NUM_CENTROIDS; ++curcentroid)
					sub_centroids[base][cursubspace][curcentroid] =
							current_centroids[curcentroid];
				for (size_t neighbour = 0; neighbour < adj[base].size(); ++neighbour)
					codes[base][neighbour][cursubspace] = uint8_t(sub_labels[neighbour]);
			}
		}
		std::cout << "Done running k-means on each neighbourhood" << std::endl;
	}
	virtual std::unique_ptr<quantized_scorer>
	generate_scorer(const fvec& query) override {
		auto padded_query = pad(query);
		return std::make_unique<product_quantizer_scorer>(padded_query, codes,
																											sub_centroids);
	}
};
