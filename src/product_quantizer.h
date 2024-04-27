#pragma once

#include "clusterer.h"
#include "quantizer.h"
#include <iostream>
#include <optional>

class product_quantizer_scorer : public quantized_scorer {
	using fvec = vec<float>::Underlying;
	fvec query;

	static constexpr size_t SUBSPACE_INDEX_SIZE = 4;
	static constexpr size_t NUM_SUBSPACES = 16;
	static constexpr size_t NUM_CENTROIDS = 16;

	// using codes_t = std::array<uint8_t, NUM_SUBSPACES>;
	//  using codes_t = uint64_t;

	// const std::vector<std::vector<codes_t>>& codes;
	// const std::vector<std::vector<uint8_t>>& codes_per_subspace;
	const std::vector<Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES>>&
			codes_per_subspace;

	using centroids_t =
			Eigen::Matrix<float, NUM_CENTROIDS, Eigen::Dynamic, Eigen::RowMajor>;
	const std::vector<centroids_t>& sub_centroids_compact;
	// const std::vector<std::array<std::array<Eigen::VectorXf, NUM_CENTROIDS>,
	//														 NUM_SUBSPACES>>& sub_centroids;

public:
	product_quantizer_scorer(
			const fvec& _query,
			// const std::vector<std::vector<codes_t>>& _codes,
			// const std::vector<std::vector<uint8_t>>& _codes_per_subspace,
			const std::vector<Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES>>&
					_codes_per_subspace,
			const std::vector<centroids_t>& _sub_centroids_compact)
			//,const std::vector<std::array<std::array<Eigen::VectorXf,
			// NUM_CENTROIDS>, 														 NUM_SUBSPACES>>&
			// _sub_centroids)
			: query(_query), codes_per_subspace(_codes_per_subspace),
				// codes(_codes),
				sub_centroids_compact(_sub_centroids_compact)
	//,sub_centroids(_sub_centroids)
	{}
	virtual ~product_quantizer_scorer() {}

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
		// std::array<std::array<float, NUM_SUBSPACES>, NUM_CENTROIDS>
		// distance_table;
		Eigen::Matrix<float, NUM_CENTROIDS, NUM_SUBSPACES, Eigen::RowMajor>
				distance_table =
						Eigen::Matrix<float, NUM_CENTROIDS, NUM_SUBSPACES,
													Eigen::RowMajor>::Zero(NUM_CENTROIDS, NUM_SUBSPACES);
		if (false) {
			for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace) {
				Eigen::VectorXf subquery =
						query.segment(cursubspace * subspace_size, subspace_size);
				for (size_t curcentroid = 0; curcentroid < NUM_CENTROIDS;
						 ++curcentroid) {
					// auto cur_sub_centroid_v1 =
					//		sub_centroids[cur_vert][cursubspace][curcentroid];
					// auto cur_sub_centroid_v2 =
					//		sub_centroids_compact[cur_vert]
					//				.row(curcentroid)
					//				.segment(cursubspace * subspace_size, subspace_size);
					//  std::cout << "cur_sub_centroid_v1=" <<
					//  cur_sub_centroid_v1.transpose()
					//					<< std::endl;
					//   std::cout << "cur_sub_centroid_v2=" << cur_sub_centroid_v2 <<
					//   std::endl;
					auto res = (sub_centroids_compact[cur_vert]
													.row(curcentroid)
													.segment(cursubspace * subspace_size, subspace_size)
													.transpose() -
											subquery)
												 .squaredNorm();
					// auto res =
					//		(sub_centroids[cur_vert][cursubspace][curcentroid] - subquery)
					//				.squaredNorm();
					//  distance_table[cursubspace][curcentroid] = res;
					// distance_table[curcentroid][cursubspace] = res;
					distance_table(curcentroid, cursubspace) = res;
				}
			}
		} else if (true) {
			auto reshaped_query = query.reshaped(subspace_size, NUM_SUBSPACES);
			for (size_t curcentroid = 0; curcentroid < NUM_CENTROIDS; ++curcentroid) {
				// Extract each centroid's subspace data into a matrix of shape
				// (NUM_SUBSPACES, subspace_size)
				auto centroid_subspaces = sub_centroids_compact[cur_vert]
																			.row(curcentroid)
																			.reshaped(subspace_size, NUM_SUBSPACES);
				//.transpose();

				// Compute the squared norm of the difference, summing over the columns
				// of each subspace
				auto norms =
						(centroid_subspaces - reshaped_query).colwise().squaredNorm();
				distance_table.row(curcentroid) = norms;
				//(centroid_subspaces - reshaped_query).colwise().squaredNorm();
			}
			// distances_table[curcentorid][cursubspace]
		} else {
			// There's a bug in here
			auto squared_diffs =
					(sub_centroids_compact[cur_vert].rowwise() - query.transpose())
							.array()
							.square();
			auto reshaped_squared_diffs =
					squared_diffs.reshaped(subspace_size, NUM_SUBSPACES * NUM_CENTROIDS);
			distance_table =
					reshaped_squared_diffs
							.colwise() // changing colwise to rowwise does not fix the bug
							.sum()
							.reshaped(NUM_CENTROIDS, NUM_SUBSPACES);

			//.reshaped(subspace_size, NUM_SUBSPACES);
		}
		// Use pq table to compute distances for to_filter_offsets
		if (false) {
			/*
			for (size_t entry_index = 0; entry_index < to_filter_offsets.size();
					 ++entry_index) {
				float res = 0;
				size_t entry_offset = to_filter_offsets[entry_index];
				const auto& cur_codes = codes[cur_vert][entry_offset];
				for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES;
						 ++cursubspace) {
					// res += distance_table[cur_codes[cursubspace]][cursubspace];
					// res += distance_table(cur_codes[cursubspace], cursubspace);
					res += distance_table(
							(cur_codes >> (cursubspace * SUBSPACE_INDEX_SIZE)) & 0b1111,
							cursubspace);
				}
				if (res < cutoff) {
					filtered_out.emplace_back(to_filter[entry_index]);
					distances.emplace_back(res);
				}
			}
			*/
		} else if (false) {
			/*
			Eigen::Matrix<float, NUM_CENTROIDS, NUM_SUBSPACES, Eigen::ColMajor>
					distance_table_colmaj = distance_table;
			// Eigen::VectorX<uint64_t> codes_table(to_filter_offsets.size());
			Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES> codes_table =
					Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES>::Zero(
							to_filter_offsets.size(), NUM_SUBSPACES);
			for (size_t entry_index = 0; entry_index < to_filter_offsets.size();
					 ++entry_index) {
				size_t entry_offset = to_filter_offsets[entry_index];
				const auto& cur_codes = codes[cur_vert][entry_offset];
				for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES;
						 ++cursubspace) {
					codes_table(entry_index, cursubspace) =
							(cur_codes >> (cursubspace * SUBSPACE_INDEX_SIZE)) & 0b1111;
				}
			}
			Eigen::VectorXf distance_vec =
					Eigen::VectorXf::Zero(to_filter_offsets.size());
			for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace) {
				distance_vec += distance_table_colmaj.col(cursubspace)(
						codes_table.col(cursubspace));
			}
			for (size_t entry_index = 0; entry_index < to_filter_offsets.size();
					 ++entry_index) {
				auto res = distance_vec[entry_index];
				if (res < cutoff) {
					filtered_out.emplace_back(to_filter[entry_index]);
					distances.emplace_back(res);
				}
			}
			*/
		} else {
			Eigen::Matrix<float, NUM_CENTROIDS, NUM_SUBSPACES, Eigen::ColMajor>
					distance_table_colmaj = distance_table;
			size_t adj_size = codes_per_subspace[cur_vert].size() / NUM_SUBSPACES;

			// Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES,
			//												 Eigen::ColMajor>>
			// Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES,
			//							Eigen::ColMajor>
			//		codes_table; //(codes_per_subspace[cur_vert].data(), adj_size,
			// NUM_SUBSPACES);
			Eigen::VectorXf distance_vec = Eigen::VectorXf::Zero(adj_size);
			for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace) {
				distance_vec += distance_table_colmaj.col(cursubspace)(
						codes_per_subspace[cur_vert].col(cursubspace));
			}
			for (size_t entry_index = 0; entry_index < to_filter_offsets.size();
					 ++entry_index) {
				size_t entry_offset = to_filter_offsets[entry_index];
				auto res = distance_vec[entry_offset];
				if (res < cutoff) {
					filtered_out.emplace_back(to_filter[entry_index]);
					distances.emplace_back(res);
				}
			}
		}
	}
};

class product_quantizer : public quantizer {
	using fvec = vec<float>::Underlying;

	static constexpr size_t SUBSPACE_INDEX_SIZE = 4;
	static constexpr size_t NUM_SUBSPACES = 16;
	static constexpr size_t NUM_CENTROIDS = 16;

	// using codes_t = std::array<uint8_t, NUM_SUBSPACES>;
	// using codes_t = uint64_t;
	// std::vector<std::vector<codes_t>> codes;
	// std::vector<std::vector<uint8_t>> codes_per_subspace;
	std::vector<Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES>>
			codes_per_subspace;
	using centroids_t =
			Eigen::Matrix<float, NUM_CENTROIDS, Eigen::Dynamic, Eigen::RowMajor>;
	std::vector<centroids_t> sub_centroids_compact;
	// std::vector<
	//		std::array<std::array<Eigen::VectorXf, NUM_CENTROIDS>, NUM_SUBSPACES>>
	//		sub_centroids;

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
	virtual ~product_quantizer() = default;

	virtual void build(const std::vector<fvec>& unquantized,
										 const std::vector<std::vector<size_t>>& adj) override {
		size_t dimension = unquantized[0].size();
		size_t subspace_size = dimension / NUM_SUBSPACES;
		std::cout << "Initializing memory for centroids" << std::endl;
		sub_centroids_compact = std::vector<centroids_t>(
				unquantized.size(), centroids_t::Zero(NUM_CENTROIDS, dimension));
		// sub_centroids.resize(unquantized.size());
		codes_per_subspace.resize(unquantized.size());
		// codes.resize(unquantized.size());
		for (size_t base = 0; base < unquantized.size(); ++base) {
			// codes[base].resize(adj[base].size());
			// codes[base] = std::vector<codes_t>(adj[base].size(), 0);
			// codes_per_subspace[base].resize(adj[base].size() * NUM_SUBSPACES);
			codes_per_subspace[base] =
					Eigen::Matrix<uint8_t, Eigen::Dynamic, NUM_SUBSPACES>(
							adj[base].size(), NUM_SUBSPACES);
		}
		std::cout << "Running k-means on each neighbourhood" << std::endl;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::vector<std::vector<Eigen::VectorXf>> neighbours_per_sub(NUM_SUBSPACES);
		for (size_t base = 0; base < unquantized.size(); ++base) {
			if (base % 5000 == 0)
				std::cout << "base=" << base << std::endl;
			for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace)
				neighbours_per_sub[cursubspace].clear();
			for (size_t neighbour_index : adj[base]) {
				auto padded_neighbour = pad(unquantized[neighbour_index]);
				size_t subspace_size = padded_neighbour.size() / NUM_SUBSPACES;
				for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES;
						 ++cursubspace) {
					neighbours_per_sub[cursubspace].emplace_back(padded_neighbour.segment(
							cursubspace * subspace_size, subspace_size));
				}
			}
			for (size_t cursubspace = 0; cursubspace < NUM_SUBSPACES; ++cursubspace) {
				std::vector<Eigen::VectorXf> current_centroids;
				std::vector<size_t> sub_labels;
				clusterer clust(gen);
				clust.k_means(neighbours_per_sub[cursubspace], NUM_CENTROIDS,
											sub_labels, current_centroids);
				for (size_t curcentroid = 0; curcentroid < NUM_CENTROIDS;
						 ++curcentroid) {
					if (subspace_size != size_t(current_centroids[curcentroid].size()))
						throw std::runtime_error("Subdimension-size mismatch");
					if (size_t(sub_centroids_compact[base].row(curcentroid).size()) !=
							dimension) {
						throw std::runtime_error("Dimensions were mixed up");
					}
					sub_centroids_compact[base]
							.row(curcentroid)
							.segment(cursubspace * subspace_size, subspace_size) =
							current_centroids[curcentroid];
					// sub_centroids[base][cursubspace][curcentroid] =
					//		current_centroids[curcentroid];
				}
				for (size_t neighbour = 0; neighbour < adj[base].size(); ++neighbour) {
					if (sub_labels[neighbour] >= 16) {
						throw std::runtime_error("Sub_label too big");
					}
					codes_per_subspace[base](neighbour, cursubspace) =
							uint8_t(sub_labels[neighbour]);
					// codes_per_subspace[base][neighbour + cursubspace *
					// adj[base].size()] = 		uint8_t(sub_labels[neighbour]);
					//  codes[base][neighbour] |= ((0b1111 & sub_labels[neighbour])
					//													 << (cursubspace * SUBSPACE_INDEX_SIZE));
					//[cursubspace] = uint8_t(sub_labels[neighbour]);
					//((codes[base][neighbour] >> (cursubspace * 4)) & 0b1111)
				}
				/*
				for (size_t neighbour = 0; neighbour < adj[base].size(); ++neighbour) {
					if (((codes[base][neighbour] >> (cursubspace * SUBSPACE_INDEX_SIZE)) &
							 0b1111) != sub_labels[neighbour]) {
						throw std::runtime_error("The labels don't convert properly");
					}
				}
				*/
			}
		}
		std::cout << "Done running k-means on each neighbourhood" << std::endl;
	}
	virtual std::unique_ptr<quantized_scorer>
	generate_scorer(const fvec& query) override {
		auto padded_query = pad(query);
		return std::make_unique<product_quantizer_scorer>(
				padded_query, codes_per_subspace,
				sub_centroids_compact); //, sub_centroids);
	}
};
