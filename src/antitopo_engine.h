#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
// #include "product_quantizer.h"
#include "distance.h"
#include "quantizer.h"
#include "robin_hood.h"
#include "topk_t.h"
#include "vec.h"

namespace {
template <typename A, typename B> auto dist2(const A& a, const B& b) {
#ifdef DIM
	return distance_compare_avx512f_f16_batch128(a.data(), b.data(), DIM);
//  return distance_compare_avx512f_f16(a.data(), b.data(), DIM);
//   return distance_compare_avx512f_f16_prefetched(a.data(), b.data(), DIM);
#else
	return (a - b).squaredNorm();
#endif
}
// template <typename Au, typename A, typename B>
// auto dist2_compressed(const Au& a_unswizzled, const A& a, const B& b) {
template <typename A, typename B>
auto dist2_compressed(const A& a, const B& b) {
	// b is uint8_t eigenvec
	// turbo lvq method:
	// Eigen::Map<const Eigen::VectorX<uint64_t>> b_chunked(
	//		reinterpret_cast<const uint64_t*>(b.data()), b.size());
	// auto masked_chunks = b_chunked.array() & uint64_t(0xff);
	//  auto shifted_b = b.array() >> 8;
	auto b_chunked = reinterpret_cast<const uint32_t*>(b.data());
	// take 16 chunks at a time (512 bits, 4 dims per chunk, swizzled), 64 dims
	__m512i sum_squared_diff = _mm512_setzero_si512();
	__m512i mask = _mm512_set1_epi32(0xff);
	std::vector<uint8_t> b_in_order;
	for (size_t i = 0; i < a.size(); i += 64) {
		__m512i cur_b = _mm512_loadu_si512((__m512i*)&b_chunked[i / 4]);
		// a.data()[i...(i+63)] contains 16 32-bit integers (64 bytes total=512bit)
		for (size_t j = 0; j < 64; j += 16) {
			// handle 16 32-bit integers at a time from a.data()
			__m512i cur_a = _mm512_loadu_si512((__m512i*)&a.data()[i + j]);
			__m512i cur_b_masked = _mm512_and_si512(cur_b, mask);
			__m512i diff = _mm512_sub_epi32(cur_a, cur_b_masked);
			__m512i sqr_diff = _mm512_mullo_epi32(diff, diff);
			sum_squared_diff = _mm512_add_epi32(sum_squared_diff, sqr_diff);
			cur_b = _mm512_srli_epi32(cur_b, 8);
		}
	}
	/*
	{
		uint32_t res = 0;
		std::cout << "b_manual_order=\n";
		for (size_t i = 0; i < a.size(); i += 64) {
			std::vector<uint32_t> cur_b(16);
			for (size_t k = 0; k < 16; ++k)
				cur_b[k] = b_chunked[k];
			// a.data()[i...(i+63)] contains 16 32-bit integers (64 bytes
			// total=512bit)
			for (size_t j = 0; j < 64; j += 16) {
				std::vector<uint32_t> cur_a(16);
				for (size_t k = 0; k < 16; ++k)
					cur_a[k] = a[i + j + k];
				std::vector<uint32_t> cur_b_masked(cur_b);
				for (auto& to_mask : cur_b_masked) {
					to_mask &= 0xff;
					std::cout << to_mask << '\t';
				}
				for (size_t k = 0; k < 16; ++k) {
					auto diff = (cur_a[k] - cur_b_masked[k]);
					res += diff * diff;
				}
				for (auto& to_shift : cur_b)
					to_shift >>= 8;
			}
			std::cout << '\n';
		}
		std::cout << "manual order res=" << res << std::endl;
	}
	auto ret = _mm512_reduce_add_epi32(sum_squared_diff);
	auto ret1 = (a_unswizzled - b.template cast<float>()).squaredNorm();
	if (ret - ret1 > 1e-1) {
		std::cout << "a=\n" << a.transpose() << std::endl;
		std::cout << "a_unswizzled=\n" << a_unswizzled.transpose() << std::endl;
		std::cout << "b=\n" << b.transpose() << std::endl;
		std::cout << "a_unswizzled.reshaped=\n"
							<< a_unswizzled.reshaped(64, 2).transpose() << std::endl;
		std::cout << "b.reshaped=\n" << b.reshaped(64, 2).transpose() << std::endl;
		std::cout << "a.reshaped=\n"
							<< a.reshaped(16, 128 / 16).transpose() << std::endl;
		std::cout << "b_chunked size="
							<< b.size() * sizeof(uint8_t) / sizeof(uint32_t) << std::endl;
		std::cout << "ret_compressed=" << ret << std::endl;
		std::cout << "ret_uncompressed=" << ret1 << std::endl;
		throw std::runtime_error(
				"Mismatch between compressed and uncomrpessed distance");
	}
	*/
	return _mm512_reduce_add_epi32(sum_squared_diff);

	// return (a - b.template cast<float>()).squaredNorm();
}
} // namespace

struct antitopo_engine_query_config {
	size_t ef_search_mult;
	std::optional<size_t> ef_search;
	antitopo_engine_query_config(size_t _ef_search_mult,
															 std::optional<size_t> _ef_search = std::nullopt)
			: ef_search_mult(_ef_search_mult), ef_search(_ef_search) {}
};

struct antitopo_engine_config : public antitopo_engine_query_config {
	using antitopo_engine_query_config::ef_search;
	using antitopo_engine_query_config::ef_search_mult;
	size_t M;
	size_t M0;
	size_t ef_construction;
	size_t ortho_count;
	float ortho_factor;
	float ortho_bias;
	size_t prune_overflow;
	bool use_compression;
	bool use_largest_direction_filtering;
	std::string index_filename;
	bool read_index;
	bool write_index;
	antitopo_engine_config(size_t _M, size_t _M0, size_t _ef_search_mult,
												 size_t _ef_construction, size_t _ortho_count,
												 float _ortho_factor, float _ortho_bias,
												 size_t _prune_overflow, bool _use_compression = false,
												 bool _use_largest_direction_filtering = false,
												 std::string _index_filename = "",
												 bool _read_index = false, bool _write_index = false)
			: antitopo_engine_query_config(_ef_search_mult), M(_M), M0(_M0),
				ef_construction(_ef_construction), ortho_count(_ortho_count),
				ortho_factor(_ortho_factor), ortho_bias(_ortho_bias),
				prune_overflow(_prune_overflow), use_compression(_use_compression),
				use_largest_direction_filtering(_use_largest_direction_filtering),
				index_filename(_index_filename), read_index(_read_index),
				write_index(_write_index) {}
};

template <typename T>
struct antitopo_engine : public ann_engine<T, antitopo_engine<T>> {
	using fvec = typename vec<T>::Underlying;
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	std::optional<size_t> ef_search;
	size_t ef_construction;
	size_t ortho_count;
	float ortho_factor;
	float ortho_bias;
	size_t prune_overflow;
	bool use_compression;
	bool use_largest_direction_filtering;
	std::string index_filename;
	bool read_index;
	bool write_index;
	size_t max_layer;
#ifdef RECORD_STATS
	size_t num_distcomps = 0;
	size_t num_distcomps_compressed = 0;
#endif
	void constructor_helper() {
		quant = &quant_impl;
		// quant = std::make_unique<product_quantizer>();
		// quant = std::make_unique<quantizer_simple<float>>();
		std::filesystem::path directory =
				std::filesystem::path(index_filename).parent_path();
		if (!std::filesystem::exists(directory)) {
			std::filesystem::create_directories(directory);
			std::printf("Directory %s created\n", directory.c_str());
		}
		if (read_index) {
			if (std::filesystem::exists(index_filename)) {
				std::printf("Index file %s exists, disabling write\n",
										index_filename.c_str());
				write_index = false;
			} else {
				std::printf("Index file %s does not exist, disabling read\n",
										index_filename.c_str());
				read_index = false;
			}
		}
	}
	antitopo_engine(size_t _M, size_t _ef_construction, size_t _ortho_count,
									float _ortho_factor, float _ortho_bias,
									size_t _prune_overflow)
			: rd(), gen(0), distribution(0, 1), M(_M), M0(2 * _M), ef_search_mult(1),
				ef_search(std::nullopt), ef_construction(_ef_construction),
				ortho_count(_ortho_count), ortho_factor(_ortho_factor),
				ortho_bias(_ortho_bias), prune_overflow(_prune_overflow),
				use_compression(false), use_largest_direction_filtering(false),
				index_filename(""), read_index(false), write_index(false),
				max_layer(0) {
		constructor_helper();
	}
	antitopo_engine(antitopo_engine_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult), ef_search(conf.ef_search),
				ef_construction(conf.ef_construction), ortho_count(conf.ortho_count),
				ortho_factor(conf.ortho_factor), ortho_bias(conf.ortho_bias),
				prune_overflow(conf.prune_overflow),
				use_compression(conf.use_compression),
				use_largest_direction_filtering(conf.use_largest_direction_filtering),
				index_filename(conf.index_filename), read_index(conf.read_index),
				write_index(conf.write_index), max_layer(0) {
		constructor_helper();
	}
	using config = antitopo_engine_config;
	using query_config = antitopo_engine_query_config;
	void serialize(std::ostream& out) const;
	void deserialize(std::istream& in);
	void set_ef_search(size_t _ef_search) { ef_search = _ef_search; }
	std::vector<fvec> all_entries;
	using compressed_t = uint8_t;
	quantizer_simple<compressed_t> quant_impl;
	// std::unique_ptr<quantizer> quant;
	quantizer* quant;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<std::vector<std::vector<std::pair<T, size_t>>>>
			hadj_flat_with_lengths; // vector -> layer -> edges with lengths
	void _store_vector(const vec<T>& v0, bool silent = false);
	void _build();
	std::vector<char> visited; // booleans
	std::vector<size_t> visited_recent;
	void update_edges(size_t layer, size_t from) {
		hadj_flat[from][layer].clear();
		hadj_flat[from][layer].reserve(hadj_flat_with_lengths[from][layer].size());
		for (auto& [_, val] : hadj_flat_with_lengths[from][layer]) {
			hadj_flat[from][layer].emplace_back(val);
		}
		if (layer == 0)
			hadj_bottom[from] = hadj_flat[from][layer];
	}
	void add_new_edges(size_t layer, size_t from) {
		for (size_t i = hadj_flat[from][layer].size();
				 i < hadj_flat_with_lengths[from][layer].size(); ++i) {
			hadj_flat[from][layer].emplace_back(
					hadj_flat_with_lengths[from][layer][i].second);
			if (layer == 0)
				hadj_bottom[from].emplace_back(
						hadj_flat_with_lengths[from][layer][i].second);
		}
	}
	void prune_edges(size_t layer, size_t from, bool lazy);
	template <bool use_bottomlayer, bool use_compressed, bool use_ortho>
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const vec<T>& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k,
									 const std::vector<size_t>& ortho_points);
	std::vector<std::pair<T, size_t>>
	query_k_bottom_compressed(const vec<T>& q,
														const std::vector<size_t>& entry_points, size_t k);

	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	std::vector<std::pair<T, size_t>> query_k_combined(const vec<T>& v, size_t k);
	const std::string _name() { return "Anti-Topo Engine+"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		add_param(pl, ortho_count);
		add_param(pl, ortho_factor);
		add_param(pl, ortho_bias);
		add_param(pl, prune_overflow);
		add_param(pl, use_compression);
		add_param(pl, use_largest_direction_filtering);
#ifdef RECORD_STATS
		add_param(pl, num_distcomps);
		add_param(pl, num_distcomps_compressed);
#endif
		return pl;
	}
};

template <typename T>
void antitopo_engine<T>::prune_edges(size_t layer, size_t from, bool lazy) {
	auto& to = hadj_flat_with_lengths[from][layer];

	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

	if (lazy && to.size() <= edge_count_mult) {
		add_new_edges(layer, from);
		// update_edges(layer, from);
		return;
	}

	sort(to.begin(), to.end());
	std::set<std::pair<T, size_t>> candidates(to.begin(), to.end());
	std::vector<std::pair<T, size_t>> ret;
	// std::vector<fvec> normalized_ret;
	// auto origin = all_entries[from];
	// std::unordered_set<size_t> taken;

	float prune_score = std::numeric_limits<float>::max();
	auto score =
			[&](const std::pair<T, size_t>& data_index_with_length) constexpr {
				T basic_dist = data_index_with_length.first;
				T res = basic_dist;
				size_t data_index = data_index_with_length.second;
				size_t leniency = prune_overflow + 1;
				for (auto [_, prev_index] : ret) {
					T co_dist = dist2(all_entries[prev_index], all_entries[data_index]);
					if (co_dist < basic_dist) {
						res += ortho_factor * (basic_dist - co_dist) + ortho_bias;
						if (--leniency == 0) {
							return prune_score;
						}
					}
				}
				return res;
			};
	while (ret.size() < edge_count_mult && !candidates.empty()) {
		auto it = std::ranges::min_element(candidates, {}, score);
		if (score(*it) == prune_score)
			break;
		ret.emplace_back(*it);
		candidates.erase(it);
	}

	/*
	for (const auto& md : to) {
		bool choose = true;
		auto v1 = (all_entries[md.second] - origin).normalized();
		size_t max_i = 0;
		float max_val = 0;
		if (use_largest_direction_filtering) {
			for (size_t i = 0; i < size_t(v1.size()); ++i) {
				float cur_val = v1[i];
				if (cur_val > max_val) {
					max_i = i;
					max_val = cur_val;
				} else if (-cur_val > max_val) {
					max_i = i + v1.size();
					max_val = -cur_val;
				}
			}
			if (taken.contains(max_i)) {
				choose = false;
			}
		} else {
			for (size_t next_i = 0; next_i < normalized_ret.size(); ++next_i) {
				const auto& v2 = normalized_ret[next_i];
				// if (dist2(all_entries[md.second], all_entries[ret[next_i].second]) <
				//		dist2(all_entries[md.second], origin)) {
				if (dist2(v1, v2) <= 1.0) {
					choose = false;
					break;
				}
			}
		}
		if (choose) {
			taken.insert(max_i);
			ret.emplace_back(md);
			normalized_ret.emplace_back(v1);
			if (ret.size() >= edge_count_mult)
				break;
		}
	}
	*/
	to = ret;
	to.shrink_to_fit();
	update_edges(layer, from);
}

template <typename T>
void antitopo_engine<T>::_store_vector(const vec<T>& v0, bool silent) {
	if (read_index)
		return;

	auto v = v0.internal;
	size_t v_index = all_entries.size();
	all_entries.emplace_back(v);

	if (v_index % 1000 == 0 && !silent) {
		std::cout << "Storing v_index=" << v_index << std::endl;
	}

	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));
	// size_t new_max_layer = 0;

	hadj_flat_with_lengths.emplace_back();
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat_with_lengths[v_index].emplace_back();
	}

	// get kNN for each layer
	std::vector<std::vector<std::pair<T, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		{
			std::vector<size_t> entry_points;
			for (size_t i = 0; i < ortho_count; ++i) {
				const auto& q = v;
				size_t entry_point = starting_vertex;
#ifdef RECORD_STATS
				++num_distcomps;
#endif
				auto score = [&](size_t data_index) constexpr {
					T basic_dist = dist2(all_entries[data_index], q);
					T res = basic_dist;
					for (size_t prev_index : entry_points) {
						T co_dist = dist2(all_entries[prev_index], all_entries[data_index]);
						if (co_dist < basic_dist)
							res += ortho_factor * (basic_dist - co_dist) + ortho_bias;
					}
					return res;
				};
				T ep_dist = score(entry_point);
				for (size_t layer = max_layer - 1; layer > new_max_layer; --layer) {
					bool changed = true;
					while (changed) {
						changed = false;
						for (auto& neighbour : hadj_flat[entry_point][layer]) {
							_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
#ifdef RECORD_STATS
							++num_distcomps;
#endif
							T neighbour_dist = score(neighbour);
							if (neighbour_dist < ep_dist) {
								entry_point = neighbour;
								ep_dist = neighbour_dist;
								changed = true;
							}
						}
					}
				}
				bool dupe = false;
				for (size_t existing_ep : entry_points)
					if (existing_ep == entry_point) {
						dupe = true;
						break;
					}
				if (!dupe)
					entry_points.emplace_back(entry_point);
			}
			cur = entry_points;
		}
		for (int layer = std::min(new_max_layer, max_layer - 1); layer >= 0;
				 --layer) {
			std::vector<std::vector<std::pair<T, size_t>>> result_lists;
			std::vector<size_t> new_cur;
			std::vector<size_t> seeds = cur;
			std::set<size_t> seeds_set(seeds.begin(), seeds.end());
			auto populate_seeds = [&]() {
				for (const auto& [_, new_point] : result_lists.back()) {
					if (!seeds_set.contains(new_point)) {
						seeds.emplace_back(new_point);
						seeds_set.insert(new_point);
					}
				}
			};
			for (size_t i = 0; i < ortho_count; ++i) {
				if (layer == 0) {
					result_lists.emplace_back(query_k_at_layer<true, false, true>(
							v0, layer, seeds, ef_construction, new_cur));
				} else {
					result_lists.emplace_back(query_k_at_layer<false, false, true>(
							v0, layer, seeds, ef_construction, new_cur));
				}
				populate_seeds();
				bool dupe = false;
				size_t candidate = result_lists.back()[0].second;
				for (size_t already : new_cur)
					if (already == candidate) {
						dupe = true;
						break;
					}
				if (!dupe)
					new_cur.emplace_back(candidate);
			}
			std::set<std::pair<T, size_t>> results_combined_set;
			for (const auto& result_list : result_lists) {
				for (const auto& result : result_list)
					results_combined_set.emplace(result);
			}
			std::vector<std::pair<T, size_t>> results_combined;
			for (const auto& p : results_combined_set)
				results_combined.emplace_back(p);
			kNN_per_layer.emplace_back(results_combined);
			cur = new_cur;
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}

	hadj_flat.emplace_back();
	hadj_bottom.emplace_back();
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat[v_index].emplace_back();
	}

	// add the found edges to the graph
	for (size_t layer = 0; layer < std::min(max_layer, new_max_layer + 1);
			 ++layer) {
		hadj_flat_with_lengths[v_index][layer] = kNN_per_layer[layer];
		prune_edges(layer, v_index, false);
		//  add bidirectional connections, prune if necessary
		for (auto& md : hadj_flat_with_lengths[v_index][layer]) {
			bool edge_exists = false;
			for (auto& md_other : hadj_flat_with_lengths[md.second][layer]) {
				if (md_other.second == v_index) {
					edge_exists = true;
					break;
				}
			}
			if (!edge_exists) {
				hadj_flat_with_lengths[md.second][layer].emplace_back(md.first,
																															v_index);
				prune_edges(layer, md.second, true);
			}
		}
	}

	// add new layers if necessary
	while (new_max_layer >= max_layer) {
		++max_layer;
		starting_vertex = v_index;
	}

	visited.emplace_back();
}

template <typename T> void antitopo_engine<T>::_build() {
	// serialize
	if (write_index) {
		std::ofstream out(index_filename, std::ios::binary);
		serialize(out);
		out.close();
		std::cout << "Wrote index to " << index_filename << std::endl;
	}
	// deserialize
	if (read_index) {
		std::ifstream in(index_filename, std::ios::binary);
		deserialize(in);
		in.close();
		std::cout << "Read index from " << index_filename << std::endl;
	}

	assert(all_entries.size() > 0);

	if (use_compression)
		quant->build(all_entries, hadj_bottom);

#ifdef RECORD_STATS
	// reset before queries
	num_distcomps = 0;
	num_distcomps_compressed = 0;
#endif
}

template <typename T>
template <bool use_bottomlayer, bool use_compressed, bool use_ortho>
std::vector<std::pair<T, size_t>> antitopo_engine<T>::query_k_at_layer(
		const vec<T>& q0, size_t layer, const std::vector<size_t>& entry_points,
		size_t k, const std::vector<size_t>& ortho_points) {
	using measured_data = std::pair<T, size_t>;
	const auto& q = q0.internal;
	size_t dimension = q.size();

	auto get_vertex = [&](const size_t& index) constexpr -> std::vector<size_t>& {
		if constexpr (use_bottomlayer) {
			return hadj_bottom[index];
		} else {
			return hadj_flat[index][layer];
		}
	};
	auto get_data = [&](const size_t& data_index) constexpr -> auto& {
		return all_entries[data_index];
	};
	auto score = [&](size_t data_index) constexpr {
		if constexpr (use_ortho) {
			T basic_dist = dist2(all_entries[data_index], q);
			T res = basic_dist;
			for (size_t prev_index : ortho_points) {
				T co_dist = dist2(all_entries[prev_index], all_entries[data_index]);
				if (co_dist < basic_dist)
					res += ortho_factor * (basic_dist - co_dist) + ortho_bias;
			}
			return res;
		} else {
#ifdef RECORD_STATS
			++num_distcomps;
#endif
			return dist2(q, get_data(data_index));
		}
	};
	auto scorer = [&]() constexpr {
		if constexpr (use_compressed)
			return quant->generate_scorer(q);
		else
			return []() {};
	}();

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points) {
		entry_points_with_dist.emplace_back(score(entry_point), entry_point);
	}

	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points_with_dist.begin(), entry_points_with_dist.end(),
								 best_elem);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points_with_dist.begin(), entry_points_with_dist.end(),
							worst_elem);
	while (nearest.size() > k)
		nearest.pop();
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest_big(entry_points_with_dist.begin(), entry_points_with_dist.end(),
									worst_elem);
	bool live_reranking = false;
	size_t big_factor = 3;
	if (!live_reranking)
		big_factor = 1;
	if constexpr (use_compressed) {
		while (nearest_big.size() > big_factor * k)
			nearest_big.pop();
	}

	for (auto& entry_point : entry_points) {
		visited[entry_point] = true;
		visited_recent.emplace_back(entry_point);
	}
	std::vector<measured_data> candidates_backup;
	auto clean_candidates = [&]() constexpr {
		return;
		candidates_backup.resize(k);
		if (candidates.size() > k * 4) {
			while (candidates_backup.size() < k) {
				candidates_backup.emplace_back(candidates.top());
				candidates.pop();
			}
			// candidates.clear();
			candidates =
					std::priority_queue<measured_data, std::vector<measured_data>,
															decltype(best_elem)>(
							candidates_backup.begin(), candidates_backup.end(), best_elem);
			candidates_backup.clear();
		}
	};

	std::vector<size_t> neighbour_list, neighbour_list_unfiltered,
			neighbour_list_unfiltered_offsets;
	std::vector<T> distances;
	size_t iter = 0;
	while (!candidates.empty()) {
		++iter;
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		neighbour_list.clear();
		neighbour_list_unfiltered.clear();
		neighbour_list_unfiltered_offsets.clear();
		size_t neighbour_index = 0;
		for (size_t neighbour : get_vertex(cur.second)) {
			if (!visited[neighbour]) {
				if constexpr (use_compressed) {
					neighbour_list_unfiltered.emplace_back(neighbour);
					neighbour_list_unfiltered_offsets.emplace_back(neighbour_index);
				} else {
					neighbour_list.emplace_back(neighbour);
				}
				visited[neighbour] = true;
				visited_recent.emplace_back(neighbour);
			}
			++neighbour_index;
		}
		/*
		if (use_compressed && !live_reranking) {
			if constexpr (use_compressed) {
				constexpr size_t in_advance = 4;
				constexpr size_t in_advance_extra = 2;
				auto do_loop_prefetch = [&](size_t i) constexpr {
					scorer->prefetch(i);
				};
				for (size_t next_i_pre = 0;
						 next_i_pre <
						 std::min(in_advance, neighbour_list_unfiltered.size());
						 ++next_i_pre) {
					do_loop_prefetch(next_i_pre);
				}
				auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
														 size_t next_i) constexpr {
					if constexpr (inAdvanceIterExtra) {
						_mm_prefetch(&neighbour_list_unfiltered[next_i + in_advance +
																										in_advance_extra],
												 _MM_HINT_T0);
					}
					if constexpr (inAdvanceIter) {
						do_loop_prefetch(next_i + in_advance);
					}
					const auto& next = neighbour_list_unfiltered[next_i];
					float d_next = score(next);
					if (nearest.size() < k || d_next < nearest.top().first) {
						candidates.emplace(d_next, next);
						nearest.emplace(d_next, next);
						if (nearest.size() > k)
							nearest.pop();
					}
				};
				size_t next_i = 0;
				for (; next_i + in_advance + in_advance_extra <
							 neighbour_list_unfiltered.size();
						 ++next_i) {
					loop_iter.template operator()<true, true>(next_i);
				}
				for (; next_i + in_advance < neighbour_list_unfiltered.size();
						 ++next_i) {
					loop_iter.template operator()<true, false>(next_i);
				}
				for (; next_i < neighbour_list_unfiltered.size(); ++next_i) {
					loop_iter.template operator()<false, false>(next_i);
				}
			}
		} else
			*/
		if constexpr (use_compressed) {
			// if (nearest_big.size() < big_factor * k) {
			if (live_reranking && nearest_big.size() < big_factor * k) {
				// std::cout << "Skipping compressed computations, nearest_big.size()="
				//					<< nearest_big.size() << "(nearest.size()=" <<
				// nearest.size()
				//					<< "/k=" << k << "),";
				// std::cout << "big_factor=" << big_factor << ",iter=" << iter
				//					<< std::endl;
				neighbour_list = neighbour_list_unfiltered;
			} else {
				float cutoff = nearest_big.size() < big_factor * k
													 ? std::numeric_limits<T>::max()
													 : nearest_big.top().first;

#ifdef RECORD_STATS
				num_distcomps_compressed += neighbour_list_unfiltered.size();
#endif
				distances.clear();
				scorer->filter_by_score(cur.second, neighbour_list_unfiltered,
																neighbour_list_unfiltered_offsets,
																neighbour_list, distances, cutoff);
			}
		}
		if (use_compressed && !live_reranking) {
			for (size_t i = 0; i < neighbour_list.size(); ++i) {
				const size_t& next = neighbour_list[i];
				T d_next = distances[i];
				if (nearest.size() < k || d_next < nearest.top().first) {
					candidates.emplace(d_next, next);
					nearest.emplace(d_next, next);
					if (nearest.size() > k)
						nearest.pop();
				}
			}
		} else if constexpr (!use_compressed) {
			constexpr size_t in_advance = 4;
			constexpr size_t in_advance_extra = 2;
			auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
				for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
					_mm_prefetch(((char*)&get_data(neighbour_list[i])) + mult * 64,
											 _MM_HINT_T0);
#else
				for (size_t mult = 0; mult < dimension * sizeof(T) / 64; ++mult)
					_mm_prefetch(((char*)&get_data(neighbour_list[i])) + mult * 64,
											 _MM_HINT_T0);
#endif
			};
			for (size_t next_i_pre = 0;
					 next_i_pre < std::min(in_advance, neighbour_list.size());
					 ++next_i_pre) {
				do_loop_prefetch(next_i_pre);
			}
			auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
													 size_t next_i) constexpr {
				if constexpr (inAdvanceIterExtra) {
					_mm_prefetch(&neighbour_list[next_i + in_advance + in_advance_extra],
											 _MM_HINT_T0);
				}
				if constexpr (inAdvanceIter) {
					do_loop_prefetch(next_i + in_advance);
				}
				const auto& next = neighbour_list[next_i];
				T d_next = score(next);
				// if (use_compressed && !live_reranking) {
				// d_next = distances[next_i];
				//  std::cout << "d_next=" << d_next << ",";
				//  T compressed_dist = 0;
				//  if (distances.size() > next_i)
				//  compressed_dist = distances[next_i];
				//  std::cout << "d_next(compressed)=" << compressed_dist << std::endl;
				//   d_next = compressed_dist;
				//} else {
				// d_next = score(next);
				//}
				if (nearest.size() < k || d_next < nearest.top().first) {
					candidates.emplace(d_next, next);
					nearest.emplace(d_next, next);
					if (nearest.size() > k)
						nearest.pop();
					if constexpr (use_compressed) {
						nearest_big.emplace(d_next, next);
						if (nearest_big.size() > big_factor * k)
							nearest_big.pop();
					}
				}
			};
			size_t next_i = 0;
			for (; next_i + in_advance + in_advance_extra < neighbour_list.size();
					 ++next_i) {
				loop_iter.template operator()<true, true>(next_i);
			}
			for (; next_i + in_advance < neighbour_list.size(); ++next_i) {
				loop_iter.template operator()<true, false>(next_i);
			}
			for (; next_i < neighbour_list.size(); ++next_i) {
				loop_iter.template operator()<false, false>(next_i);
			}
		}
		// clean_candidates();
	}
	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<measured_data> ret;
	std::sort(ret.begin(), ret.end());
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	/*
	if constexpr (use_compressed) {
		for (auto& [d, data_index] : ret) {
			d = score(data_index);
		}
	}
	*/
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<std::pair<T, size_t>> antitopo_engine<T>::query_k_bottom_compressed(
		const vec<T>& q0, const std::vector<size_t>& entry_points, size_t k) {
	using measured_data = std::pair<T, size_t>;
	const auto& q = q0.internal;
#ifndef DIM
	size_t dimension = q.size();
#endif

	auto get_vertex = [&](const size_t& index) constexpr -> std::vector<size_t>& {
		return hadj_bottom[index];
	};
	auto entries_array = quant_impl.stored.data();
	auto get_data = [&](const size_t& data_index) constexpr -> auto& {
		// return all_entries[data_index];
		return entries_array[data_index];
	};
	Eigen::VectorX<uint32_t> q_swizzled(q.size()); // = q;
	// TODO this currently assumes q.size() % 128==0
	// perform swizzle on input vector, instead of like how svs-turbo does it (on
	// saved data)
	for (size_t i = 0; i < size_t(q.size()); i += 64) {
		for (size_t j = 0; j < 4; ++j) {
			for (size_t k = 0; k < 16; ++k) {
				// q_swiz=q[3],q[7],q[11],...,q[63]
				q_swizzled[i + j * 16 + k] = uint32_t(q[i + k * 4 + j]);
			}
		}
	}
	auto score = [&](size_t data_index) constexpr {
#ifdef RECORD_STATS
		++num_distcomps;
#endif
		// return dist2(q, get_data(data_index));
		// return dist2_compressed(q, get_data(data_index));
		// return dist2_compressed(q, q_swizzled, get_data(data_index));
		return dist2_compressed(q_swizzled, get_data(data_index));
	};

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points) {
		entry_points_with_dist.emplace_back(score(entry_point), entry_point);
	}

	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points_with_dist.begin(), entry_points_with_dist.end(),
								 best_elem);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points_with_dist.begin(), entry_points_with_dist.end(),
							worst_elem);
	while (nearest.size() > k)
		nearest.pop();

	for (auto& entry_point : entry_points) {
		visited[entry_point] = true;
		visited_recent.emplace_back(entry_point);
	}

	std::vector<size_t> neighbour_list;
	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		neighbour_list.clear();
		for (size_t neighbour : get_vertex(cur.second)) {
			if (!visited[neighbour]) {
				neighbour_list.emplace_back(neighbour);
				visited[neighbour] = true;
				visited_recent.emplace_back(neighbour);
			}
		}
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
			for (size_t mult = 0; mult < DIM * sizeof(compressed_t) / 64; ++mult)
				_mm_prefetch(((char*)&get_data(neighbour_list[i])) + mult * 64,
										 _MM_HINT_T0);
#else
			for (size_t mult = 0; mult < dimension * sizeof(compressed_t) / 64;
					 ++mult)
				_mm_prefetch(((char*)&get_data(neighbour_list[i])) + mult * 64,
										 _MM_HINT_T0);
#endif
			// scorer_ref.prefetch_simple(i);
		};
		for (size_t next_i_pre = 0;
				 next_i_pre < std::min(in_advance, neighbour_list.size());
				 ++next_i_pre) {
			do_loop_prefetch(next_i_pre);
		}
		auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
												 size_t next_i) constexpr {
			if constexpr (inAdvanceIterExtra) {
				_mm_prefetch(&neighbour_list[next_i + in_advance + in_advance_extra],
										 _MM_HINT_T0);
			}
			if constexpr (inAdvanceIter) {
				do_loop_prefetch(next_i + in_advance);
			}
			const auto& next = neighbour_list[next_i];
			// T d_next = scorer_ref.score_simple(next);
			T d_next = score(next);
			if (nearest.size() < k || d_next < nearest.top().first) {
				candidates.emplace(d_next, next);
				nearest.emplace(d_next, next);
				if (nearest.size() > k)
					nearest.pop();
			}
		};
		size_t next_i = 0;
		for (; next_i + in_advance + in_advance_extra < neighbour_list.size();
				 ++next_i) {
			loop_iter.template operator()<true, true>(next_i);
		}
		for (; next_i + in_advance < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<true, false>(next_i);
		}
		for (; next_i < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<false, false>(next_i);
		}
	}
	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<measured_data> ret;
	std::sort(ret.begin(), ret.end());
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	// final rerank
	for (auto& [d, data_index] : ret) {
		// d = score(data_index);
		d = dist2(all_entries[data_index], q);
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> antitopo_engine<T>::_query_k(const vec<T>& q0, size_t k) {
	if (!ef_search.has_value())
		ef_search = k * ef_search_mult;

	const auto& q = q0.internal;
	std::vector<size_t> entry_points;
	for (size_t i = 0; i < 1; ++i) {
		size_t entry_point = starting_vertex;
		auto score = [&](size_t data_index) constexpr {
#ifdef RECORD_STATS
			++num_distcomps;
#endif
			T basic_dist = dist2(all_entries[data_index], q);
			T res = basic_dist;
			for (size_t prev_index : entry_points) {
				T co_dist = dist2(all_entries[prev_index], all_entries[data_index]);
				if (co_dist < basic_dist)
					res += ortho_factor * (basic_dist - co_dist) + ortho_bias;
			}
			return res;
		};
		T ep_dist = score(entry_point);
		for (size_t layer = max_layer - 1; layer > 0; --layer) {
			bool changed = true;
			while (changed) {
				changed = false;
				for (auto& neighbour : hadj_flat[entry_point][layer]) {
					_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
					T neighbour_dist = score(neighbour);
					if (neighbour_dist < ep_dist) {
						entry_point = neighbour;
						ep_dist = neighbour_dist;
						changed = true;
					}
				}
			}
		}
		bool dupe = false;
		for (size_t existing_ep : entry_points)
			if (existing_ep == entry_point) {
				dupe = true;
				break;
			}
		if (!dupe)
			entry_points.emplace_back(entry_point);
	}

	std::vector<std::pair<T, size_t>> ret_combined;
	if (use_compression) {
		// ret_combined = query_k_at_layer<true, true, false>(q0, 0, entry_points,
		//																									 ef_search.value(), {});
		ret_combined =
				query_k_bottom_compressed(q0, entry_points, ef_search.value());
	} else {
		ret_combined = query_k_at_layer<true, false, false>(q0, 0, entry_points,
																												ef_search.value(), {});
	}
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}

// Serialization
template <typename T>
void antitopo_engine<T>::serialize(std::ostream& out) const {
	// Serialize simple types
	out.write(reinterpret_cast<const char*>(&starting_vertex),
						sizeof(starting_vertex));
	out.write(reinterpret_cast<const char*>(&M), sizeof(M));
	out.write(reinterpret_cast<const char*>(&M0), sizeof(M0));
	out.write(reinterpret_cast<const char*>(&ef_search_mult),
						sizeof(ef_search_mult));

	// Optional type needs special handling
	bool has_ef_search = ef_search.has_value();
	out.write(reinterpret_cast<const char*>(&has_ef_search),
						sizeof(has_ef_search));
	if (has_ef_search) {
		const size_t value = ef_search.value();
		out.write(reinterpret_cast<const char*>(&value), sizeof(value));
	}

	out.write(reinterpret_cast<const char*>(&ef_construction),
						sizeof(ef_construction));
	out.write(reinterpret_cast<const char*>(&ortho_count), sizeof(ortho_count));
	out.write(reinterpret_cast<const char*>(&ortho_factor), sizeof(ortho_factor));
	out.write(reinterpret_cast<const char*>(&ortho_bias), sizeof(ortho_bias));
	out.write(reinterpret_cast<const char*>(&prune_overflow),
						sizeof(prune_overflow));
	out.write(reinterpret_cast<const char*>(&use_compression),
						sizeof(use_compression));
	out.write(reinterpret_cast<const char*>(&use_largest_direction_filtering),
						sizeof(use_largest_direction_filtering));
	out.write(reinterpret_cast<const char*>(&max_layer), sizeof(max_layer));

	// Serialize complex types
	size_t entries_count = all_entries.size();
	out.write(reinterpret_cast<const char*>(&entries_count),
						sizeof(entries_count));
	for (const auto& entry : all_entries) {
		size_t entry_size = entry.size();
		out.write(reinterpret_cast<const char*>(&entry_size), sizeof(entry_size));
		out.write(reinterpret_cast<const char*>(entry.data()),
							entry_size * sizeof(T));
	}

	// Serialize hadj_flat_with_lengths
	size_t num_vectors = hadj_flat_with_lengths.size();
	out.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
	for (const auto& vec : hadj_flat_with_lengths) {
		size_t num_layers = vec.size();
		out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
		for (const auto& layer : vec) {
			size_t num_edges = layer.size();
			out.write(reinterpret_cast<const char*>(&num_edges), sizeof(num_edges));
			for (const auto& edge : layer) {
				out.write(reinterpret_cast<const char*>(&edge.first),
									sizeof(edge.first));
				out.write(reinterpret_cast<const char*>(&edge.second),
									sizeof(edge.second));
			}
		}
	}
}

// Deserialization
template <typename T> void antitopo_engine<T>::deserialize(std::istream& in) {
	// Deserialize simple types
	in.read(reinterpret_cast<char*>(&starting_vertex), sizeof(starting_vertex));
	in.read(reinterpret_cast<char*>(&M), sizeof(M));
	in.read(reinterpret_cast<char*>(&M0), sizeof(M0));

	// search-time param, don't load
	size_t original_ef_search_mult = ef_search_mult;
	in.read(reinterpret_cast<char*>(&ef_search_mult), sizeof(ef_search_mult));
	ef_search_mult = original_ef_search_mult;

	bool has_ef_search;
	in.read(reinterpret_cast<char*>(&has_ef_search), sizeof(has_ef_search));
	if (has_ef_search) {
		size_t value;
		in.read(reinterpret_cast<char*>(&value), sizeof(value));
		ef_search = value;
	} else {
		ef_search.reset();
	}

	in.read(reinterpret_cast<char*>(&ef_construction), sizeof(ef_construction));
	in.read(reinterpret_cast<char*>(&ortho_count), sizeof(ortho_count));
	in.read(reinterpret_cast<char*>(&ortho_factor), sizeof(ortho_factor));
	in.read(reinterpret_cast<char*>(&ortho_bias), sizeof(ortho_bias));
	in.read(reinterpret_cast<char*>(&prune_overflow), sizeof(prune_overflow));

	// search-time param, don't load
	bool original_use_compression = use_compression;
	in.read(reinterpret_cast<char*>(&use_compression), sizeof(use_compression));
	use_compression = original_use_compression;

	in.read(reinterpret_cast<char*>(&use_largest_direction_filtering),
					sizeof(use_largest_direction_filtering));
	in.read(reinterpret_cast<char*>(&max_layer), sizeof(max_layer));

	// Deserialize complex types
	size_t entries_count;
	in.read(reinterpret_cast<char*>(&entries_count), sizeof(entries_count));
	all_entries.resize(entries_count);
	for (auto& entry : all_entries) {
		size_t entry_size;
		in.read(reinterpret_cast<char*>(&entry_size), sizeof(entry_size));
		entry.resize(entry_size); // Ensure the vector has the correct size
		in.read(reinterpret_cast<char*>(entry.data()), entry_size * sizeof(T));
	}

	size_t num_vectors;
	in.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));
	hadj_flat_with_lengths.resize(num_vectors);
	for (auto& vec : hadj_flat_with_lengths) {
		size_t num_layers;
		in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
		vec.resize(num_layers);
		for (auto& layer : vec) {
			size_t num_edges;
			in.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
			layer.resize(num_edges);
			for (auto& edge : layer) {
				in.read(reinterpret_cast<char*>(&edge.first), sizeof(edge.first));
				in.read(reinterpret_cast<char*>(&edge.second), sizeof(edge.second));
			}
		}
	}

	// Update hadj_flat and hadj_bottom
	hadj_flat.resize(hadj_flat_with_lengths.size());
	hadj_bottom.resize(hadj_flat_with_lengths.size());
	for (size_t entry_index = 0; entry_index < hadj_flat_with_lengths.size();
			 ++entry_index) {
		hadj_flat[entry_index].resize(hadj_flat_with_lengths[entry_index].size());
		for (size_t layer = 0; layer < hadj_flat_with_lengths[entry_index].size();
				 ++layer) {
			update_edges(layer, entry_index);
		}
	}

	// Build
	visited.resize(all_entries.size());
	visited_recent.reserve(visited.size());
}
