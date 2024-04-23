#pragma once

#include <algorithm>
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
#include "distance.h"
#include "quantizer.h"
#include "robin_hood.h"
#include "topk_t.h"

namespace {
template <typename A, typename B> auto dist2(const A& a, const B& b) {
	return (a - b).squaredNorm();
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
	antitopo_engine_config(size_t _M, size_t _M0, size_t _ef_search_mult,
												 size_t _ef_construction, size_t _ortho_count,
												 float _ortho_factor, float _ortho_bias,
												 size_t _prune_overflow, bool _use_compression = false,
												 bool _use_largest_direction_filtering = false)
			: antitopo_engine_query_config(_ef_search_mult), M(_M), M0(_M0),
				ef_construction(_ef_construction), ortho_count(_ortho_count),
				ortho_factor(_ortho_factor), ortho_bias(_ortho_bias),
				prune_overflow(_prune_overflow), use_compression(_use_compression),
				use_largest_direction_filtering(_use_largest_direction_filtering) {}
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
	size_t max_layer;
#ifdef RECORD_STATS
	size_t num_distcomps;
	size_t num_distcomps_compressed;
#endif
	antitopo_engine(size_t _M, size_t _ef_construction, size_t _ortho_count,
									float _ortho_factor, float _ortho_bias,
									size_t _prune_overflow)
			: rd(), gen(0), distribution(0, 1), M(_M), M0(2 * _M), ef_search_mult(1),
				ef_search(std::nullopt), ef_construction(_ef_construction),
				ortho_count(_ortho_count), ortho_factor(_ortho_factor),
				ortho_bias(_ortho_bias), prune_overflow(_prune_overflow),
				use_compression(false), use_largest_direction_filtering(false),
				max_layer(0) {
		quant = std::make_unique<quantizer_simple>();
	}
	antitopo_engine(antitopo_engine_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult), ef_search(conf.ef_search),
				ef_construction(conf.ef_construction), ortho_count(conf.ortho_count),
				ortho_factor(conf.ortho_factor), ortho_bias(conf.ortho_bias),
				prune_overflow(conf.prune_overflow),
				use_compression(conf.use_compression),
				use_largest_direction_filtering(conf.use_largest_direction_filtering),
				max_layer(0) {
		quant = std::make_unique<quantizer_simple>();
	}
	using config = antitopo_engine_config;
	using query_config = antitopo_engine_query_config;
	void set_ef_search(size_t _ef_search) { ef_search = _ef_search; }
	std::vector<fvec> all_entries;
	std::unique_ptr<quantizer> quant;
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
	update_edges(layer, from);
}

template <typename T>
void antitopo_engine<T>::_store_vector(const vec<T>& v0, bool silent) {
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
	assert(all_entries.size() > 0);

	quant->build(all_entries);

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
			/*
#ifdef DIM
			return distance_compare_avx512f_f16(q.data(), get_data(data_index).data(),
																					DIM);
#else
			return distance_compare_avx512f_f16(q.data(), get_data(data_index).data(),
																					dimension);
#endif
*/
			return dist2(q, get_data(data_index));
		}
	};
	auto scorer = quant->generate_scorer(q);
	auto score_compressed = [&](size_t data_index) constexpr {
#ifdef RECORD_STATS
		++num_distcomps_compressed;
#endif
		return scorer->score(data_index);
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
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest_big(entry_points_with_dist.begin(), entry_points_with_dist.end(),
									worst_elem);
	size_t big_factor = 1;
	if constexpr (use_compressed) {
		while (nearest_big.size() > big_factor * k)
			nearest_big.pop();
	}

	for (auto& entry_point : entry_points) {
		visited[entry_point] = true;
		visited_recent.emplace_back(entry_point);
	}

	std::vector<size_t> neighbour_list, neighbour_list_unfiltered;
	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		neighbour_list.clear();
		neighbour_list_unfiltered.clear();
		for (size_t neighbour : get_vertex(cur.second))
			if (!visited[neighbour]) {
				if constexpr (use_compressed) {
					neighbour_list_unfiltered.emplace_back(neighbour);
				} else {
					neighbour_list.emplace_back(neighbour);
				}
				visited[neighbour] = true;
				visited_recent.emplace_back(neighbour);
			}
		if constexpr (use_compressed) {
			constexpr size_t in_advance = 4;
			constexpr size_t in_advance_extra = 2;
			auto do_loop_prefetch = [&](size_t i) constexpr { scorer->prefetch(i); };
			for (size_t next_i_pre = 0;
					 next_i_pre < std::min(in_advance, neighbour_list_unfiltered.size());
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
				T d_next = score_compressed(next);
				if (nearest_big.size() < big_factor * k ||
						d_next < nearest_big.top().first) {
					neighbour_list.emplace_back(next);
				}
			};
			size_t next_i = 0;
			for (; next_i + in_advance + in_advance_extra <
						 neighbour_list_unfiltered.size();
					 ++next_i) {
				loop_iter.template operator()<true, true>(next_i);
			}
			for (; next_i + in_advance < neighbour_list_unfiltered.size(); ++next_i) {
				loop_iter.template operator()<true, false>(next_i);
			}
			for (; next_i < neighbour_list_unfiltered.size(); ++next_i) {
				loop_iter.template operator()<false, false>(next_i);
			}
		}
		{
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
	if constexpr (use_compressed) {
		for (auto& [d, data_index] : ret) {
			d = score(data_index);
		}
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
		ret_combined = query_k_at_layer<true, true, false>(q0, 0, entry_points,
																											 ef_search.value(), {});
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
