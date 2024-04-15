#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"
#include "topk_t.h"

namespace {
// template <typename A, typename B> auto dist2(const A& a, const B& b) {
// 	return (a - b).squaredNorm();
// }
} // namespace

struct mips_antitopo_engine_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	std::string scalar_quant;
	bool use_largest_direction_filtering;
	mips_antitopo_engine_config(size_t _M, size_t _M0, size_t _ef_search_mult,
															size_t _ef_construction,
															std::string _scalar_quant,
															bool _use_largest_direction_filtering = false)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction), scalar_quant(_scalar_quant),
				use_largest_direction_filtering(_use_largest_direction_filtering) {}
};

template <typename T>
struct mips_antitopo_engine : public ann_engine<T, mips_antitopo_engine<T>> {
	using fvec = typename vec<T>::Underlying;
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	std::string scalar_quant;
	bool use_largest_direction_filtering;
	size_t max_layer;
#ifdef RECORD_STATS
	size_t num_distcomps;
#endif
	mips_antitopo_engine(mips_antitopo_engine_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction), scalar_quant(conf.scalar_quant),
				use_largest_direction_filtering(conf.use_largest_direction_filtering),
				max_layer(0) {}
	using config = mips_antitopo_engine_config;
	std::vector<fvec> all_entries;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<std::vector<std::vector<std::pair<T, size_t>>>>
			hadj_flat_with_lengths; // vector -> layer -> edges with lengths
	void _store_vector(const vec<T>& v0);
	void improve_entries(const std::vector<size_t>& data_indices);
	std::vector<std::vector<std::pair<T, size_t>>>
	get_knn_per_layer(size_t data_index);
	void add_edges(size_t layer, size_t data_index,
								 const std::vector<std::pair<T, size_t>>& new_edges);
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
	template <bool use_bottomlayer>
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const fvec& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	std::vector<std::pair<T, size_t>> query_k_combined(const vec<T>& v, size_t k);
	const std::string _name() { return "MIPS Anti-Topo Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		add_param_str(pl, scalar_quant);
		add_param(pl, use_largest_direction_filtering);
#ifdef RECORD_STATS
		add_param(pl, num_distcomps);
#endif
		return pl;
	}
};

template <typename T>
void mips_antitopo_engine<T>::add_edges(
		size_t layer, size_t data_index,
		const std::vector<std::pair<T, size_t>>& new_edges) {
	std::vector<std::pair<T, size_t>> pruned_edges_dupe;
	{
		for (const auto& edge : new_edges)
			hadj_flat_with_lengths[data_index][layer].emplace_back(edge);
		prune_edges(layer, data_index, false);
		pruned_edges_dupe = hadj_flat_with_lengths[data_index][layer];
	}
	std::vector<std::pair<T, size_t>> pruned_edges_dupe_filtered;
	size_t edge_count_mult = M;
	if (layer == 0)
		edge_count_mult *= 2;
	for (auto& md : pruned_edges_dupe) {
		bool same_edge_exists = false;
		bool smaller_edge_exists =
				hadj_flat_with_lengths[md.second][layer].size() < edge_count_mult;
		;
		for (auto& md_other : hadj_flat_with_lengths[md.second][layer]) {
			if (md_other.second == data_index) {
				same_edge_exists = true;
				break;
			}
			if (md_other.first < md.first) {
				smaller_edge_exists = true; // reduce contention for starting nodes
			}
		}
		if (!same_edge_exists && smaller_edge_exists) {
			pruned_edges_dupe_filtered.emplace_back(md);
		}
	}
	std::vector<std::pair<T, size_t>> pruned_edges_dupe_refiltered;
	for (auto& md : pruned_edges_dupe_filtered) {
		if (hadj_flat_with_lengths[md.second][layer].size() ==
				hadj_flat[md.second][layer].size())
			pruned_edges_dupe_refiltered.emplace_back(md);
		hadj_flat_with_lengths[md.second][layer].emplace_back(md.first, data_index);
	}
	for (auto& md : pruned_edges_dupe_refiltered) {
		prune_edges(layer, md.second, true);
	}
}

template <typename T>
std::vector<std::vector<std::pair<T, size_t>>>
mips_antitopo_engine<T>::get_knn_per_layer(size_t data_index) {
	const auto& v = all_entries[data_index];
	const auto local_max_layer = hadj_flat_with_lengths[data_index].size() - 1;
	std::vector<std::vector<std::pair<T, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		{
			size_t entry_point = starting_vertex;
			T ep_dist = dist2(v, all_entries[entry_point]);
			for (size_t layer = max_layer - 1; layer > local_max_layer; --layer) {
				bool changed = true;
				while (changed) {
					changed = false;
					for (auto& neighbour : hadj_flat[entry_point][layer]) {
						_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
						T neighbour_dist = dist2(v, all_entries[neighbour]);
						if (neighbour_dist < ep_dist) {
							entry_point = neighbour;
							ep_dist = neighbour_dist;
							changed = true;
						}
					}
				}
			}
			cur = {entry_point};
		}
		for (int layer = std::min(local_max_layer, max_layer - 1); layer >= 0;
				 --layer) {
			// add old edges as candidates (if any)
			for (size_t old_neighbour : hadj_flat[data_index][layer]) {
				if (old_neighbour != cur[0])
					cur.emplace_back(old_neighbour);
			}
			if (layer == 0) {
				kNN_per_layer.emplace_back(
						query_k_at_layer<false>(v, layer, cur, ef_construction));
			} else {
				kNN_per_layer.emplace_back(
						query_k_at_layer<false>(v, layer, cur, ef_construction));
			}
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
			cur.resize(1);
			// Don't add data_index to its own neighbours (but do use it for next
			// layer of queries)
			if (!kNN_per_layer.back().empty() &&
					kNN_per_layer.back()[0].second == data_index) {
				kNN_per_layer.back().erase(kNN_per_layer.back().begin());
			}
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}
	return kNN_per_layer;
}

template <typename T>
void mips_antitopo_engine<T>::improve_entries(
		const std::vector<size_t>& data_indices) {
	for (size_t v_index : data_indices) {
		auto kNN_per_layer = get_knn_per_layer(v_index);

		const auto local_max_layer = hadj_flat_with_lengths[v_index].size() - 1;

		// add the found edges to the graph
		size_t layer = 0;
		for (; layer < std::min(max_layer, local_max_layer + 1); ++layer) {
			add_edges(layer, v_index, kNN_per_layer[layer]);
		}
	}
}

template <typename T>
void mips_antitopo_engine<T>::prune_edges(size_t layer, size_t from,
																					bool lazy) {
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
	std::vector<std::pair<T, size_t>> ret;
	std::vector<fvec> normalized_ret;
	auto origin = all_entries[from];
	// std::unordered_set<size_t> taken;
	for (const auto& md : to) {
		bool choose = true;
		auto v1 = (all_entries[md.second] - origin).normalized();
		// size_t max_i = 0;
		float max_val = 0;
		if (use_largest_direction_filtering) {
			for (size_t i = 0; i < size_t(v1.size()); ++i) {
				float cur_val = v1[i];
				if (cur_val > max_val) {
					// max_i = i;
					max_val = cur_val;
				} else if (-cur_val > max_val) {
					// max_i = i + v1.size();
					max_val = -cur_val;
				}
			}
			// if (taken.contains(max_i)) {
			// choose = false;
			//}
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
			// taken.insert(max_i);
			ret.emplace_back(md);
			normalized_ret.emplace_back(v1);
			if (ret.size() >= edge_count_mult)
				break;
		}
	}
	to = ret;
	update_edges(layer, from);
}

template <typename T>
void mips_antitopo_engine<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.emplace_back(v.internal);

	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));

	visited.emplace_back();
	hadj_flat_with_lengths.emplace_back();
	hadj_flat.emplace_back();
	hadj_bottom.emplace_back();
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat_with_lengths[v_index].emplace_back();
		hadj_flat[v_index].emplace_back();
	}

	// add new layers if necessary
	while (new_max_layer >= max_layer) {
		++max_layer;
		starting_vertex = v_index;
	}
}

template <typename T> void mips_antitopo_engine<T>::_build() {
	assert(all_entries.size() > 0);
	// TODO init random graph
	for (size_t iter = 0; iter < 2; ++iter) {
		std::cerr << "Full improvement round iteration=" << iter << std::endl;
		for (size_t v_index = 0; v_index < all_entries.size(); ++v_index) {
			if (v_index % 20000 == 0) {
				std::cerr << "Improving v_index=" << v_index << std::endl;
			}
			improve_entries({v_index});
		}
	}

	// TODO now create quantized mips vecs, store in a substructure with flat
	// graph
#ifdef RECORD_STATS
	// reset before queries
	num_distcomps = 0;
#endif
}

template <typename T>
template <bool use_bottomlayer>
std::vector<std::pair<T, size_t>> mips_antitopo_engine<T>::query_k_at_layer(
		const fvec& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;

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

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points) {
#ifdef RECORD_STATS
		++num_distcomps;
#endif
		entry_points_with_dist.emplace_back(dist2(q, get_data(entry_point)),
																				entry_point);
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

	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		std::vector<size_t> neighbour_list; // = get_vertex(cur.second);
		for (size_t neighbour : get_vertex(cur.second))
			if (!visited[neighbour]) {
				neighbour_list.emplace_back(neighbour);
				visited[neighbour] = true;
				visited_recent.emplace_back(neighbour);
			}
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
			for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
				_mm_prefetch(((char*)&get_data(neighbour_list[i])) + mult * 64,
										 _MM_HINT_T0);
#endif
			//_mm_prefetch(&visited[neighbour_list[i]], _MM_HINT_T0);
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
			// if (!visited[next]) {
			// visited[next] = true;
			// visited_recent.emplace_back(next);
#ifdef RECORD_STATS
			++num_distcomps;
#endif
			T d_next = dist2(q, get_data(next));
			if (nearest.size() < k || d_next < nearest.top().first) {
				candidates.emplace(d_next, next);
				nearest.emplace(d_next, next);
				if (nearest.size() > k)
					nearest.pop();
			}
			//}
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
	// TODO use reranking here
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> mips_antitopo_engine<T>::_query_k(const vec<T>& q0,
																											size_t k) {
	const auto& q = q0.internal;
	size_t entry_point = starting_vertex;
#ifdef RECORD_STATS
	++num_distcomps;
#endif
	T ep_dist = dist2(all_entries[entry_point], q);
	for (size_t layer = max_layer - 1; layer > 0; --layer) {
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto& neighbour : hadj_flat[entry_point][layer]) {
				_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
#ifdef RECORD_STATS
				++num_distcomps;
#endif
				T neighbour_dist = dist2(q, all_entries[neighbour]);
				if (neighbour_dist < ep_dist) {
					entry_point = neighbour;
					ep_dist = neighbour_dist;
					changed = true;
				}
			}
		}
	}

	std::vector<std::pair<T, size_t>> ret_combined;
	ret_combined =
			query_k_at_layer<true>(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}

template <typename T>
std::vector<std::pair<T, size_t>>
mips_antitopo_engine<T>::query_k_combined(const vec<T>& q, size_t k) {
	size_t entry_point = starting_vertex;
#ifdef RECORD_STATS
	++num_distcomps;
#endif
	T ep_dist = dist2(all_entries[entry_point], q);
	for (int layer = max_layer - 1; layer >= 0; --layer) {
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto& neighbour : hadj_flat[entry_point][layer]) {
				_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
#ifdef RECORD_STATS
				++num_distcomps;
#endif
				T neighbour_dist = dist2(q, all_entries[neighbour]);
				if (neighbour_dist < ep_dist) {
					entry_point = neighbour;
					ep_dist = neighbour_dist;
					changed = true;
				}
			}
		}
	}

	auto ret_combined =
			query_k_at_layer<true>(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	return ret_combined;
}
