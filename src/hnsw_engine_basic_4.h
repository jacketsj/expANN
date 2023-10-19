#pragma once

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"
#include "topk_t.h"

struct hnsw_engine_basic_4_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	hnsw_engine_basic_4_config(size_t _M, size_t _M0, size_t _ef_search_mult,
														 size_t _ef_construction)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction) {}
};

template <typename T>
struct hnsw_engine_basic_4 : public ann_engine<T, hnsw_engine_basic_4<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	hnsw_engine_basic_4(hnsw_engine_basic_4_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction) {}
	std::vector<vec<T>> all_entries;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<
			robin_hood::unordered_flat_map<size_t, std::vector<std::pair<T, size_t>>>>
			hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<char> visited; // booleans
	std::vector<size_t> visited_recent;
	std::vector<std::pair<T, size_t>>
	prune_edges(size_t layer, std::vector<std::pair<T, size_t>> to);
	template <bool use_bottomlayer>
	std::vector<std::pair<T, size_t>>
	query_k_alt(const vec<T>& q, size_t layer,
							const std::vector<size_t>& entry_points, size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "HNSW Engine Basic 4"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		return pl;
	}
};

template <typename T>
std::vector<std::pair<T, size_t>>
hnsw_engine_basic_4<T>::prune_edges(size_t layer,
																		std::vector<std::pair<T, size_t>> to) {
	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

	// reference impl vs paper difference
	if (to.size() <= edge_count_mult) {
		return to;
	}

	sort(to.begin(), to.end());
	std::vector<std::pair<T, size_t>> ret;
	for (const auto& md : to) {
		_mm_prefetch(&all_entries[md.second], _MM_HINT_T0);
		if (ret.size() >= edge_count_mult)
			break;
		bool choose = true;
		for (const auto& md_chosen : ret) {
			if (md.first == md_chosen.first ||
					dist2(all_entries[md.second], all_entries[md_chosen.second]) <=
							md.first) {
				choose = false;
				break;
			}
		}
		if (choose) {
			ret.emplace_back(md);
		}
	}

	return ret;
}

template <typename T>
void hnsw_engine_basic_4<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.push_back(v);

	auto convert_el = [](std::vector<std::pair<T, size_t>> el) constexpr {
		std::vector<size_t> ret;
		for (auto& [_, val] : el) {
			ret.emplace_back(val);
		}
		return ret;
	};

	// get kNN for each layer
	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));
	std::vector<std::vector<std::pair<T, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		for (int layer = hadj.size() - 1; layer > int(new_max_layer); --layer) {
			kNN_per_layer.emplace_back(query_k_alt<false>(v, layer, cur, 1));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
		}
		for (int layer = std::min(new_max_layer, hadj.size() - 1); layer >= 0;
				 --layer) {
			kNN_per_layer.emplace_back(
					query_k_alt<false>(v, layer, cur, ef_construction));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
			cur.resize(1); // present in reference impl, but not in hnsw paper
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}

	// add the found edges to the graph
	for (size_t layer = 0; layer < std::min(hadj.size(), new_max_layer + 1);
			 ++layer) {
		hadj[layer][v_index] = prune_edges(layer, kNN_per_layer[layer]);
		//  add bidirectional connections, prune if necessary
		for (auto& md : kNN_per_layer[layer]) {
			bool edge_exists = false;
			for (auto& md_other : hadj[layer][md.second]) {
				if (md_other.second == v_index) {
					edge_exists = true;
				}
			}
			if (!edge_exists) {
				hadj[layer][md.second].emplace_back(md.first, v_index);
				hadj[layer][md.second] = prune_edges(layer, hadj[layer][md.second]);
				hadj_flat[md.second][layer] = convert_el(hadj[layer][md.second]);
				if (layer == 0)
					hadj_bottom[md.second] = hadj_flat[md.second][layer];
			}
		}
	}

	// add new layers if necessary
	while (new_max_layer >= hadj.size()) {
		hadj.emplace_back();
		hadj.back()[v_index] = std::vector<std::pair<T, size_t>>();
		starting_vertex = v_index;
	}

	visited.emplace_back();
	hadj_flat.emplace_back();
	hadj_bottom.emplace_back();
	hadj_bottom[v_index] = convert_el(hadj[0][v_index]);
	for (size_t layer = 0; layer < hadj.size(); ++layer) {
		if (hadj[layer].contains(v_index)) {
			hadj_flat[v_index].emplace_back();
			hadj_flat[v_index][layer] = convert_el(hadj[layer][v_index]);
		} else {
			break;
		}
	}
}

template <typename T> void hnsw_engine_basic_4<T>::_build() {
	assert(all_entries.size() > 0);
}

template <typename T>
template <bool use_bottomlayer>
std::vector<std::pair<T, size_t>>
hnsw_engine_basic_4<T>::query_k_alt(const vec<T>& q, size_t layer,
																		const std::vector<size_t>& entry_points,
																		size_t k) {
	using measured_data = std::pair<T, size_t>;

	auto get_vertex = [&](const size_t& index) constexpr->std::vector<size_t>& {
		if constexpr (use_bottomlayer) {
			return hadj_bottom[index];
		} else {
			return hadj_flat[index][layer];
		}
	};

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points)
		entry_points_with_dist.emplace_back(dist2(q, all_entries[entry_point]),
																				entry_point);

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
		_mm_prefetch(&get_vertex(cur.second)[0], _MM_HINT_T0);
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto do_loop_prefetch = [&](size_t i) {
#ifdef DIM
			for (size_t mult = 0; mult < DIM * 4 / 64; ++mult)
				_mm_prefetch(((char*)&all_entries[get_vertex(cur.second)[i]]) +
												 mult * 64,
										 _MM_HINT_T0);
#endif
			_mm_prefetch(&visited[get_vertex(cur.second)[i]], _MM_HINT_T0);
		};
		for (size_t next_i_pre = 0;
				 next_i_pre < std::min(in_advance, get_vertex(cur.second).size());
				 ++next_i_pre) {
			do_loop_prefetch(next_i_pre);
		}
		auto loop_iter =
				[&]<bool inAdvanceIter, bool inAdvanceIterExtra>(size_t next_i) {
					if constexpr (inAdvanceIterExtra) {
						_mm_prefetch(
								&get_vertex(cur.second)[next_i + in_advance + in_advance_extra],
								_MM_HINT_T0);
					}
					if constexpr (inAdvanceIter) {
						do_loop_prefetch(next_i + in_advance);
					}
					const auto& next = get_vertex(cur.second)[next_i];
					if (!visited[next]) {
						visited[next] = true;
						visited_recent.emplace_back(next);
						T d_next = dist2(q, all_entries[next]);
						if (nearest.size() < k || d_next < nearest.top().first) {
							candidates.emplace(d_next, next);
							nearest.emplace(d_next, next);
							if (nearest.size() > k)
								nearest.pop();
						}
					}
				};
		size_t next_i = 0;
		for (;
				 next_i + in_advance + in_advance_extra < get_vertex(cur.second).size();
				 ++next_i) {
			loop_iter.template operator()<true, true>(next_i);
		}
		for (; next_i + in_advance < get_vertex(cur.second).size(); ++next_i) {
			loop_iter.template operator()<true, false>(next_i);
		}
		for (; next_i < get_vertex(cur.second).size(); ++next_i) {
			loop_iter.template operator()<false, false>(next_i);
		}
	}
	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> hnsw_engine_basic_4<T>::_query_k(const vec<T>& q,
																										 size_t k) {
	size_t entry_point = starting_vertex;
	T ep_dist = dist2(all_entries[entry_point], q);
	for (size_t layer = hadj.size() - 1; layer > 0; --layer) {
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto& neighbour : hadj_flat[entry_point][layer]) {
				_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
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
			query_k_alt<true>(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}
