#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"
#include "topk_t.h"

struct hnsw_engine_basic_2_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	hnsw_engine_basic_2_config(size_t _M, size_t _M0, size_t _ef_search_mult,
														 size_t _ef_construction)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction) {}
};

template <typename T>
struct hnsw_engine_basic_2 : public ann_engine<T, hnsw_engine_basic_2<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	hnsw_engine_basic_2(hnsw_engine_basic_2_config conf)
			: rd(), gen(rd()), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction) {}
	std::vector<vec<T>> all_entries;
	std::vector<
			robin_hood::unordered_flat_map<size_t, std::vector<std::pair<T, size_t>>>>
			hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const vec<T>& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k);
	std::vector<std::pair<T, size_t>>
	prune_edges(size_t layer, std::vector<std::pair<T, size_t>> to);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "HNSW Engine Basic 2"; }
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
hnsw_engine_basic_2<T>::prune_edges(size_t layer,
																		std::vector<std::pair<T, size_t>> to) {
	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

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
void hnsw_engine_basic_2<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.push_back(v);

	// get kNN for each layer
	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));
	std::vector<std::vector<std::pair<T, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		for (int layer = hadj.size() - 1; layer > new_max_layer; --layer) {
			kNN_per_layer.emplace_back(query_k_at_layer(v, layer, cur, 1));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
		}
		for (int layer = std::min(new_max_layer, hadj.size() - 1); layer >= 0;
				 --layer) {
			kNN_per_layer.emplace_back(
					query_k_at_layer(v, layer, cur, ef_construction));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}

	// add the found edges to the graph
	for (size_t layer = 0; layer <= hadj.size(); ++layer) {
		hadj.back()[v_index] = prune_edges(layer, kNN_per_layer[layer]);
		// add bidirectional connections, prune if necessary
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
			}
		}
	}

	while (new_max_layer >= hadj.size()) {
		hadj.emplace_back();
		hadj.back()[v_index] = std::vector<std::pair<T, size_t>>();
		starting_vertex = v_index;
	}
}

template <typename T> void hnsw_engine_basic_2<T>::_build() {
	assert(all_entries.size() > 0);
}

template <typename T>
std::vector<std::pair<T, size_t>> hnsw_engine_basic_2<T>::query_k_at_layer(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;
	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (const auto& ep : entry_points) {
		entry_points_with_dist.emplace_back(dist2(q, all_entries[ep]), ep);
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

	robin_hood::unordered_flat_set<size_t> visited;
	for (const auto& ep : entry_points)
		visited.insert(ep);

	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			// TODO second condition should be unnecessary as written
			break;
		}
		// TODO this might affect things positively or negatively
		// for (size_t next : hadj[layer][cur.first]) {
		//	_mm_prefetch(&all_entries[next], _MM_HINT_T0);
		// }
		for (const auto& [_, next] : hadj[layer][cur.first]) {
			_mm_prefetch(&all_entries[next], _MM_HINT_T0);
			if (!visited.contains(next)) {
				visited.insert(next);
				T d_next = dist2(q, all_entries[next]);
				if (d_next < nearest.top().first || nearest.size() < k) {
					candidates.emplace(d_next, next);
					nearest.emplace(d_next, next);
					if (nearest.size() > k)
						nearest.pop();
				}
			}
		}
	}
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> hnsw_engine_basic_2<T>::_query_k(const vec<T>& q,
																										 size_t k) {
	size_t cur_vert = 0;
	int layer;
	for (layer = hadj.size() - 1; layer > 0; --layer)
		cur_vert = query_k_at_layer(q, layer, {cur_vert}, 1)[0].second;
	auto ret_combined = query_k_at_layer(q, 0, {cur_vert}, k * ef_search_mult);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}
