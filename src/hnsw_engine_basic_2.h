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
	size_t ef_search;
	size_t ef_construction;
	hnsw_engine_basic_2_config(size_t _M, size_t _M0, size_t _ef_search,
														 size_t _ef_construction)
			: M(_M), M0(_M0), ef_search(_ef_search),
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
	size_t ef_search;
	size_t ef_construction;
	hnsw_engine_basic_2(hnsw_engine_basic_2_config conf)
			: rd(), gen(rd()), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search(conf.ef_search), ef_construction(conf.ef_construction) {}
	std::vector<vec<T>> all_entries;
	std::vector<
			robin_hood::unordered_flat_map<size_t, std::vector<std::pair<T, size_t>>>>
			hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<std::pair<T, size_t>>
	_query_k_at_layer(const vec<T>& q, size_t layer,
										const std::vector<size_t>& entry_points, size_t k);
	std::vector<std::pair<T, size_t>>
	prune_edges(size_t layer, std::vector<std::pair<T, size_t>> to);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "HNSW Engine Basic 2"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search);
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
		_mm_prefetch(&all_entries[md.second]);
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
		std::vector<size_t> cur = starting_vertex;
		for (int layer = hadj.size() - 1; layer > new_max_layer; --layer) {
			kNN_per_layer.emplace_back(query_k_at_layer(v, layer, cur, 1));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
		}
		for (int layer = std::min(max_new_layer, hadj.size() - 1); layer >= 0;
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
	auto add_layer = [&](size_t v) {
		hadj.emplace_back();
		hadj.back()[v] = std::set<std::pair<T, size_t>>();
		starting_vertex = v;
	};
	// add one layer to start, with first vertex
	add_layer(0);
	for (size_t i = 1; i < all_entries.size(); ++i) {
		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;
		// get the layer this entry will go up to
		size_t cur_layer_ub = floor(-log(distribution(gen)) * 1 / log(double(M)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// if it is a new layer, add a layer
		while (cur_layer >= hadj.size())
			add_layer(i);
		// search for new edges to add
		size_t cur_vert = starting_vertex;
		int layer;
		for (layer = hadj.size() - 1; layer > int(cur_layer); --layer) {
			topk_t<T> tk(1);
			tk.consider(dist2(all_entries[i], all_entries[cur_vert]), cur_vert);
			cur_vert = _query_k_at_layer_internal(all_entries[i], tk, layer)[0];
		}
		for (; layer >= 0; --layer) {
			topk_t<T> tk(ef_construction);
			tk.consider(dist2(all_entries[i], all_entries[cur_vert]), cur_vert);
			auto closest_entries =
					_query_k_at_layer_internal(all_entries[i], tk, layer);
			cur_vert = closest_entries[0];
			for (size_t j : closest_entries) {
				add_edge(layer, i, j);
				// "neighbour selection heuristic" from paper
				for (auto& [_, j0] : hadj[layer][j]) {
					add_edge(layer, i, j0);
				}
			}
		}
		// tk.consider(dist2(all_entries[starting_vertex], all_entries[i]),
		//						starting_vertex);
		// for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		//	// get kNN at current layer
		//	tk.discard_until_size(1);
		//	std::vector<size_t> kNN =
		//			_query_k_at_layer_internal(all_entries[i], tk, layer);
		//	if (layer <= int(cur_layer))
		//		for (size_t j : kNN) {
		//			add_edge(layer, i, j);
		//		}
		// }
	}
}
template <typename T>
std::vector<std::pair<T, size_t>> hnsw_engine_basic_2<T>::_query_k_at_layer(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;
	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points.begin(), entry_points.end(), best_elem);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points.begin(), entry_points.end(), worst_elem);
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
		for (size_t next : hadj[layer][cur.first]) {
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
}

template <typename T>
std::vector<size_t> hnsw_engine_basic_2<T>::_query_k(const vec<T>& q,
																										 size_t k) {
	size_t cur_vert = 0;
	int layer;
	for (layer = hadj.size() - 1; layer > 0; --layer)
		cur_vert = _query_k_at_layer_internal(q, layer, {cur_vert}, 1)[0].second;
	auto ret_combined =
			_query_k_at_layer_internal(q, 0, {cur_vert}, k * ef_search);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}
