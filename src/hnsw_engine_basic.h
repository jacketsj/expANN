#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"
#include "topk_t.h"

template <typename T>
struct hnsw_engine_basic : public ann_engine<T, hnsw_engine_basic<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t max_depth;
	size_t M;
	size_t M0;
	size_t ef_search;
	size_t ef_construction;
	hnsw_engine_basic(size_t _max_depth, size_t _M,
										// size_t _M0,
										size_t _ef_search
										//,size_t _ef_construction
										)
			: rd(), gen(rd()), distribution(0, 1), max_depth(_max_depth), M(_M),
				M0(2 * M), ef_search(_ef_search), ef_construction(6 * M)
	//, ef_construction(_ef_construction)
	{}
	std::vector<vec<T>> all_entries;
	std::vector<
			robin_hood::unordered_flat_map<size_t, std::set<std::pair<T, size_t>>>>
			hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k_at_layer_internal(const vec<T>& v, topk_t<T>& tk,
																								 size_t layer);
	void add_edge(size_t layer, size_t i, size_t j);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "HNSW Engine Basic"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search);
		add_param(pl, ef_construction);
		return pl;
	}
};

template <typename T>
void hnsw_engine_basic<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
void hnsw_engine_basic<T>::add_edge(size_t layer, size_t i, size_t j) {
	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;
	T d = dist2(all_entries[i], all_entries[j]);
	hadj[layer][i].emplace(-d, j);
	hadj[layer][j].emplace(-d, i);
	if (hadj[layer][i].size() > edge_count_mult)
		hadj[layer][i].erase(hadj[layer][i].begin());
	if (hadj[layer][j].size() > edge_count_mult)
		hadj[layer][j].erase(hadj[layer][j].begin());
}

template <typename T> void hnsw_engine_basic<T>::_build() {
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
			for (size_t j : closest_entries)
				add_edge(layer, i, j);
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
std::vector<size_t>
hnsw_engine_basic<T>::_query_k_at_layer_internal(const vec<T>& v, topk_t<T>& tk,
																								 size_t layer) {
	std::priority_queue<std::pair<T, size_t>> to_visit;
	robin_hood::unordered_flat_set<size_t> visited;
	auto visit = [&](T d, size_t u) {
		bool is_good = !visited.contains(u) && tk.consider(d, u);
		visited.insert(u);
		if (is_good) {
			to_visit.emplace(-d, u); // to_visit is a min heap
		}
		return is_good;
	};
	for (auto& u : tk.to_vector())
		to_visit.emplace(dist2(v, all_entries[u]), u);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (-nd > tk.worst_val())
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (auto& [_, u] : hadj[layer][cur]) {
			T d_next = dist2(v, all_entries[u]);
			visit(d_next, u);
		}
	}
	return tk.to_vector();
}

template <typename T>
std::vector<size_t> hnsw_engine_basic<T>::_query_k(const vec<T>& v, size_t k) {
	size_t cur_vert = 0;
	int layer;
	for (layer = hadj.size() - 1; layer > 0; --layer) {
		topk_t<T> tk(1);
		tk.consider(dist2(v, all_entries[cur_vert]), cur_vert);
		cur_vert = _query_k_at_layer_internal(v, tk, layer)[0];
	}
	topk_t<T> tk(ef_search * k);
	tk.consider(dist2(v, all_entries[cur_vert]), cur_vert);
	return _query_k_at_layer_internal(v, tk, layer);
}
