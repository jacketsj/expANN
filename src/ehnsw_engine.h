#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

struct ehnsw_engine_config {
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t num_cuts;
	bool quick_search;
	ehnsw_engine_config(size_t _max_depth, size_t _edge_count_mult,
											size_t _num_cuts, size_t _num_for_1nn, bool _quick_search)
			: max_depth(_max_depth), edge_count_mult(_edge_count_mult),
				num_for_1nn(_num_for_1nn), num_cuts(_num_cuts),
				quick_search(_quick_search) {}
};

// Expander HNSW Engine
template <typename T>
struct ehnsw_engine : public ann_engine<T, ehnsw_engine<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t starting_vertex;
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t num_cuts;
	bool quick_search;
	ehnsw_engine(ehnsw_engine_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				max_depth(conf.max_depth), edge_count_mult(conf.edge_count_mult),
				num_for_1nn(conf.num_for_1nn), num_cuts(conf.num_cuts),
				quick_search(conf.quick_search) {}
	std::vector<vec<T>> all_entries;
	std::vector<robin_hood::unordered_flat_map<size_t, std::vector<bool>>>
			e_labels; // level -> vertex -> cut labels (*num_cuts)
	std::vector<robin_hood::unordered_flat_map<
			size_t, std::vector<std::set<std::pair<T, size_t>>>>>
			hadj_oppo, hadj_same; // maps level -> vertex index -> cut index -> worst
														// edges to opposite side/same side of cut
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k_at_layer(
			const vec<T>& v, size_t k, size_t starting_point, size_t layer,
			std::function<bool(size_t, size_t)> filter = [](size_t, size_t) {
				return true;
			});
	void add_edge(size_t layer, size_t cut, size_t i, size_t j);
	const std::vector<std::vector<size_t>> _query_k_internal(
			const vec<T>& v, size_t k, bool fill_all_layers = false,
			std::function<bool(size_t, size_t)> filter = [](size_t, size_t) {
				return true;
			});
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "EHNSW Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		return pl;
	}
};

template <typename T> void ehnsw_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
void ehnsw_engine<T>::add_edge(size_t layer, size_t cut, size_t i, size_t j) {
	T d = dist(all_entries[i], all_entries[j]);
	auto* hadj_ptr = &hadj_oppo;
	if (e_labels[layer][i][cut] == e_labels[layer][j][cut])
		hadj_ptr = &hadj_same;
	auto& hadj = *hadj_ptr;

	hadj[layer][i][cut].emplace(-d, j);
	hadj[layer][j][cut].emplace(-d, i);
	if (hadj[layer][i][cut].size() > edge_count_mult)
		hadj[layer][i][cut].erase(hadj[layer][i][cut].begin());
	if (hadj[layer][j][cut].size() > edge_count_mult)
		hadj[layer][j][cut].erase(hadj[layer][j][cut].begin());
}

template <typename T> void ehnsw_engine<T>::_build() {
	assert(all_entries.size() > 0);
	auto add_layer = [&](size_t v) {
		hadj_same.emplace_back();
		hadj_oppo.emplace_back();
		e_labels.emplace_back();
		starting_vertex = v;
	};
	auto add_vertex_info = [&](size_t v, size_t top_layer) {
		for (size_t layer = 0; layer <= top_layer; ++layer) {
			for (size_t cut = 0; cut < num_cuts; ++cut) {
				hadj_oppo[layer][v].emplace_back();
				hadj_same[layer][v].emplace_back();
				e_labels[layer][v].emplace_back(int_distribution(gen));
			}
		}
	};
	// add one layer to start, with first vertex
	add_layer(0);
	add_vertex_info(0, 0);
	for (size_t i = 1; i < all_entries.size(); ++i) {
		// get kNN at each layer
		// get the layer this entry will go up to
		size_t cur_layer_ub =
				floor(-log(distribution(gen)) * 1 / log(double(edge_count_mult)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// if it is a new layer, add a layer
		while (cur_layer >= hadj_same.size())
			add_layer(i);
		add_vertex_info(i, cur_layer);
		std::vector<std::vector<size_t>> kNN = _query_k_internal(
				all_entries[i], edge_count_mult, true, [&](size_t layer, size_t v) {
					return e_labels[layer][v] != e_labels[layer][i];
				});
		// determine cut sides + add all the neighbours as edges
		for (size_t layer = 0; layer <= cur_layer && layer < kNN.size(); ++layer) {
			for (size_t cut = 0; cut < num_cuts; ++cut) {
				for (size_t j : kNN[layer]) {
					add_edge(layer, cut, i, j);
				}
			}
		}
	}
}
template <typename T>
const std::vector<size_t>
ehnsw_engine<T>::_query_k_at_layer(const vec<T>& v, size_t k,
																	 size_t starting_point, size_t layer,
																	 std::function<bool(size_t, size_t)> filter) {
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::priority_queue<std::pair<T, size_t>> to_visit;
	robin_hood::unordered_flat_set<size_t> visited;
	auto visit = [&](T d, size_t u) {
		bool is_good =
				!visited.contains(u) && (top_k.size() < k || top_k.top().first > d);
		//&& filter(layer, u);
		visited.insert(u);
		if (is_good) {
			if (filter(layer, u))
				top_k.emplace(d, u); // top_k is a max heap
			else
				top_k.emplace(std::numeric_limits<T>::max(),
											u);			 // top k should contain unfiltered items in the
															 // event that everything is filtered out
			to_visit.emplace(-d, u); // to_visit is a min heap
		}
		if (top_k.size() > k)
			top_k.pop();
		return is_good;
	};
	visit(dist(v, all_entries[starting_point]), starting_point);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (size_t cut = 0; cut < num_cuts; ++cut) {
			for (auto& [_, u] : hadj_same[layer][cur][cut]) {
				T d_next = dist(v, all_entries[u]);
				visit(d_next, u);
			}
			for (auto& [_, u] : hadj_oppo[layer][cur][cut]) {
				T d_next = dist(v, all_entries[u]);
				visit(d_next, u);
			}
		}
	}
	std::vector<size_t> ret;
	while (!top_k.empty()) {
		ret.push_back(top_k.top().second);
		top_k.pop();
	}
	reverse(ret.begin(), ret.end()); // sort from closest to furthest
	return ret;
}

template <typename T>
const std::vector<std::vector<size_t>>
ehnsw_engine<T>::_query_k_internal(const vec<T>& v, size_t k,
																	 bool fill_all_layers,
																	 std::function<bool(size_t, size_t)> filter) {
	auto current = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer = hadj_same.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (!fill_all_layers && layer > 0)
			layer_k = 1;
		ret.push_back(_query_k_at_layer(v, layer_k, current, layer, filter));
		current = ret.back().front();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> ehnsw_engine<T>::_query_k(const vec<T>& v, size_t k) {
	auto ret = _query_k_internal(v, k * num_for_1nn, !quick_search)[0];
	ret.resize(std::min(k, ret.size()));
	return ret;
}
