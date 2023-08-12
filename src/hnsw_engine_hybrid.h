#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

// basic lsh method
template <typename T>
struct hnsw_engine_hybrid : public ann_engine<T, hnsw_engine_hybrid<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t max_depth;
	size_t edge_count_mult;
	size_t starting_vertex;
	size_t num_for_1nn;
	hnsw_engine_hybrid(size_t _max_depth, size_t _edge_count_mult,
										 size_t _num_for_1nn)
			: rd(), gen(rd()), distribution(0, 1), max_depth(_max_depth),
				edge_count_mult(_edge_count_mult), num_for_1nn(_num_for_1nn) {}
	std::vector<vec<T>> all_entries;
	std::vector<
			robin_hood::unordered_flat_map<size_t, std::set<std::pair<T, size_t>>>>
			hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k_at_layer(const vec<T>& v, size_t k,
																							size_t starting_point,
																							size_t layer);
	void add_edge(size_t layer, size_t i, size_t j, size_t extra_factor = 1);
	const std::vector<std::vector<size_t>> _query_k(const vec<T>& v, size_t k,
																									bool fill_all_layers = false);
	const vec<T>& _query(const vec<T>& v);
	const std::string _name() { return "HNSW Hybrid Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		return pl;
	}
};

template <typename T>
void hnsw_engine_hybrid<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
void hnsw_engine_hybrid<T>::add_edge(size_t layer, size_t i, size_t j,
																		 size_t extra_factor) {
	T d = dist(all_entries[i], all_entries[j]);
	hadj[layer][i].emplace(-d, j);
	hadj[layer][j].emplace(-d, i);
	if (hadj[layer][i].size() > edge_count_mult * extra_factor)
		hadj[layer][i].erase(hadj[layer][i].begin());
	if (hadj[layer][j].size() > edge_count_mult * extra_factor)
		hadj[layer][j].erase(hadj[layer][j].begin());
}

template <typename T> void hnsw_engine_hybrid<T>::_build() {
	assert(all_entries.size() > 0);
	auto add_layer = [&](size_t v) {
		hadj.emplace_back();
		hadj.back()[v] = std::set<std::pair<T, size_t>>();
		starting_vertex = v;
	};
	// add one layer to start, with first vertex
	add_layer(0);
	for (size_t i = 1; i < all_entries.size(); ++i) {
		// get kNN at each layer
		std::vector<std::vector<size_t>> kNN =
				_query_k(all_entries[i], edge_count_mult, true);
		// get the layer this entry will go up to
		size_t cur_layer_ub =
				floor(-log(distribution(gen)) * 1 / log(double(edge_count_mult)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// if it is a new layer, add a layer
		if (cur_layer >= hadj.size())
			add_layer(i);
		// add all the neighbours as edges
		for (size_t layer = 0; layer <= cur_layer && layer < kNN.size(); ++layer)
			for (size_t j : kNN[layer]) {
				add_edge(layer, i, j);
			}
	}
	// add a random graph at each level to make everything a better expander
	for (size_t depth = 0; depth < hadj.size(); ++depth) {
		// find all entries at depth
		std::vector<size_t> vertices;
		for (auto& [v, _] : hadj[depth]) {
			vertices.push_back(v);
		}
		// if none or one, terminate
		if (vertices.size() <= 2)
			break;
		// for n vertices, build a graph with edge_count_mult*n random edges
		// (good+sparse expander in expectation)
		std::uniform_int_distribution<> d_verts(0, vertices.size() - 1);
		for (size_t i = 0; i < vertices.size() * edge_count_mult; ++i) {
			// pick two random vertices
			size_t a = d_verts(gen), b;
			do {
				b = d_verts(gen);
			} while (b == a);
			// create an edge between them at depth
			add_edge(depth, vertices[a], vertices[b], 2);
			// hadj[depth][vertices[a]].push_back(vertices[b]);
			// hadj[depth][vertices[b]].push_back(vertices[a]);
		}
	}
}

template <typename T>
const std::vector<size_t>
hnsw_engine_hybrid<T>::_query_k_at_layer(const vec<T>& v, size_t k,
																				 size_t starting_point, size_t layer) {
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::priority_queue<std::pair<T, size_t>> to_visit;
	robin_hood::unordered_flat_set<size_t> visited;
	auto visit = [&](T d, size_t u) {
		bool is_good =
				!visited.contains(u) && (top_k.size() < k || top_k.top().first > d);
		visited.insert(u);
		if (is_good) {
			top_k.emplace(d, u);		 // top_k is a max heap
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
		if (-nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (auto& [_, u] : hadj[layer][cur]) {
			T d_next = dist(v, all_entries[u]);
			visit(d_next, u);
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
hnsw_engine_hybrid<T>::_query_k(const vec<T>& v, size_t k,
																bool fill_all_layers) {
	auto current = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		ret.push_back(_query_k_at_layer(v, k, current, layer));
		current = ret.back().front();
	}
	reverse(ret.begin(), ret.end());
	if (!fill_all_layers)
		ret.resize(1);
	return ret;
}

template <typename T>
const vec<T>& hnsw_engine_hybrid<T>::_query(const vec<T>& v) {
	return all_entries[_query_k(v, num_for_1nn)[0][0]];
}