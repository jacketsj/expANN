#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

struct hnsw_engine_2_config {
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	bool quick_search;
	hnsw_engine_2_config(size_t _max_depth, size_t _edge_count_mult,
											 size_t _num_for_1nn, bool _quick_search)
			: max_depth(_max_depth), edge_count_mult(_edge_count_mult),
				num_for_1nn(_num_for_1nn), quick_search(_quick_search) {}
};

// basic lsh method
template <typename T>
struct hnsw_engine_2 : public ann_engine<T, hnsw_engine_2<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	bool quick_search;
	hnsw_engine_2(hnsw_engine_2_config conf)
			: rd(),
				// TODO revert seeding to rd()
				gen(2),
				// gen(rd()),
				distribution(0, 1), max_depth(conf.max_depth),
				edge_count_mult(conf.edge_count_mult), num_for_1nn(conf.num_for_1nn),
				quick_search(conf.quick_search) {}
	std::vector<vec<T>> all_entries;
	std::vector<
			robin_hood::unordered_flat_map<size_t, std::set<std::pair<T, size_t>>>>
			hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k_at_layer(const vec<T>& v, size_t k,
																							size_t starting_point,
																							size_t layer);
	void add_edge(size_t layer, size_t i, size_t j);
	const std::vector<std::vector<size_t>>
	_query_k_internal(const vec<T>& v, size_t k, bool fill_all_layers = false);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	// const std::string _name() { return "HNSW Engine"; }
	const std::string _name() { return "HNSW Engine 2"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		return pl;
	}
};

template <typename T> void hnsw_engine_2<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
void hnsw_engine_2<T>::add_edge(size_t layer, size_t i, size_t j) {
	T d = dist2(all_entries[i], all_entries[j]);
	// TODO remove debug output below
	if (hadj[layer][i].size() < edge_count_mult ||
			hadj[layer][i].begin()->first < -d) {
		if (hadj[layer][i].size() < edge_count_mult)
			std::cerr << "\"Yes\" reason: size=" << hadj[layer][i].size()
								<< std::endl;
		else
			std::cerr << "\"Yes\" reason: ranks top="
								<< -hadj[layer][i].begin()->first << ", rank_val=" << d
								<< std::endl;
		std::cerr << "Adding edge: (layer=" << layer << ", " << i << "->" << j
							<< ", val=" << sqrt(d) << ")" << std::endl;
	} else {
		std::cerr << "\"No\" reason: size=" << hadj[layer][i].size()
							<< " and ranks top=" << -hadj[layer][i].begin()->first
							<< ", rank_val=" << d << std::endl;
		std::cerr << "Not adding edge: (layer=" << layer << ", " << i << "->" << j
							<< ", val=" << sqrt(d) << ")" << std::endl;
	}
	if (hadj[layer][j].size() < edge_count_mult ||
			hadj[layer][j].begin()->first < -d) {
		if (hadj[layer][j].size() < edge_count_mult)
			std::cerr << "\"Yes\" reason: size=" << hadj[layer][j].size()
								<< std::endl;
		else
			std::cerr << "\"Yes\" reason: ranks top="
								<< -hadj[layer][j].begin()->first << ", rank_val=" << d
								<< std::endl;
		std::cerr << "Adding edge: (layer=" << layer << ", " << j << "->" << i
							<< ", val=" << sqrt(d) << ")" << std::endl;
	} else {
		std::cerr << "\"No\" reason: size=" << hadj[layer][j].size()
							<< " and ranks top=" << -hadj[layer][j].begin()->first
							<< ", rank_val=" << d << std::endl;
		std::cerr << "Not adding edge: (layer=" << layer << ", " << j << "->" << i
							<< ", val=" << sqrt(d) << ")" << std::endl;
	}
	hadj[layer][i].emplace(-d, j);
	hadj[layer][j].emplace(-d, i);
	if (hadj[layer][i].size() > edge_count_mult)
		hadj[layer][i].erase(hadj[layer][i].begin());
	if (hadj[layer][j].size() > edge_count_mult)
		hadj[layer][j].erase(hadj[layer][j].begin());
}

template <typename T> void hnsw_engine_2<T>::_build() {
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
		// get kNN at each layer
		std::vector<std::vector<size_t>> kNN =
				_query_k_internal(all_entries[i], edge_count_mult, true);
		// get the layer this entry will go up to
		size_t cur_layer_ub =
				floor(-log(distribution(gen)) * 1 / log(double(edge_count_mult)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// if it is a new layer, add a layer
		while (cur_layer >= hadj.size())
			add_layer(i);
		// add all the neighbours as edges
		for (size_t layer = 0; layer <= cur_layer && layer < kNN.size(); ++layer)
			for (size_t j : kNN[layer]) {
				add_edge(layer, i, j);
			}
	}
}
template <typename T>
const std::vector<size_t>
hnsw_engine_2<T>::_query_k_at_layer(const vec<T>& v, size_t k,
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
	visit(dist2(v, all_entries[starting_point]), starting_point);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (auto& [_, u] : hadj[layer][cur]) {
			T d_next = dist2(v, all_entries[u]);
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
hnsw_engine_2<T>::_query_k_internal(const vec<T>& v, size_t k,
																		bool fill_all_layers) {
	auto current = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (!fill_all_layers && layer > 0)
			layer_k = 1;
		ret.push_back(_query_k_at_layer(v, layer_k, current, layer));
		current = ret.back().front();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> hnsw_engine_2<T>::_query_k(const vec<T>& v, size_t k) {
	auto ret = _query_k_internal(v, k * num_for_1nn, !quick_search)[0];
	ret.resize(std::min(k, ret.size()));
	return ret;
}
