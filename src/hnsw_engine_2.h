#pragma once

#include <algorithm>
#include <map>
#include <queue>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

// basic lsh method
template <typename T>
struct hnsw_engine_2 : public ann_engine<T, hnsw_engine_2<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::geometric_distribution<> d;
	size_t max_depth;
	size_t edge_count_mult;
	double coinflip_p;
	size_t starting_vertex;
	hnsw_engine_2(size_t _max_depth, size_t _edge_count_mult, double _coinflip_p)
			: rd(), gen(rd()), d(_coinflip_p), max_depth(_max_depth),
				edge_count_mult(_edge_count_mult), coinflip_p(_coinflip_p) {}
	std::vector<vec<T>> all_entries;
	std::vector<robin_hood::unordered_flat_map<size_t, std::vector<size_t>>> hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<std::vector<size_t>> _query_k(const vec<T>& v, size_t k,
																									bool fill_all_layers = false);
	const vec<T>& _query(const vec<T>& v);
	// const std::string _name() { return "HNSW Engine"; }
	const std::string _name() { return "HNSW Engine 2"; }
	const std::string _name_long() {
		return "HNSW Engine 2 (p=" + std::to_string(coinflip_p) +
					 ",edge_count_mult=" + std::to_string(edge_count_mult) + ")";
	}
};

template <typename T> void hnsw_engine_2<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void hnsw_engine_2<T>::_build() {
	assert(all_entries.size() > 0);
	auto add_layer = [&](size_t v) {
		hadj.emplace_back();
		hadj.back()[v] = std::vector<size_t>();
		starting_vertex = v;
	};
	// add one layer to start, with first vertex
	add_layer(0);
	for (size_t i = 1; i < all_entries.size(); ++i) {
		// get kNN at each layer
		std::vector<std::vector<size_t>> kNN =
				_query_k(all_entries[i], edge_count_mult, true);
		// get the layer this entry will go up to
		size_t cur_layer = std::min(size_t(d(gen)), max_depth);
		// if it is a new layer, add a layer
		if (cur_layer >= hadj.size())
			add_layer(i);
		// add all the neighbours as edges
		for (size_t layer = 0; layer <= cur_layer && layer < kNN.size(); ++layer)
			for (size_t j : kNN[layer]) {
				hadj[layer][i].push_back(j);
				hadj[layer][j].push_back(i);
			}
	}
}

template <typename T>
const std::vector<std::vector<size_t>>
hnsw_engine_2<T>::_query_k(const vec<T>& v, size_t k, bool fill_all_layers) {
	size_t cur = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret(1);
	if (fill_all_layers)
		ret.resize(hadj.size());
	auto visit = [&](size_t item) {
		T d = dist(all_entries[item], v);
		if (top_k.size() < k) {
			top_k.emplace(d, item);
		}
		if (d < top_k.top().first) {
			top_k.pop();
			top_k.emplace(d, item);
		}
	};
	// for each layer, in decreasing depth
	for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		// find the best vertex in the current layer by local search
		bool improvement_found = false;
		do {
			improvement_found = false;
			// otherwise look at all incident edges first
			T best_dist2 = dist2(all_entries[cur], v);
			size_t best = cur;
			visit(cur);
			for (size_t adj_vert : hadj[layer][cur]) {
				visit(adj_vert);
				T next_dist2 = dist2(all_entries[adj_vert], v);
				if (next_dist2 < best_dist2) {
					improvement_found = true;
					best = adj_vert;
					best_dist2 = next_dist2;
				}
			}
			cur = best;
		} while (improvement_found);
		// push best k vertices at this layer into ret, if it is the last layer or
		// fill_all_layers is active
		if (layer == 0 || fill_all_layers) {
			auto top_k_dupe = top_k;
			while (!top_k_dupe.empty()) {
				ret[layer].push_back(top_k_dupe.top().second);
				top_k_dupe.pop();
			}
		}
	}
	return ret;
}

template <typename T> const vec<T>& hnsw_engine_2<T>::_query(const vec<T>& v) {
	return all_entries[_query_k(v, 1)[0][0]];
}
