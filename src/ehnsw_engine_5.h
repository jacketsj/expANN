#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

struct ehnsw_engine_5_config {
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t edge_count_search_factor;
	ehnsw_engine_5_config(size_t _edge_count_mult, size_t _num_for_1nn,
												size_t _edge_count_search_factor = 1)
			: edge_count_mult(_edge_count_mult), num_for_1nn(_num_for_1nn),
				edge_count_search_factor(_edge_count_search_factor) {}
};

template <typename T>
struct ehnsw_engine_5 : public ann_engine<T, ehnsw_engine_5<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t starting_vertex;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t edge_count_search_factor;
	const size_t num_cuts;
	ehnsw_engine_5(ehnsw_engine_5_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				edge_count_mult(conf.edge_count_mult), num_for_1nn(conf.num_for_1nn),
				edge_count_search_factor(conf.edge_count_search_factor),
				num_cuts(conf.edge_count_mult - 1) {}
	std::vector<vec<T>> all_entries;
	struct layer_data {
		std::vector<std::vector<size_t>>
				adj; // vertex -> cut -> outgoing_edge data_index
		std::vector<std::vector<std::tuple<T, size_t, size_t>>>
				edge_ranks; // vertex -> closest connected [distance, bin, edge_index]
		std::vector<vec<T>> vals;					 // vertex -> data
		std::vector<size_t> to_data_index; // vertex -> data_index
		robin_hood::unordered_flat_map<size_t, size_t>
				to_vertex; // data_index -> vertex
		std::vector<std::vector<bool>>
				e_labels; // vertex -> cut labels (*num_cuts=edge_count_mult-1)
		void add_vertex(size_t data_index, const vec<T>& data, size_t max_degree,
										std::function<bool()> generate_elabel) {
			to_vertex[data_index] = to_data_index.size();
			to_data_index.emplace_back(data_index);
			adj.emplace_back();
			edge_ranks.emplace_back();
			vals.emplace_back(data);

			e_labels.emplace_back();
			for (size_t cut = 0; cut + 1 < max_degree; ++cut)
				e_labels.back().emplace_back(generate_elabel());
		}
		bool is_valid_edge(size_t vertex_i, size_t vertex_j, size_t bin) {
			//  the last bin permits any edge (no cut)
			if (bin == e_labels[vertex_i].size())
				return true;
			//  an edge is permitted in a bin if it crosses the cut for that bin
			return e_labels[vertex_i][bin] != e_labels[vertex_j][bin];
		}
	};
	std::vector<layer_data> layers;
	std::vector<size_t> vertex_heights; // data_index -> max_height
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<std::pair<T, size_t>>
	_query_k_internal(const vec<T>& v, size_t k,
										const std::vector<size_t>& starting_points, size_t layer);
	size_t _query_1_internal(const vec<T>& v, size_t starting_point,
													 size_t layer);
	void add_edge(size_t i, size_t j, T d, size_t layer);
	void add_edge_directional(size_t i, size_t j, T d, size_t layer);
	const std::vector<std::vector<std::pair<T, size_t>>>
	_query_k_internal_wrapper(const vec<T>& v, size_t k,
														size_t full_search_top_layer);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "EHNSW Engine 5('simple')"; }

	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, edge_count_search_factor);
		return pl;
	}
	bool generate_elabel() { return int_distribution(gen); }
};

template <typename T> void ehnsw_engine_5<T>::_store_vector(const vec<T>& v) {
	size_t data_index = all_entries.size();
	all_entries.push_back(v);

	vertex_heights.emplace_back(
			size_t(floor(-log(distribution(gen)) * log(double(edge_count_mult)))));

	for (size_t layer = 0; layer <= vertex_heights[data_index]; ++layer) {
		if (layers.size() <= layer) {
			starting_vertex = data_index;
			layers.emplace_back();
		}
		size_t max_degree_layer = edge_count_mult;
		if (layer == 0)
			max_degree_layer *= 2;
		layers[layer].add_vertex(data_index, v, max_degree_layer,
														 [&]() { return generate_elabel(); });
	}
}

template <typename T>
void ehnsw_engine_5<T>::add_edge_directional(size_t i, size_t j, T d,
																						 size_t layer) {
	size_t max_node_size = edge_count_mult;
	// if (layer == 0)
	//	max_node_size *= 2;
	auto& edge_ranks = layers[layer].edge_ranks;
	auto& to_data_index = layers[layer].to_data_index;
	auto& adj = layers[layer].adj;
	// keep track of all bins that have been used already
	std::set<size_t> used_bins;
	// iterate through all the edges, smallest to biggest, while maintaining a
	// current edge. Greedily improve the currently visited bin each time.
	// don't bother if nothing will get modified
	if (edge_ranks[i].size() < max_node_size ||
			d < std::get<0>(edge_ranks[i].back())) {
		for (auto& [other_d, bin, edge_index] : edge_ranks[i]) {
			used_bins.insert(bin);
			if (j == adj[i][edge_index]) {
				// duplicate edge found, discard and return
				// will only happen if no swaps have occurred so far
				return;
			}
			if (d < other_d && layers[layer].is_valid_edge(i, j, bin)) {
				// (i,j) is a better edge than (i, adj[i][edge_index]) for the current
				// bin, so swap
				std::swap(j, adj[i][edge_index]);
				std::swap(d, other_d);
			}
		}
		// iterate through all unused bins, use one of them if it is compatible
		if (used_bins.size() < max_node_size)
			for (size_t bin = 0; bin < num_cuts + 1; ++bin)
				if (!used_bins.contains(bin) &&
						layers[layer].is_valid_edge(i, j, bin)) {
					size_t edge_index = adj[i].size();
					adj[i].emplace_back(j);
					edge_ranks[i].emplace_back(d, bin, edge_index);
					break;
				}
		// sort edge ranks by increasing distance again
		std::sort(edge_ranks[i].begin(), edge_ranks[i].end());
	}
}

template <typename T>
void ehnsw_engine_5<T>::add_edge(size_t i, size_t j, T d, size_t layer) {
	add_edge_directional(i, j, d, layer);
	add_edge_directional(j, i, d, layer);
}

template <typename T> void ehnsw_engine_5<T>::_build() {
	assert(all_entries.size() > 0);
	size_t op_count = 0;

	auto improve_vertex_edges = [&](size_t v) {
		// get current approx kNN
		std::vector<std::vector<std::pair<T, size_t>>> kNNs =
				_query_k_internal_wrapper(all_entries[v],
																	edge_count_mult * edge_count_search_factor,
																	layers.size() - 1);
		// add all the found neighbours as edges (if they are good)
		for (size_t layer = 0; layer < std::min(kNNs.size(), vertex_heights[v] + 1);
				 ++layer) {
			sort(kNNs[layer].begin(), kNNs[layer].end());
			size_t v_in_layer = layers[layer].to_vertex[v];
			for (auto [d, u_in_layer] : kNNs[layer]) {
				add_edge(v_in_layer, u_in_layer, d, layer);
			}
		}
	};

	for (size_t i = 0; i < all_entries.size(); ++i) {
		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;

		++op_count;
		improve_vertex_edges(i);
	}
}
template <typename T>
const std::vector<std::pair<T, size_t>>
ehnsw_engine_5<T>::_query_k_internal(const vec<T>& v, size_t k,
																		 const std::vector<size_t>& starting_points,
																		 size_t layer) {
	auto& adj = layers[layer].adj;
	auto& vals = layers[layer].vals;
	// auto& to_data_index = layers[layer].to_data_index;
	static auto compare = [](const std::pair<T, size_t>& a,
													 const std::pair<T, size_t>& b) {
		return a.first < b.first;
	};
	std::priority_queue<std::pair<T, size_t>, std::vector<std::pair<T, size_t>>,
											decltype(compare)>
			top_k(compare);
	std::priority_queue<std::pair<T, size_t>, std::vector<std::pair<T, size_t>>,
											decltype(compare)>
			to_visit(compare);
	robin_hood::unordered_flat_map<size_t, T> visited;

	auto visit = [&](T d, size_t u) {
		bool is_good =
				!visited.contains(u) && (top_k.size() < k || top_k.top().first > d);
		visited[u] = d;
		if (is_good) {
			top_k.emplace(d, u);		 // top_k is a max heap
			to_visit.emplace(-d, u); // to_visit is a min heap
		}
		if (top_k.size() > k)
			top_k.pop();
		return is_good;
	};
	for (const auto& sp : starting_points)
		// visit(dist2(v, vals[sp]), sp);
		visit(dist2fast(v, vals[sp]), sp);
	// visit(dist2fast(v, all_entries[to_data_index[sp]]), sp);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		_mm_prefetch(&adj[cur], _MM_HINT_T0);
		for (const auto& u : adj[cur]) {
			_mm_prefetch(&vals[u], _MM_HINT_T0);
		}
		for (const auto& u : adj[cur]) {
			T d_next = dist2fast(v, vals[u]);
			// T d_next = dist2(v, vals[u]);
			// T d_next = dist2fast(v, all_entries[to_data_index[u]]);
			visit(d_next, u);
		}
	}
	std::vector<std::pair<T, size_t>> ret;
	while (!top_k.empty()) {
		ret.push_back(top_k.top());
		top_k.pop();
	}
	reverse(ret.begin(), ret.end()); // sort from closest to furthest
	return ret;
}

template <typename T>
const std::vector<std::vector<std::pair<T, size_t>>>
ehnsw_engine_5<T>::_query_k_internal_wrapper(const vec<T>& v, size_t k,
																						 size_t full_search_top_layer) {
	auto current = starting_vertex;
	std::vector<std::vector<std::pair<T, size_t>>> ret;
	// for each layer, in decreasing depth
	for (int layer = layers.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (layer > int(full_search_top_layer))
			layer_k = 1;
		if (layer == 0)
			layer_k *= 2;
		ret.emplace_back(_query_k_internal(
				v, layer_k, {layers[layer].to_vertex[current]}, layer));
		current = layers[layer].to_data_index[ret.back().front().second];
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
size_t ehnsw_engine_5<T>::_query_1_internal(const vec<T>& v,
																						size_t starting_point,
																						size_t layer) {
	auto& adj = layers[layer].adj;
	auto& vals = layers[layer].vals;
	// auto& to_data_index = layers[layer].to_data_index;
	size_t best = starting_point;
	// T d = dist2fast(v, all_entries[starting_point]);
	T d = dist2(v, all_entries[starting_point]);
	bool changed = true;
	while (changed) {
		changed = false;
		for (const auto& u : adj[best]) {
			// T d_next = dist2fast(v, vals[u]);
			T d_next = dist2(v, vals[u]);
			// T d_next = dist2fast(v, all_entries[to_data_index[u]]);
			if (d_next < d) {
				changed = true;
				d = d_next;
				best = u;
			}
		}
	}
	return best;
}

template <typename T>
std::vector<size_t> ehnsw_engine_5<T>::_query_k(const vec<T>& v, size_t k) {
	auto current = starting_vertex;
	for (int layer = layers.size() - 1; layer > 0; --layer) {
		current = layers[layer].to_data_index[_query_1_internal(
				v, layers[layer].to_vertex[current], layer)];
	}
	auto ret_combined =
			_query_k_internal(v, k * num_for_1nn, {layers[0].to_vertex[current]}, 0);

	/*
	auto ret_combined = _query_k_internal_wrapper(v, k * num_for_1nn, 0)[0];
	*/
	ret_combined.resize(std::min(k, ret_combined.size()));
	auto ret = std::vector<size_t>(ret_combined.size());
	for (size_t i = 0; i < ret.size(); ++i)
		ret[i] = ret_combined[i].second;
	return ret;
}
