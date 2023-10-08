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

struct ehnsw_engine_4_config {
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t max_depth;
	float re_improve_wait_ratio;
	bool include_visited_during_build;
	bool run_improves;
	bool cut_off_visited_if_long;
	size_t cut_off_visited_if_long_ratio;
	bool include_visited_only_higher;
	ehnsw_engine_4_config(size_t _edge_count_mult, size_t _num_for_1nn,
												size_t _max_depth = 100,
												float _re_improve_wait_ratio = 1.0f,
												bool _include_visited_during_build = true,
												bool _run_improves = true,
												bool _cut_off_visited_if_long = false,
												size_t _cut_off_visited_if_long_ratio = 4,
												bool _include_visited_only_higher = true)
			: edge_count_mult(_edge_count_mult), num_for_1nn(_num_for_1nn),
				max_depth(_max_depth), re_improve_wait_ratio(_re_improve_wait_ratio),
				include_visited_during_build(_include_visited_during_build),
				run_improves(_run_improves),
				cut_off_visited_if_long(_cut_off_visited_if_long),
				cut_off_visited_if_long_ratio(_cut_off_visited_if_long_ratio),
				include_visited_only_higher(_include_visited_only_higher) {}
};

template <typename T>
struct ehnsw_engine_4 : public ann_engine<T, ehnsw_engine_4<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t starting_vertex;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t max_depth;
	float re_improve_wait_ratio;
	bool include_visited_during_build;
	bool run_improves;
	bool cut_off_visited_if_long;
	size_t cut_off_visited_if_long_ratio;
	bool include_visited_only_higher;
	const size_t num_cuts;
	ehnsw_engine_4(ehnsw_engine_4_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				edge_count_mult(conf.edge_count_mult), num_for_1nn(conf.num_for_1nn),
				max_depth(conf.max_depth),
				re_improve_wait_ratio(conf.re_improve_wait_ratio),
				include_visited_during_build(conf.include_visited_during_build),
				run_improves(conf.run_improves),
				cut_off_visited_if_long(conf.cut_off_visited_if_long),
				cut_off_visited_if_long_ratio(conf.cut_off_visited_if_long_ratio),
				include_visited_only_higher(conf.include_visited_only_higher),
				num_cuts(conf.edge_count_mult - 1) {}
	std::vector<vec<T>> all_entries;
	// TODO make these vectors, not hash maps
	struct layer_data {
		robin_hood::unordered_flat_map<size_t, std::vector<size_t>>
				adj; // vertex -> cut -> outgoing_edge data_index
		robin_hood::unordered_flat_map<size_t,
																	 std::vector<std::tuple<T, size_t, size_t>>>
				edge_ranks; // vertex -> closest connected [distance, bin, edge_index]
	};
	std::vector<layer_data> layers;
	robin_hood::unordered_flat_map<size_t, size_t>
			vertex_heights; // vertex -> max_height
	robin_hood::unordered_flat_map<size_t, std::vector<bool>>
			e_labels; // vertex -> cut labels (*num_cuts=edge_count_mult-1)
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<std::pair<T, size_t>>
	_query_k_internal(const vec<T>& v, size_t k,
										const std::vector<size_t>& starting_points,
										bool include_visited, size_t layer);
	bool is_valid_edge(size_t i, size_t j, size_t bin);
	bool add_edge(size_t i, size_t j, T d, size_t layer);
	bool add_edge_directional(size_t i, size_t j, T d, size_t layer);
	const std::vector<std::vector<std::pair<T, size_t>>>
	_query_k_internal_wrapper(const vec<T>& v, size_t k, bool include_visited,
														size_t full_search_top_layer);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "EHNSW Engine 4"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, re_improve_wait_ratio);
		add_param(pl, max_depth);
		add_param(pl, include_visited_during_build);
		add_param(pl, run_improves);
		add_param(pl, cut_off_visited_if_long);
		add_param(pl, cut_off_visited_if_long_ratio);
		add_param(pl, include_visited_only_higher);
		return pl;
	}
	bool generate_elabel() { return int_distribution(gen); }
};

template <typename T> void ehnsw_engine_4<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
bool ehnsw_engine_4<T>::is_valid_edge(size_t i, size_t j, size_t bin) {
	// the last bin permits any edge
	if (bin == num_cuts)
		return true;
	// if (bin <= vertex_heights[i]) {
	//	return bin <= vertex_heights[j] && e_labels[i][bin] != e_labels[j][bin];
	// }
	//  an edge is permitted in a bin if it crosses the cut for that bin
	return e_labels[i][bin] != e_labels[j][bin];
}

template <typename T>
bool ehnsw_engine_4<T>::add_edge_directional(size_t i, size_t j, T d,
																						 size_t layer) {
	auto& edge_ranks = layers[layer].edge_ranks;
	auto& adj = layers[layer].adj;
	// keep track of all bins that have been used already
	std::set<size_t> used_bins;
	// iterate through all the edges, smallest to biggest, while maintaining a
	// current edge. Greedily improve the currently visited bin each time.
	// don't bother if nothing will get modified
	if (edge_ranks[i].size() < edge_count_mult ||
			d < std::get<0>(edge_ranks[i].back())) {
		for (auto& [other_d, bin, edge_index] : edge_ranks[i]) {
			used_bins.insert(bin);
			if (j == adj[i][edge_index]) {
				// duplicate edge found, discard and return
				// will only happen if no swaps have occurred so far
				return true;
			}
			if (d < other_d && is_valid_edge(i, j, bin)) {
				// (i,j) is a better edge than (i, adj[i][edge_index]) for the current
				// bin, so swap
				std::swap(j, adj[i][edge_index]);
				std::swap(d, other_d);
			}
		}
		// iterate through all unused bins, use one of them if it is compatible
		if (used_bins.size() < edge_count_mult)
			for (size_t bin = 0; bin < num_cuts + 1; ++bin)
				if (!used_bins.contains(bin) && is_valid_edge(i, j, bin)) {
					size_t edge_index = adj[i].size();
					adj[i].emplace_back(j);
					edge_ranks[i].emplace_back(d, bin, edge_index);
					break;
				}
		// sort edge ranks by increasing distance again
		std::sort(edge_ranks[i].begin(), edge_ranks[i].end());
		return true;
	} else {
		// returns false if d > all outgoing edge lengths from i
		return false;
	}
}

template <typename T>
bool ehnsw_engine_4<T>::add_edge(size_t i, size_t j, T d, size_t layer) {
	add_edge_directional(j, i, d, layer);
	// returns false if d > all outgoing edge lengths from i
	return add_edge_directional(i, j, d, layer);
}

template <typename T> void ehnsw_engine_4<T>::_build() {
	assert(all_entries.size() > 0);
	size_t op_count = 0;
	size_t& total_size = op_count; // TODO change this for dynamic version

	auto add_vertex_base = [&](size_t v) {
		if (v != 0) // TODO come up with a better bugfix solution
			vertex_heights[v] = std::min(
					size_t(floor(-log(distribution(gen)) / log(double(edge_count_mult)))),
					num_cuts - 1);

		for (size_t layer = 0; layer <= vertex_heights[v]; ++layer)
			layers[layer].edge_ranks[v] =
					std::vector<std::tuple<T, size_t, size_t>>();

		for (size_t cut = 0; cut < num_cuts; ++cut)
			e_labels[v].emplace_back(generate_elabel());
		++op_count;
	};
	size_t highest_value = num_cuts - 1;
	layers.resize(highest_value + 1);
	starting_vertex = 0;
	vertex_heights[0] = highest_value; // make sure vertex 0 is on top level

	layers.resize(num_cuts - 1);

	std::queue<std::pair<size_t, size_t>> improve_queue;

	auto improve_vertex_edges = [&](size_t v) {
		// get current approx kNN
		std::vector<std::vector<std::pair<T, size_t>>> kNNs =
				_query_k_internal_wrapper(all_entries[v], edge_count_mult,
																	include_visited_during_build, highest_value);
		// add all the found neighbours as edges (if they are good)
		for (size_t layer = 0; layer < std::min(kNNs.size(), vertex_heights[v] + 1);
				 ++layer) {
			sort(kNNs[layer].begin(), kNNs[layer].end());
			for (size_t neighbour_index = 0; neighbour_index < kNNs[layer].size();
					 ++neighbour_index) {
				auto [d, u] = kNNs[layer][neighbour_index];
				if (neighbour_index <=
								edge_count_mult * cut_off_visited_if_long_ratio ||
						(!include_visited_only_higher || vertex_heights[u] > layer)) {
					if (!add_edge(v, u, d, layer) && cut_off_visited_if_long &&
							neighbour_index > cut_off_visited_if_long_ratio * edge_count_mult)
						break;
				}
			}
			// for (auto [d, u] : kNNs[layer]) {
			//	add_edge(v, u, d, layer);
			// }
		}
		if (run_improves)
			improve_queue.emplace(
					op_count + size_t(re_improve_wait_ratio * total_size) + 1, v);
	};

	for (size_t i = 0; i < all_entries.size(); ++i) {
		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;

		add_vertex_base(i);
		improve_vertex_edges(i);
		while (improve_queue.front().first < op_count) {
			improve_vertex_edges(improve_queue.front().second);
			improve_queue.pop();
		}
		// TODO for dynamic version, keep a random sample point as a starting point
		// (i.e. if existing starting point is deleted, sample another)
		// potentially use a few different starting points, for better sampling
	}
	// TODO test this, does improve querytime-recall curve/does it slow down build
	// times?
	// TODO this will run infinitely, need to fix it so that it doesn't
	// re-enqueue, or so that it terminates
	//
	// while (!improve_queue.empty()) {
	//	improve_vertex_edges(improve_queue.front().second);
	//	improve_queue.pop();
	//}
}
template <typename T>
const std::vector<std::pair<T, size_t>>
ehnsw_engine_4<T>::_query_k_internal(const vec<T>& v, size_t k,
																		 const std::vector<size_t>& starting_points,
																		 bool include_visited, size_t layer) {
	auto& adj = layers[layer].adj;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::priority_queue<std::pair<T, size_t>> to_visit;
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
		visit(dist(v, all_entries[sp]), sp);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (const auto& u : adj[cur]) {
			T d_next = dist(v, all_entries[u]);
			visit(d_next, u);
		}
	}
	std::vector<std::pair<T, size_t>> ret;
	while (!top_k.empty()) {
		ret.push_back(top_k.top());
		top_k.pop();
	}
	if (include_visited) {
		ret.clear();
		for (auto [node, d] : visited)
			ret.emplace_back(d, node);
	}
	reverse(ret.begin(), ret.end()); // sort from closest to furthest
	return ret;
}

template <typename T>
const std::vector<std::vector<std::pair<T, size_t>>>
ehnsw_engine_4<T>::_query_k_internal_wrapper(const vec<T>& v, size_t k,
																						 bool include_visited,
																						 size_t full_search_top_layer) {
	// return _query_k_internal(v, k, {starting_vertex}, include_visited);
	auto current = starting_vertex;
	std::vector<std::vector<std::pair<T, size_t>>> ret;
	// for each layer, in decreasing depth
	for (int layer = layers.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (layer > int(full_search_top_layer))
			layer_k = 1;
		ret.emplace_back(
				_query_k_internal(v, layer_k, {current}, include_visited, layer));
		current = ret.back().front().second;
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> ehnsw_engine_4<T>::_query_k(const vec<T>& v, size_t k) {
	auto ret_combined =
			_query_k_internal_wrapper(v, k * num_for_1nn, false, 0)[0];
	ret_combined.resize(std::min(k, ret_combined.size()));
	auto ret = std::vector<size_t>(ret_combined.size());
	for (size_t i = 0; i < ret.size(); ++i)
		ret[i] = ret_combined[i].second;
	return ret;
}
