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

struct ehnsw_engine_6_config {
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t edge_count_search;
	bool extend_to_neighbours;
	ehnsw_engine_6_config(size_t _edge_count_mult, size_t _num_for_1nn,
												size_t _edge_count_search,
												bool _extend_to_neighbours = true)
			: edge_count_mult(_edge_count_mult), num_for_1nn(_num_for_1nn),
				edge_count_search(_edge_count_search),
				extend_to_neighbours(_extend_to_neighbours) {
		assert(edge_count_search >= edge_count_mult);
	}
};

template <typename T>
struct ehnsw_engine_6 : public ann_engine<T, ehnsw_engine_6<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t starting_vertex;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t edge_count_search;
	bool extend_to_neighbours;
	ehnsw_engine_6(ehnsw_engine_6_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				edge_count_mult(conf.edge_count_mult), num_for_1nn(conf.num_for_1nn),
				edge_count_search(conf.edge_count_search),
				extend_to_neighbours(conf.extend_to_neighbours) {}
	std::vector<vec<T>> all_entries;
	struct layer_data {
		std::vector<std::vector<size_t>>
				adj;									// vertex -> cut -> outgoing_edge data_index
		std::vector<vec<T>> vals; // vertex -> data
		std::vector<size_t> to_data_index; // vertex -> data_index
		robin_hood::unordered_flat_map<size_t, size_t>
				to_vertex;													 // data_index -> vertex
		std::vector<std::vector<bool>> e_labels; // vertex -> cut labels (*num_cuts)
		size_t num_cuts() { return e_labels[0].size(); }
		void add_vertex(size_t data_index, const vec<T>& data, size_t max_degree,
										std::function<bool()> generate_elabel) {
			to_vertex[data_index] = to_data_index.size();
			to_data_index.emplace_back(data_index);
			adj.emplace_back();
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
	void improve_vertex_edges(size_t v);
	void _build();
	const std::vector<std::pair<T, size_t>>
	_query_k_internal(const vec<T>& v, size_t k,
										const std::vector<size_t>& starting_points, size_t layer);
	size_t _query_1_internal(const vec<T>& v, size_t starting_point,
													 size_t layer);
	void add_edges(size_t from, std::vector<std::pair<T, size_t>> to,
								 size_t layer);
	const std::vector<std::vector<std::pair<T, size_t>>>
	_query_k_internal_wrapper(const vec<T>& v, size_t k,
														size_t full_search_top_layer);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "EHNSW Engine 6"; }

	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, edge_count_search);
		add_param(pl, extend_to_neighbours);
		return pl;
	}
	bool generate_elabel() { return int_distribution(gen); }
};

template <typename T> void ehnsw_engine_6<T>::_store_vector(const vec<T>& v) {
	size_t data_index = all_entries.size();
	all_entries.push_back(v);

	vertex_heights.emplace_back(
			size_t(floor(-log(distribution(gen)) / log(double(edge_count_mult)))));

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

		// TODO consider starting with a random graph
	}
}

template <typename T>
void ehnsw_engine_6<T>::add_edges(size_t from,
																	std::vector<std::pair<T, size_t>> to,
																	size_t layer) {
	size_t max_degree = edge_count_mult;
	if (layer == 0)
		max_degree *= 2;

	// TODO consider extending by neighbours
	auto& neighbours = layers[layer].adj[from];
	if (extend_to_neighbours) {
		// TODO ...
		assert(false); // not implemented
	}
	sort(to.begin(), to.end());
	std::set<size_t> available_bins;
	for (size_t bin = 0; bin < layers[layer].num_cuts() + 1; ++bin)
		available_bins.insert(bin);
	std::queue<size_t> discard_queue;
	for (const auto& [to_d, to_vert] : to) {
		int chosen_bin = 0;
		bool include = false;
		// prune based on cuts
		for (size_t bin : available_bins) {
			if (layers[layer].is_valid_edge(from, to_vert, bin)) {
				chosen_bin = bin;
				include = true;
				break;
			}
		}
		if (include) {
			// prune based on distance
			// TODO consider better (graph-based? higher-dimensional?)
			// anti-clique methods
			for (const size_t& chosen_vert : neighbours) {
				if (dist2(vals[chosen_vert], vals[to_vert]) < to_d) {
					include = false;
					break;
				}
			}
		}
		if (include) {
			neighbours.emplace_back(to_vert);
		} else {
			discard_queue.emplace_back(to_vert);
		}
		if (neighbours.size() >= max_degree)
			break;
	}
	while (neighbours.size() < max_degree) {
		neighbours.emplace_back(discard_queue.top());
		discard_queue.pop();
	}

	// TODO consider keeping candidate lists stored for other vertices, and adding
	// reverse edges that way (later) --- probably not a good idea, since they can
	// be found later when doing an improve call
}

template <typename T> void ehnsw_engine_6<T>::improve_vertex_edges(size_t v) {
	// get current approx kNN
	std::vector<std::vector<std::pair<T, size_t>>> kNNs =
			_query_k_internal_wrapper(all_entries[v], edge_count_search,
																layers.size() - 1);
	// add all the found neighbours as edges (if they are good)
	for (size_t layer = 0; layer < std::min(kNNs.size(), vertex_heights[v] + 1);
			 ++layer) {
		sort(kNNs[layer].begin(), kNNs[layer].end());
		add_edges(v, kNNs[layer], layer);
	}
}

template <typename T> void ehnsw_engine_6<T>::_build() {
	assert(all_entries.size() > 0);
	size_t op_count = 0;

	// TODO use a random permutation
	for (size_t i = 0; i < all_entries.size(); ++i) {
		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;

		++op_count;
		improve_vertex_edges(i);
	}
	// TODO consider using a second pass
}

template <typename T>
const std::vector<std::pair<T, size_t>>
ehnsw_engine_6<T>::_query_k_internal(const vec<T>& v, size_t k,
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
		visit(dist2(v, vals[sp]), sp);
	// visit(dist2fast(v, vals[sp]), sp);
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
			// T d_next = dist2fast(v, vals[u]);
			T d_next = dist2(v, vals[u]);
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
ehnsw_engine_6<T>::_query_k_internal_wrapper(const vec<T>& v, size_t k,
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
size_t ehnsw_engine_6<T>::_query_1_internal(const vec<T>& v,
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
std::vector<size_t> ehnsw_engine_6<T>::_query_k(const vec<T>& v, size_t k) {
	auto current = starting_vertex;
	for (int layer = layers.size() - 1; layer > 0; --layer) {
		current = layers[layer].to_data_index[_query_1_internal(
				v, layers[layer].to_vertex[current], layer)];
	}
	auto ret_combined =
			_query_k_internal(v, k * num_for_1nn, {layers[0].to_vertex[current]}, 0);

	ret_combined.resize(std::min(k, ret_combined.size()));
	auto ret = std::vector<size_t>(ret_combined.size());
	for (size_t i = 0; i < ret.size(); ++i)
		ret[i] = ret_combined[i].second;
	return ret;
}
