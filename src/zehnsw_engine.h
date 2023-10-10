#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

struct zehnsw_engine_config {
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t edge_count_search_factor;
	zehnsw_engine_config(size_t _edge_count_mult, size_t _num_for_1nn,
											 size_t _edge_count_search_factor = 1)
			: edge_count_mult(_edge_count_mult), num_for_1nn(_num_for_1nn),
				edge_count_search_factor(_edge_count_search_factor) {}
};

size_t useless = 0;
size_t total_comps = 0;

template <typename T>
struct zehnsw_engine : public ann_engine<T, zehnsw_engine<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t starting_vertex;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t edge_count_search_factor;
	const size_t num_cuts;
	zehnsw_engine(zehnsw_engine_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				edge_count_mult(conf.edge_count_mult), num_for_1nn(conf.num_for_1nn),
				edge_count_search_factor(conf.edge_count_search_factor),
				num_cuts(conf.edge_count_mult - 1) {}
	std::vector<vec<T>> all_entries;
	struct layer_data {
		std::vector<std::vector<std::pair<T, size_t>>>
				adj; // vertex -> list of outgoing edges, sorted by increasing distance
		std::vector<std::vector<size_t>> edge_bins; // vertex -> edge_index -> bin
		std::vector<vec<T>> vals;										// vertex -> data
		std::vector<size_t> to_data_index;					// vertex -> data_index
		robin_hood::unordered_flat_map<size_t, size_t>
				to_vertex; // data_index -> vertex
		void add_vertex(size_t data_index, const vec<T>& data) {
			to_vertex[data_index] = to_data_index.size();
			to_data_index.emplace_back(data_index);
			adj.emplace_back();
			edge_bins.emplace_back();
			vals.emplace_back(data);
		}
	};
	std::vector<layer_data> layers;
	std::vector<size_t> vertex_heights; // data_index -> max_height
	std::vector<std::vector<bool>>
			e_labels; // data_index -> cut labels (*num_cuts=edge_count_mult-1)
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<std::pair<T, size_t>>
	_query_k_internal(const vec<T>& v, size_t k,
										const std::vector<size_t>& starting_points, size_t layer);
	size_t _query_1_internal(const vec<T>& v, size_t starting_point,
													 size_t layer);
	bool is_valid_edge(size_t i, size_t j, size_t bin);
	void add_edge(size_t i, size_t j, T d, size_t layer);
	void add_edge_directional(size_t i, size_t j, T d, size_t layer);
	const std::vector<std::vector<std::pair<T, size_t>>>
	_query_k_internal_wrapper(const vec<T>& v, size_t k,
														size_t full_search_top_layer);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "ZEHNSW Engine"; }

	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, edge_count_search_factor);
		return pl;
	}
	bool generate_elabel() { return int_distribution(gen); }
	~zehnsw_engine() {
		std::cout << "count: " << useless << "/" << total_comps << std::endl;
	}
};

template <typename T> void zehnsw_engine<T>::_store_vector(const vec<T>& v) {
	size_t data_index = all_entries.size();
	all_entries.push_back(v);

	vertex_heights.emplace_back(
			data_index == 0 ? num_cuts - 1
											: std::min(size_t(floor(-log(distribution(gen)) /
																							log(double(edge_count_mult)))),
																 num_cuts - 1));

	for (size_t layer = 0; layer <= vertex_heights[data_index]; ++layer) {
		if (layers.size() <= layer) {
			starting_vertex = data_index;
			layers.emplace_back();
		}
		layers[layer].add_vertex(data_index, v);
	}

	e_labels.emplace_back();
	for (size_t cut = 0; cut < num_cuts; ++cut)
		e_labels[data_index].emplace_back(generate_elabel());
}

// TODO try e_labels specific to each layer and see if it helps
template <typename T>
bool zehnsw_engine<T>::is_valid_edge(size_t i, size_t j, size_t bin) {
	//  the last bin permits any edge (no cut)
	if (bin == num_cuts)
		return true;
	//  an edge is permitted in a bin if it crosses the cut for that bin
	return e_labels[i][bin] != e_labels[j][bin];
}

template <typename T>
void zehnsw_engine<T>::add_edge_directional(size_t i, size_t j, T d,
																						size_t layer) {
	// TODO do double-bottom with cuts across entire bottom
	size_t max_node_size = edge_count_mult;
	// if (layer == 0)
	//	max_node_size *= 2;
	auto& edge_bins = layers[layer].edge_bins;
	auto& to_data_index = layers[layer].to_data_index;
	auto& adj = layers[layer].adj;
	// keep track of all bins that have been used already
	std::set<size_t> used_bins;
	// iterate through all the edges, smallest to biggest, while maintaining a
	// current edge. Greedily improve the currently visited bin each time.
	// don't bother if nothing will get modified
	if (adj[i].size() < max_node_size || d < std::get<0>(adj[i].back())) {
		for (size_t edge_index = 0; edge_index < adj[i].size(); ++edge_index) {
			auto& bin = edge_bins[i][edge_index];
			if (j == adj[i][edge_index].second) {
				// duplicate edge found, discard and return
				// will only happen if no swaps have occurred so far
				return;
			}
			if (d < adj[i][edge_index].first &&
					is_valid_edge(to_data_index[i], to_data_index[j], bin)) {
				std::swap(j, adj[i][edge_index].second);
				std::swap(d, adj[i][edge_index].first);
			}
		}
		// iterate through all unused bins, use one of them if it is compatible
		if (used_bins.size() < max_node_size)
			for (size_t bin = 0; bin < num_cuts + 1; ++bin)
				// TODO there was a bug in the following line in the old version (wrong
				// indexing), need to go back and test that
				if (!used_bins.contains(bin) &&
						is_valid_edge(to_data_index[i], to_data_index[j], bin)) {
					// is_valid_edge(i, j, bin)) {
					adj[i].emplace_back(d, j);
					edge_bins[i].emplace_back(bin);
					break;
				}
		// sort edge ranks by increasing distance again (while keeping bins)
		// C++23 version (unsupported by gcc 11.2)
		// std::ranges::sort(
		//		std::views::zip(adj[i], edge_bins[i]),
		//		[](auto&& a, auto&& b) { return std::get<0>(a) < std::get<0>(b); });
		std::vector<std::pair<decltype(adj[i][0]), decltype(edge_bins[i][0])>>
				zipped;
		for (size_t edge_index = 0; edge_index < adj[i].size(); ++edge_index) {
			zipped.emplace_back(adj[i][edge_index], edge_bins[i][edge_index]);
		}
		std::sort(zipped.begin(), zipped.end());
		for (size_t edge_index = 0; edge_index < adj[i].size(); ++edge_index) {
			adj[i][edge_index] = zipped[edge_index].first;
			edge_bins[i][edge_index] = zipped[edge_index].second;
		}
	}
}

template <typename T>
void zehnsw_engine<T>::add_edge(size_t i, size_t j, T d, size_t layer) {
	add_edge_directional(i, j, d, layer);
	add_edge_directional(j, i, d, layer);
}

template <typename T> void zehnsw_engine<T>::_build() {
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
zehnsw_engine<T>::_query_k_internal(const vec<T>& v, size_t k,
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
		// TODO modify this code so that it makes use of the sorted distances and
		// prunes
		const T& d_worst = top_k.top().first;
		_mm_prefetch(&adj[cur], _MM_HINT_T0);
		/*
		// auto it_low = std::lower_bound(
		//		adj[cur].begin(), adj[cur].end(), nd - d_worst,
		//		[](const auto& p, float val) { return p.first < val; });
		// auto it_high =
		//		std::upper_bound(adj[cur].begin(), adj[cur].end(), nd + d_worst,
		//										 [](float val, auto& p) { return val < p.first; });
		auto it_low = adj[cur].begin();
		auto it_high = adj[cur].end();
		std::for_each(it_low, it_high, [&](auto& neighbour) {
			auto& u = neighbour.second;
			_mm_prefetch(&vals[u], _MM_HINT_T0);
		});
		std::for_each(it_low, it_high, [&](auto& neighbour) {
			auto& u = neighbour.second;
			T d_next = dist2fast(v, vals[u]);
			visit(d_next, u);
			++total_comps;
		});
		*/
		for (const auto& [_, u] : adj[cur]) {
			_mm_prefetch(&vals[u], _MM_HINT_T0);
		}
		for (const auto& [_, u] : adj[cur]) {
			T d_next = dist2fast(v, vals[u]);

			// T d_star = dist2fast(vals[u], vals[cur]);
			// T d_worst = top_k.top().first;
			// if (std::max(nd, d_star) - std::min(nd, d_star) > d_worst) {
			//	// d_next is a useless distance computation
			//	++useless;
			// }
			++total_comps;

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
zehnsw_engine<T>::_query_k_internal_wrapper(const vec<T>& v, size_t k,
																						size_t full_search_top_layer) {
	auto current = starting_vertex;
	std::vector<std::vector<std::pair<T, size_t>>> ret;
	// for each layer, in decreasing depth
	for (int layer = layers.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (layer > int(full_search_top_layer))
			layer_k = 1;
		ret.emplace_back(_query_k_internal(
				v, layer_k, {layers[layer].to_vertex[current]}, layer));
		current = layers[layer].to_data_index[ret.back().front().second];
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
size_t zehnsw_engine<T>::_query_1_internal(const vec<T>& v,
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
		for (const auto& [_, u] : adj[best]) {
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
std::vector<size_t> zehnsw_engine<T>::_query_k(const vec<T>& v, size_t k) {
	// useless = 0;
	// total_comps = 0;
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
