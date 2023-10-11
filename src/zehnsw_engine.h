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

struct zehnsw_engine_config {
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t edge_count_search_factor;
	zehnsw_engine_config(size_t _edge_count_mult, size_t _num_for_1nn,
											 size_t _edge_count_search_factor = 1)
			: edge_count_mult(_edge_count_mult), num_for_1nn(_num_for_1nn),
				edge_count_search_factor(_edge_count_search_factor) {}
};

size_t counterr = 0;
size_t counterr2 = 0;
size_t total = 0;

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
	~zehnsw_engine();
	std::vector<vec<T>> all_entries;
	struct layer_data {
		std::vector<std::vector<std::pair<T, size_t>>>
				adj; // vertex -> cut -> outgoing_edge data_index
		std::vector<std::vector<std::tuple<T, size_t, size_t>>>
				edge_ranks; // vertex -> closest connected [distance, bin, edge_index]

		std::vector<vec<T>> vals;					 // vertex -> data
		std::vector<size_t> to_data_index; // vertex -> data_index
		robin_hood::unordered_flat_map<size_t, size_t>
				to_vertex; // data_index -> vertex
		void add_vertex(size_t data_index, const vec<T>& data) {
			to_vertex[data_index] = to_data_index.size();
			to_data_index.emplace_back(data_index);
			adj.emplace_back();
			edge_ranks.emplace_back();
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
	const std::string _name() { return "ZEHNSW Engine('testing')"; }

	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, edge_count_search_factor);
		return pl;
	}
	bool generate_elabel() { return int_distribution(gen); }
};

template <typename T> void zehnsw_engine<T>::_store_vector(const vec<T>& v) {
	size_t data_index = all_entries.size();
	all_entries.push_back(v);

	vertex_heights.emplace_back(
			data_index == 0 ? num_cuts - 1
											: std::min(size_t(floor(-log(distribution(gen)) *
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
			if (j == adj[i][edge_index].second) {
				// duplicate edge found, discard and return
				// will only happen if no swaps have occurred so far
				return;
			}
			if (d < other_d &&
					is_valid_edge(to_data_index[i], to_data_index[j], bin)) {
				// (i,j) is a better edge than (i, adj[i][edge_index]) for the current
				// bin, so swap
				size_t j2 = adj[i][edge_index].second;
				adj[i][edge_index].second = j;
				j = j2;
				// std::swap(j, adj[i][edge_index].second);
				std::swap(d, other_d);
				adj[i][edge_index].first = other_d;
			}
		}
		// iterate through all unused bins, use one of them if it is compatible
		if (used_bins.size() < max_node_size)
			for (size_t bin = 0; bin < num_cuts + 1; ++bin)
				if (!used_bins.contains(bin) && is_valid_edge(i, j, bin)) {
					size_t edge_index = adj[i].size();
					adj[i].emplace_back(d, j);
					edge_ranks[i].emplace_back(d, bin, edge_index);
					break;
				}
		// sort edge ranks by increasing distance again
		std::sort(edge_ranks[i].begin(), edge_ranks[i].end());
		std::vector<std::pair<T, size_t>> new_adj;
		for (auto& [other_d, bin, edge_index] : edge_ranks[i]) {
			new_adj.emplace_back(adj[i][edge_index]);
			edge_index = new_adj.size() - 1;
		}
		adj[i] = new_adj;
		/*
		// sort them in adj too
		std::vector<std::pair<decltype(adj[i][0]), decltype(edge_ranks[i][0])>>
				zipped;
		for (size_t edge_index = 0; edge_index < adj[i].size(); ++edge_index) {
			zipped.emplace_back(adj[i][edge_index], edge_ranks[i][edge_index]);
		}
		std::sort(zipped.begin(), zipped.end());
		for (size_t edge_index = 0; edge_index < adj[i].size(); ++edge_index) {
			adj[i][edge_index] = zipped[edge_index].first;
			edge_ranks[i][edge_index] = zipped[edge_index].second;
		}
		*/
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

	while (layers.back().adj.size() <= 1) {
		std::cout << "Popping useless layer" << std::endl;
		layers.pop_back();
	}

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
	/*
	for (size_t i = 0; i < all_entries.size(); ++i) {
		if (i % 5000 == 0)
			std::cerr << "Built "
								<< double(i + all_entries.size()) /
											 double(all_entries.size() * 2) * 100
								<< "%" << std::endl;

		++op_count;
		improve_vertex_edges(i);
	}
	*/
	std::cout << "Post-build counterr: " << counterr << "/" << total << std::endl;
	counterr = 0;
	total = 0;
}
template <typename T>
const std::vector<std::pair<T, size_t>>
zehnsw_engine<T>::_query_k_internal(const vec<T>& v, size_t k,
																		const std::vector<size_t>& starting_points_,
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

	// init list of starting points by doing a greedy traversal to the approx 1-NN
	std::vector<size_t> starting_points = starting_points_;
	if (false) {
		size_t cur = starting_points_[0]; // assumes size of starting points is 1
		T d_cur = dist2fast(v, all_entries[cur]);
		bool changed = true;
		while (changed) {
			changed = false;
			starting_points.emplace_back(cur);
			for (const auto& [_, u] : adj[cur]) {
				// T d_next = dist2fast(v, vals[u]);
				T d_next = dist2fast(v, vals[u]);
				// T d_next = dist2fast(v, all_entries[to_data_index[u]]);
				if (d_next < d_cur) {
					changed = true;
					d_cur = d_next;
					cur = u;
				}
			}
		}
		reverse(starting_points.begin(), starting_points.end());
	}
	// TODO make a priority queue of lower bounds instead, pop from that maybe?

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
	for (const auto& sp : starting_points) {
		// visit(dist2(v, vals[sp]), sp);
		++counterr;
		++total;
		visit(dist2fast(v, vals[sp]), sp);
	}
	// visit(dist2fast(v, all_entries[to_data_index[sp]]), sp);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		T d_worst = top_k.top().first;
		if (top_k.size() == k && -nd > d_worst)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		if (top_k.size() == k) {
			_mm_prefetch(&adj[cur], _MM_HINT_T0);
			// auto it_low = std::lower_bound(
			//		adj[cur].begin(), adj[cur].end(), nd - d_worst,
			//		[](const auto& p, float val) { return p.first < val; });
			// auto it_high =
			//		std::upper_bound(adj[cur].begin(), adj[cur].end(), nd + d_worst,
			//										 [](float val, auto& p) { return val < p.first; });
			auto it_low = std::lower_bound(
					adj[cur].begin(), adj[cur].end(), -nd - d_worst,
					[&](const std::pair<float, size_t>& neighbour, float) {
						return -nd - neighbour.first > d_worst;
					});
			// it_high can be replaced with an if statement inside the loop
			auto it_high = std::lower_bound(
					adj[cur].begin(), adj[cur].end(), -nd - d_worst,
					[&](const std::pair<float, size_t>& neighbour, float) {
						return neighbour.first - -nd <= d_worst;
					});

			// auto it_high = std::upper_bound(
			//		adj[cur].begin(), adj[cur].end(), d_worst - -nd,
			//		[](float value, const std::pair<float, size_t>& elem) {
			//			return value < elem.first;
			//		});
			//  auto it_low = adj[cur].begin();
			//  auto it_high = adj[cur].end();
			std::for_each(it_low, it_high, [&](auto& neighbour) {
				auto& u = neighbour.second;
				_mm_prefetch(&vals[u], _MM_HINT_T0);
			});
			total += adj[cur].size();
			std::for_each(it_low, it_high, [&](auto& neighbour) {
				++counterr;
				if (std::max(-nd - neighbour.first, neighbour.first - -nd) > d_worst) {
					// TODO this shouldn't happen, increment a 'useless' counter
					std::cout << "Impossible" << std::endl;
				}
				auto& u = neighbour.second;
				T d_next = dist2fast(v, vals[u]);
				visit(d_next, u);
				//}
			});
		} else {
			_mm_prefetch(&adj[cur], _MM_HINT_T0);
			auto it_low = adj[cur].begin();
			auto it_high = adj[cur].end();
			std::for_each(it_low, it_high, [&](auto& neighbour) {
				auto& u = neighbour.second;
				_mm_prefetch(&vals[u], _MM_HINT_T0);
			});
			std::for_each(it_low, it_high, [&](auto& neighbour) {
				auto& u = neighbour.second;
				++counterr;
				++total;
				T d_next = dist2fast(v, vals[u]);
				visit(d_next, u);
			});
		}
		/*
		_mm_prefetch(&adj[cur], _MM_HINT_T0);
		for (const auto& [_, u] : adj[cur]) {
			_mm_prefetch(&vals[u], _MM_HINT_T0);
		}
		for (const auto& [_, u] : adj[cur]) {
			T d_next = dist2fast(v, vals[u]);
			// T d_next = dist2(v, vals[u]);
			// T d_next = dist2fast(v, all_entries[to_data_index[u]]);
			visit(d_next, u);
		}
		*/
	}
	/*
	std::vector<std::pair<T, size_t>> neighbour_buffer;
	neighbour_buffer.reserve(edge_count_mult);
	auto visit_neighbours = [&]() {
		sort(neighbour_buffer.begin(), neighbour_buffer.end(),
			[](const auto& [d1, _], const auto& [d2, _]) {
		return d1 < d2;});
		// TODO binary search for which vectors will be added
		size_t num_to_add = std::lower_bound(neighbour_buffer.begin()
		for (auto& [d, u] : neighbour_buffer) {
			if (!visited.contains(u) && top_k.top().first > d)
				top_k.emplace(d, u);		 // top_k is a max heap
				to_visit.emplace(-d, u); // to_visit is a min heap
			visited[u] = d;
		}

		bool is_good =
				!visited.contains(u) && top_k.top().first > d;
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
		neighbour_buffer.emplace_back(dist2fast(v, vals[sp]), sp);
	visit_neighbours();
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
		neighbour_buffer.clear();
		for (size_t neighbour_index = 0; neighbour_index < adj[cur].size();
				 ++neighbour_index)
			neighbour_buffer.emplace_back(
					dist2fast(v, vals[adj[cur][neighbour_index]]));
		visit_neighbours();
		// for (const auto& u : adj[cur]) {
		//	T d_next = dist2fast(v, vals[u]);
		//	// T d_next = dist2fast(v, all_entries[to_data_index[u]]);
		//	visit(d_next, u);
		// }
	}
	*/
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
	std::vector<size_t> current = {starting_vertex};
	std::vector<std::vector<std::pair<T, size_t>>> ret;
	// for each layer, in decreasing depth
	for (int layer = layers.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (layer > int(full_search_top_layer))
			layer_k = 1;
		for (auto& current_val : current)
			current_val = layers[layer].to_vertex[current_val];
		ret.emplace_back(_query_k_internal(v, layer_k, {current}, layer));
		// ret.emplace_back(_query_k_internal(
		//		v, layer_k, {layers[layer].to_vertex[current]}, layer));
		// current = layers[layer].to_data_index[ret.back().front().second];
		current.clear();
		for (auto& new_current_val : ret.back())
			current.emplace_back(layers[layer].to_data_index[new_current_val.second]);
		// current = layers[layer].to_data_index[ret.back().front().second];
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
			T d_next = dist2fast(v, vals[u]);
			counterr2++;
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

template <typename T> zehnsw_engine<T>::~zehnsw_engine() {
	std::cout << "Post-queries counterr: " << counterr << "/" << total
						<< std::endl;
	// print off height of 0, and max 2 heights
	std::cout << "height of 0: " << vertex_heights[0] << std::endl;
	sort(vertex_heights.rbegin(), vertex_heights.rend());
	std::cout << "largest heights: " << std::endl;
	for (size_t i = 0; i < 10; ++i) {
		std::cout << vertex_heights[i] << ' ';
	}
	std::cout << std::endl;
	std::cout << "counterr2=" << counterr2 << std::endl;
}
