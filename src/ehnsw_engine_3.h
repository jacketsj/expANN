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

struct ehnsw_engine_3_config {
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t num_cuts;
	size_t min_per_cut;
	bool quick_search;
	bool bumping;
	bool quick_build;
	float elabel_prob;
	size_t cut_run_length;
	ehnsw_engine_3_config(size_t _max_depth, size_t _edge_count_mult,
												size_t _num_for_1nn, size_t _num_cuts,
												size_t _min_per_cut, bool _quick_search, bool _bumping,
												bool _quick_build, float _elabel_prob = 0.5f,
												size_t _cut_run_length = 1)
			: max_depth(_max_depth), edge_count_mult(_edge_count_mult),
				num_for_1nn(_num_for_1nn), num_cuts(_num_cuts),
				min_per_cut(_min_per_cut), quick_search(_quick_search),
				bumping(_bumping), quick_build(_quick_build), elabel_prob(_elabel_prob),
				cut_run_length(_cut_run_length) {}
};

template <typename T>
struct ehnsw_engine_3 : public ann_engine<T, ehnsw_engine_3<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	std::bernoulli_distribution elabel_distribution;
	size_t starting_vertex;
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t num_cuts;
	size_t min_per_cut;
	bool quick_search;
	bool bumping;
	bool quick_build;
	float elabel_prob;
	size_t cut_run_length;
	ehnsw_engine_3(ehnsw_engine_3_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				elabel_distribution(conf.elabel_prob), max_depth(conf.max_depth),
				edge_count_mult(conf.edge_count_mult), num_for_1nn(conf.num_for_1nn),
				num_cuts(conf.num_cuts), min_per_cut(conf.min_per_cut),
				quick_search(conf.quick_search), bumping(conf.bumping),
				quick_build(conf.quick_build), elabel_prob(conf.elabel_prob),
				cut_run_length(conf.cut_run_length) {}
	std::vector<vec<T>> all_entries;
	std::vector<robin_hood::unordered_flat_map<size_t, std::vector<size_t>>> hadj;
	std::vector<robin_hood::unordered_flat_map<
			size_t, std::vector<std::set<std::pair<T, size_t>>>>>
			edge_ranks; // layer -> vertex -> cut (or num_cuts for no cut) -> furthest
									// connected
	robin_hood::unordered_flat_map<size_t, std::vector<bool>>
			e_labels; // vertex -> cut labels (*num_cuts)
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<std::pair<T, size_t>>
	_query_k_at_layer(const vec<T>& v, size_t k,
										std::vector<size_t>& starting_points, size_t layer,
										bool include_visited);
	bool is_valid_edge(size_t i, size_t j, size_t bin);
	void add_edge(size_t layer, size_t i, size_t j, T d);
	void add_edge_directional(size_t layer, size_t i, size_t j, T d);
	const std::vector<std::vector<std::pair<T, size_t>>>
	_query_k_internal(const vec<T>& v, size_t k, size_t full_search_top_layer,
										bool include_visited);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "EHNSW Engine 3"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, num_cuts);
		add_param(pl, min_per_cut);
		add_param(pl, quick_search);
		add_param(pl, bumping);
		add_param(pl, quick_build);
		add_param(pl, elabel_prob);
		add_param(pl, cut_run_length);
		return pl;
	}
	bool generate_elabel() { return elabel_distribution(gen); }
};

template <typename T> void ehnsw_engine_3<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
bool ehnsw_engine_3<T>::is_valid_edge(size_t i, size_t j, size_t bin) {
	if (bin == num_cuts)
		return true;
	// an edge is permitted in a bin if it crosses the cut for that bin, and does
	// not cross the cuts in the `parent' bins
	bool is_valid = e_labels[i][bin] != e_labels[j][bin];
	for (int other_bin = bin; other_bin / cut_run_length == bin / cut_run_length;
			 --other_bin)
		is_valid = is_valid && e_labels[i][bin] == e_labels[j][bin];
	return is_valid;
}

template <typename T>
void ehnsw_engine_3<T>::add_edge_directional(size_t layer, size_t i, size_t j,
																						 T d) {
	// T d = dist(all_entries[i], all_entries[j]);
	std::vector<size_t> cuts;
	for (size_t cut = 0; cut <= num_cuts; ++cut)
		cuts.push_back(cut);
	std::shuffle(cuts.begin(), cuts.end(), gen);
	// choose a random sequence of cuts until a cut that allows for (i,j) is found
	size_t found_cut = num_cuts;
	for (size_t cut : cuts) {
		// if (cut == num_cuts || e_labels[i][cut] != e_labels[j][cut]) {
		if (is_valid_edge(i, j, cut)) {
			found_cut = cut;
			break;
		}
	}
	// if the total number of edges would be too large, find all cuts which are
	// not below min_per_cut, and pick out the one with the longest edge
	size_t max_cut = 0;
	T max_cut_val = std::numeric_limits<T>::max();
	size_t deleted_j = i; // nothing deleted if deleted_j == i
	if (hadj[layer][i].size() + 1 > edge_count_mult) {
		for (size_t cut : cuts) {
			size_t sz = edge_ranks[layer][i][cut].size();
			if (cut == found_cut)
				sz++;
			if (sz > min_per_cut) {
				T cur_cut_val = edge_ranks[layer][i][cut].begin()->first;
				if (cur_cut_val < max_cut_val) {
					max_cut_val = cur_cut_val;
					max_cut = cut;
				}
			}
		}
		// if that longest edge is longer than our candidate edge, make the swap
		if (-max_cut_val > d) {
			auto iter = edge_ranks[layer][i][max_cut].begin();
			size_t edge_index = iter->second;
			deleted_j = hadj[layer][i][edge_index];
			hadj[layer][i][edge_index] = j;
			edge_ranks[layer][i][max_cut].erase(iter);
			edge_ranks[layer][i][found_cut].emplace(-d, edge_index);
		}
	} else {
		hadj[layer][i].emplace_back(j);
		edge_ranks[layer][i][found_cut].emplace(-d, hadj[layer][i].size() - 1);
	}
	// try to bump a bigger edge until stability is reached (if bumping is
	// enabled)
	if (deleted_j != i && bumping)
		add_edge_directional(layer, i, deleted_j, -max_cut_val);
}

template <typename T>
void ehnsw_engine_3<T>::add_edge(size_t layer, size_t i, size_t j, T d) {
	add_edge_directional(layer, i, j, d);
	add_edge_directional(layer, j, i, d);
}

template <typename T> void ehnsw_engine_3<T>::_build() {
	assert(all_entries.size() > 0);
	auto add_layer = [&](size_t v) {
		hadj.emplace_back();
		edge_ranks.emplace_back();
		hadj.back()[v] = std::vector<size_t>();
		edge_ranks.back()[v] =
				std::vector<std::set<std::pair<T, size_t>>>(num_cuts + 1);
		starting_vertex = v;
	};
	// add one layer to start, with first vertex
	add_layer(0);
	for (size_t cut = 0; cut < num_cuts; ++cut)
		e_labels[0].emplace_back(generate_elabel());
	for (size_t i = 1; i < all_entries.size(); ++i) {

		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;

		for (size_t cut = 0; cut < num_cuts; ++cut)
			e_labels[i].emplace_back(generate_elabel());
		// get the layer this entry will go up to
		size_t cur_layer_ub =
				floor(-log(distribution(gen)) * 1 / log(double(edge_count_mult)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// get kNN at each layer
		size_t full_search_top_layer = hadj.size() - 1;
		if (quick_build)
			full_search_top_layer = cur_layer;
		std::vector<std::vector<std::pair<T, size_t>>> kNN = _query_k_internal(
				all_entries[i], edge_count_mult, full_search_top_layer, true);
		// if it is a new layer, add a layer (important that this happens AFTER kNN
		// at each layer)
		while (cur_layer >= hadj.size())
			add_layer(i);
		// add all the neighbours as edges
		for (size_t layer = 0; layer <= cur_layer && layer < kNN.size(); ++layer) {
			edge_ranks[layer][i].resize(num_cuts + 1);
			// TODO consider looking at previous layers too, but probably doesn't
			// matter since using one layer normally anyway
			for (auto [d, j] : kNN[layer]) {
				add_edge(layer, i, j, d);
			}
		}
	}
}
template <typename T>
const std::vector<std::pair<T, size_t>>
ehnsw_engine_3<T>::_query_k_at_layer(const vec<T>& v, size_t k,
																		 std::vector<size_t>& starting_points,
																		 size_t layer, bool include_visited) {
	// TODO add an option to restrict size of to_visit, default to
	// all_entries.size() (causing greedy dfs if it is size=1 basically)
	// TODO then call this function twice in a row, the first time with both the
	// above flag and include_visited, the second time with neither flag
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
	for (auto& sp : starting_points)
		visit(dist(v, all_entries[sp]), sp);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (const auto& u : hadj[layer][cur]) {
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
ehnsw_engine_3<T>::_query_k_internal(const vec<T>& v, size_t k,
																		 size_t full_search_top_layer,
																		 bool include_visited) {
	std::vector<size_t> current = {starting_vertex};
	// auto current = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<std::pair<T, size_t>>> ret;
	// for each layer, in decreasing depth
	for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (layer > int(full_search_top_layer))
			layer_k = 1;
		ret.push_back(
				_query_k_at_layer(v, layer_k, current, layer, include_visited));
		// TODO let current = ret.back() instead too, maybe doesn't matter since I'm
		// only using one layer usually though
		current = {ret.back().front().second};
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> ehnsw_engine_3<T>::_query_k(const vec<T>& v, size_t k) {
	auto ret_combined = _query_k_internal(v, k * num_for_1nn, 0, false)[0];
	ret_combined.resize(std::min(k, ret_combined.size()));
	auto ret = std::vector<size_t>(ret_combined.size());
	for (size_t i = 0; i < ret.size(); ++i)
		ret[i] = ret_combined[i].second;
	return ret;
}
