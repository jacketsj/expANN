#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

template <typename T>
struct ehnsw_engine_2 : public ann_engine<T, ehnsw_engine_2<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t max_depth;
	size_t edge_count_mult;
	size_t starting_vertex;
	size_t num_for_1nn;
	size_t num_cuts;
	size_t min_per_cut;
	ehnsw_engine_2(size_t _max_depth, size_t _edge_count_mult,
								 size_t _num_for_1nn, size_t _num_cuts, size_t _min_per_cut)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				max_depth(_max_depth), edge_count_mult(_edge_count_mult),
				num_for_1nn(_num_for_1nn), num_cuts(_num_cuts),
				min_per_cut(_min_per_cut) {}
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
	const std::vector<size_t> _query_k_at_layer(const vec<T>& v, size_t k,
																							size_t starting_point,
																							size_t layer);
	void add_edge(size_t layer, size_t i, size_t j);
	void add_edge_directional(size_t layer, size_t i, size_t j);
	const std::vector<std::vector<size_t>> _query_k(const vec<T>& v, size_t k,
																									bool fill_all_layers = false);
	const vec<T>& _query(const vec<T>& v);
	const std::string _name() { return "EHNSW Engine 2"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, num_cuts);
		add_param(pl, min_per_cut);
		return pl;
	}
};

template <typename T> void ehnsw_engine_2<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
void ehnsw_engine_2<T>::add_edge_directional(size_t layer, size_t i, size_t j) {
	T d = dist(all_entries[i], all_entries[j]);
	std::vector<size_t> cuts;
	for (size_t cut = 0; cut <= num_cuts; ++cut)
		cuts.push_back(cut);
	std::shuffle(cuts.begin(), cuts.end(), gen);
	// choose a random sequence of cuts until a cut that allows for (i,j) is found
	size_t found_cut = num_cuts;
	for (size_t cut : cuts) {
		if (cut == num_cuts || e_labels[i][cut] != e_labels[j][cut]) {
			found_cut = cut;
			break;
		}
	}
	// if the total number of edges would be too large, find all cuts which are
	// not below min_per_cut, and pick out the one with the longest edge
	size_t max_cut = 0;
	T max_cut_val = std::numeric_limits<T>::max();
	if (hadj[layer][i].size() + 1 >= edge_count_mult) {
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
			hadj[layer][i][edge_index] = j;
			edge_ranks[layer][i][max_cut].erase(iter);
			edge_ranks[layer][i][found_cut].emplace(-d, edge_index);
		}
	} else {
		hadj[layer][i].emplace_back(j);
		edge_ranks[layer][i][found_cut].emplace(-d, hadj[layer][i].size() - 1);
	}
}

template <typename T>
void ehnsw_engine_2<T>::add_edge(size_t layer, size_t i, size_t j) {
	add_edge_directional(layer, i, j);
	add_edge_directional(layer, j, i);
}

template <typename T> void ehnsw_engine_2<T>::_build() {
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
		e_labels[0].emplace_back(int_distribution(gen));
	for (size_t i = 1; i < all_entries.size(); ++i) {

		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;

		for (size_t cut = 0; cut < num_cuts; ++cut)
			e_labels[i].emplace_back(int_distribution(gen));
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
		for (size_t layer = 0; layer <= cur_layer && layer < kNN.size(); ++layer) {
			edge_ranks[layer][i].resize(num_cuts + 1);
			for (size_t j : kNN[layer]) {
				add_edge(layer, i, j);
			}
		}
	}
}
template <typename T>
const std::vector<size_t>
ehnsw_engine_2<T>::_query_k_at_layer(const vec<T>& v, size_t k,
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
		for (const auto& u : hadj[layer][cur]) {
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
ehnsw_engine_2<T>::_query_k(const vec<T>& v, size_t k, bool fill_all_layers) {
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

template <typename T> const vec<T>& ehnsw_engine_2<T>::_query(const vec<T>& v) {
	return all_entries[_query_k(v, num_for_1nn)[0][0]];
}
