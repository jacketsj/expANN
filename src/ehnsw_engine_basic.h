#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"
#include "topk_t.h"

template <typename T>
struct ehnsw_engine_basic : public ann_engine<T, ehnsw_engine_basic<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t starting_vertex;
	size_t max_depth;
	size_t M;
	size_t M0;
	size_t ef_search;
	size_t ef_construction;
	size_t num_cuts;
	size_t min_per_cut;
	ehnsw_engine_basic(size_t _max_depth, size_t _M,
										 // size_t _M0,
										 size_t _ef_search,
										 //,size_t _ef_construction
										 size_t _num_cuts, size_t _min_per_cut)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				max_depth(_max_depth), M(_M), M0(2 * M), ef_search(_ef_search),
				ef_construction(6 * M), num_cuts(_num_cuts), min_per_cut(_min_per_cut) {
	}
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
	std::vector<size_t> _query_k_at_layer_internal(const vec<T>& v, topk_t<T>& tk,
																								 size_t layer);
	void add_edge(size_t layer, size_t i, size_t j);
	void add_edge_directional(size_t layer, size_t i, size_t j);
	void improve_vertex(size_t i, size_t num_nn);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "EHNSW Engine Basic"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search);
		add_param(pl, ef_construction);
		add_param(pl, num_cuts);
		add_param(pl, min_per_cut);
		return pl;
	}
};

template <typename T>
void ehnsw_engine_basic<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
void ehnsw_engine_basic<T>::add_edge_directional(size_t layer, size_t i,
																								 size_t j) {
	size_t edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

	T d = dist2(all_entries[i], all_entries[j]);
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
void ehnsw_engine_basic<T>::add_edge(size_t layer, size_t i, size_t j) {
	add_edge_directional(layer, i, j);
	add_edge_directional(layer, j, i);
}

template <typename T> void ehnsw_engine_basic<T>::_build() {
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
		// get the layer this entry will go up to
		size_t cur_layer_ub = floor(-log(distribution(gen)) * 1 / log(double(M)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// if it is a new layer, add a layer
		while (cur_layer >= hadj.size())
			add_layer(i);
		// search for new edges to add
		size_t cur_vert = starting_vertex;
		int layer;
		for (layer = hadj.size() - 1; layer > int(cur_layer); --layer) {
			topk_t<T> tk(1);
			tk.consider(dist2(all_entries[i], all_entries[cur_vert]), cur_vert);
			cur_vert = _query_k_at_layer_internal(all_entries[i], tk, layer)[0];
		}
		for (; layer >= 0; --layer) {
			edge_ranks[layer][i].resize(num_cuts + 1);
			topk_t<T> tk(ef_construction);
			tk.consider(dist2(all_entries[i], all_entries[cur_vert]), cur_vert);
			auto closest_entries =
					_query_k_at_layer_internal(all_entries[i], tk, layer);
			cur_vert = closest_entries[0];
			for (size_t j : closest_entries)
				add_edge(layer, i, j);
		}
	}
}
template <typename T>
std::vector<size_t>
ehnsw_engine_basic<T>::_query_k_at_layer_internal(const vec<T>& v,
																									topk_t<T>& tk, size_t layer) {
	std::priority_queue<std::pair<T, size_t>> to_visit;
	robin_hood::unordered_flat_set<size_t> visited;
	auto visit = [&](T d, size_t u) {
		bool is_good = !visited.contains(u) && tk.consider(d, u);
		visited.insert(u);
		if (is_good) {
			to_visit.emplace(-d, u); // to_visit is a min heap
		}
		return is_good;
	};
	for (auto& u : tk.to_vector())
		to_visit.emplace(dist2(v, all_entries[u]), u);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (-nd > tk.worst_val())
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (const auto& u : hadj[layer][cur]) {
			T d_next = dist2(v, all_entries[u]);
			visit(d_next, u);
		}
	}
	return tk.to_vector();
}

template <typename T>
std::vector<size_t> ehnsw_engine_basic<T>::_query_k(const vec<T>& v, size_t k) {
	size_t cur_vert = 0;
	int layer;
	for (layer = hadj.size() - 1; layer > 0; --layer) {
		topk_t<T> tk(1);
		tk.consider(dist2(v, all_entries[cur_vert]), cur_vert);
		cur_vert = _query_k_at_layer_internal(v, tk, layer)[0];
	}
	topk_t<T> tk(ef_search * k);
	tk.consider(dist2(v, all_entries[cur_vert]), cur_vert);
	return _query_k_at_layer_internal(v, tk, layer);
}
