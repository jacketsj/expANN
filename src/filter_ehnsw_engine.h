#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

struct filter_ehnsw_engine_config {
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t num_cuts;
	size_t min_per_cut;
	bool quick_search;
	bool bumping;
	bool quick_build;
	filter_ehnsw_engine_config(size_t _max_depth, size_t _edge_count_mult,
														 size_t _num_for_1nn, size_t _num_cuts,
														 size_t _min_per_cut, bool _quick_search,
														 bool _bumping, bool _quick_build)
			: max_depth(_max_depth), edge_count_mult(_edge_count_mult),
				num_for_1nn(_num_for_1nn), num_cuts(_num_cuts),
				min_per_cut(_min_per_cut), quick_search(_quick_search),
				bumping(_bumping), quick_build(_quick_build) {}
};

template <typename T>
struct filter_ehnsw_engine : public ann_engine<T, filter_ehnsw_engine<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	std::uniform_int_distribution<> int_distribution;
	size_t starting_vertex;
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t num_cuts;
	size_t min_per_cut;
	bool quick_search;
	bool bumping;
	bool quick_build;
	filter_ehnsw_engine(filter_ehnsw_engine_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				max_depth(conf.max_depth), edge_count_mult(conf.edge_count_mult),
				num_for_1nn(conf.num_for_1nn), num_cuts(conf.num_cuts),
				min_per_cut(conf.min_per_cut), quick_search(conf.quick_search),
				bumping(conf.bumping), quick_build(conf.quick_build) {}
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
	const std::vector<std::vector<size_t>>
	_query_k_at_layer(const vec<T>& v, size_t k, size_t starting_point,
										size_t layer,
										std::vector<std::function<bool(size_t)>> filters);
	void add_edge(size_t layer, size_t i, size_t j);
	void add_edge_directional(size_t layer, size_t i, size_t j);
	const std::vector<std::vector<size_t>>
	_query_k_internal(const vec<T>& v, size_t k, size_t full_search_top_layer,
										std::vector<std::function<bool(size_t)>> filters =
												std::vector<std::function<bool(size_t)>>({[](size_t) {
													return true;
												}}));
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Filter EHNSW Engine"; }
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
		return pl;
	}
};

template <typename T>
void filter_ehnsw_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
void filter_ehnsw_engine<T>::add_edge_directional(size_t layer, size_t i,
																									size_t j) {
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
		add_edge_directional(layer, i, deleted_j);
}

template <typename T>
void filter_ehnsw_engine<T>::add_edge(size_t layer, size_t i, size_t j) {
	add_edge_directional(layer, i, j);
	add_edge_directional(layer, j, i);
}

template <typename T> void filter_ehnsw_engine<T>::_build() {
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
	std::vector<std::function<bool(size_t)>> filters;
	for (size_t cut = 0; cut < num_cuts; ++cut)
		filters.emplace_back(
				[&, cut](size_t index) { return e_labels[index][cut]; });
	filters.emplace_back([](size_t) { return true; });
	for (size_t i = 1; i < all_entries.size(); ++i) {

		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;

		for (size_t cut = 0; cut < num_cuts; ++cut)
			e_labels[i].emplace_back(int_distribution(gen));
		// get the layer this entry will go up to
		size_t cur_layer_ub =
				floor(-log(distribution(gen)) * 1 / log(double(edge_count_mult)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// get kNN at each layer
		size_t full_search_top_layer = hadj.size() - 1;
		if (quick_build)
			full_search_top_layer = cur_layer;
		std::vector<std::vector<size_t>> kNN = _query_k_internal(
				all_entries[i], edge_count_mult, full_search_top_layer, filters);
		// if it is a new layer, add a layer (important that this happens AFTER kNN
		// at each layer)
		while (cur_layer >= hadj.size())
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
const std::vector<std::vector<size_t>>
filter_ehnsw_engine<T>::_query_k_at_layer(
		const vec<T>& v, size_t k, size_t starting_point, size_t layer,
		std::vector<std::function<bool(size_t)>> filters) {
	assert(filters.size() > 0);
	std::vector<std::priority_queue<std::pair<T, size_t>>> top_k(filters.size());
	std::priority_queue<std::pair<T, size_t>> to_visit;
	robin_hood::unordered_flat_set<size_t> visited;
	auto visit = [&](T d, size_t u) {
		int is_good = -1;
		if (!visited.contains(u)) {
			visited.insert(u);
			for (size_t filter_index = 0; filter_index < filters.size();
					 ++filter_index) {
				if (filters[filter_index](u)) {
					if (top_k[filter_index].size() < k ||
							top_k[filter_index].top().first > d) {
						is_good = filter_index;
						break;
					}
				}
			}
		}
		// bool is_good =
		//		!visited.contains(u) && (top_k.size() < k || top_k.top().first > d);
		if (is_good != -1) {
			top_k[is_good].emplace(d, u); // top_k[...] is a max heap
			if (top_k[is_good].size() > k)
				top_k[is_good].pop();
			to_visit.emplace(-d, u); // to_visit is a min heap
		}
		// return is_good != -1;
	};
	visit(dist(v, all_entries[starting_point]), starting_point);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		to_visit.pop();
		bool forward_progress = false;
		for (size_t filter_index = 0; filter_index < filters.size(); ++filter_index)
			if (filters[filter_index](cur) &&
					!(top_k[filter_index].size() == k &&
						-nd > top_k[filter_index].top().first)) {
				forward_progress = true;
				break;
			}
		if (!forward_progress)
			// everything neighbouring current best set is already evaluated
			break;
		// if (top_k.size() == k && -nd > top_k.top().first)
		//	break;
		for (const auto& u : hadj[layer][cur]) {
			T d_next = dist(v, all_entries[u]);
			visit(d_next, u);
		}
	}
	std::vector<std::vector<size_t>> ret;
	for (size_t filter_index = 0; filter_index < filters.size(); ++filter_index) {
		std::vector<size_t> ret_filter;
		while (!top_k[filter_index].empty()) {
			ret_filter.push_back(top_k[filter_index].top().second);
			top_k[filter_index].pop();
		}
		reverse(ret_filter.begin(),
						ret_filter.end()); // sort from closest to furthest
		ret.emplace_back(ret_filter);
	}
	return ret;
}

template <typename T>
const std::vector<std::vector<size_t>>
filter_ehnsw_engine<T>::_query_k_internal(
		const vec<T>& v, size_t k, size_t full_search_top_layer,
		std::vector<std::function<bool(size_t)>> filters) {
	auto current = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (layer > int(full_search_top_layer))
			layer_k = 1;
		auto returned_list = _query_k_at_layer(v, layer_k, current, layer, filters);
		std::vector<size_t> returned_list_concat;
		for (auto& l : returned_list) {
			for (auto& value : l)
				returned_list_concat.emplace_back(value);
		}
		ret.push_back(returned_list_concat);
		// ret.push_back(_query_k_at_layer(v, layer_k, current, layer, filters));
		current = ret.back().front();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> filter_ehnsw_engine<T>::_query_k(const vec<T>& v,
																										 size_t k) {
	auto ret = _query_k_internal(v, k * num_for_1nn, 0)[0];
	ret.resize(std::min(k, ret.size()));
	return ret;
}
