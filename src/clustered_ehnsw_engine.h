#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

struct clustered_ehnsw_engine_config {
	size_t max_depth;
	size_t edge_count_mult;
	size_t num_for_1nn;
	size_t num_cuts;
	size_t min_per_cut;
	size_t cluster_size;
	size_t min_cluster_membership;
	bool quick_search;
	bool bumping;
	bool quick_build;
	clustered_ehnsw_engine_config(size_t _max_depth, size_t _edge_count_mult,
																size_t _num_for_1nn, size_t _num_cuts,
																size_t _min_per_cut, size_t _cluster_size,
																size_t _min_cluster_membership,
																bool _quick_search, bool _bumping,
																bool _quick_build)
			: max_depth(_max_depth), edge_count_mult(_edge_count_mult),
				num_for_1nn(_num_for_1nn), num_cuts(_num_cuts),
				min_per_cut(_min_per_cut), cluster_size(_cluster_size),
				min_cluster_membership(_min_cluster_membership),
				quick_search(_quick_search), bumping(_bumping),
				quick_build(_quick_build) {}
};

struct layer {
	// data index -> adjacent cluster indeces
	robin_hood::unordered_flat_map<size_t, std::vector<size_t>> cadj;
	// cluster index -> contained data indeces
	std::vector<std::vector<size_t>> ccont;
	// data index -> containing cluster indeces
	robin_hood::unordered_flat_map<size_t, std::vector<size_t>> ccont_inv;
	// data index -> cut bin (or num_cuts for no cut) -> edges in that bin
	robin_hood::unordered_flat_map<size_t, std::vector<std::vector<size_t>>>
			edge_bins;
	// TODO remove edge_ranks maybe (stub based on ehnsw2 for initial testing)
	robin_hood::unordered_flat_map<
			size_t, std::vector<std::set<std::pair<float, size_t>>>>
			edge_ranks;
	layer() {}
	void add_vertex(size_t data_index, size_t num_bins) {
		cadj[data_index] = std::vector<size_t>();
		ccont_inv[data_index] = std::vector<size_t>();
		edge_bins[data_index] = std::vector<std::vector<size_t>>(num_bins);
		edge_ranks[data_index].resize(num_bins);
	}
	size_t add_cluster() {
		size_t ret = ccont.size();
		ccont.emplace_back();
		return ret;
	}
	void add_to_cluster(size_t data_index, size_t cluster_index) {
		ccont_inv[data_index].emplace_back(cluster_index);
		ccont[cluster_index].emplace_back(data_index);
	}
};

template <typename T>
struct clustered_ehnsw_engine
		: public ann_engine<T, clustered_ehnsw_engine<T>> {
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
	size_t cluster_size;
	size_t min_cluster_membership;
	bool quick_search;
	bool bumping;
	bool quick_build;
	clustered_ehnsw_engine(clustered_ehnsw_engine_config conf)
			: rd(), gen(rd()), distribution(0, 1), int_distribution(0, 1),
				max_depth(conf.max_depth), edge_count_mult(conf.edge_count_mult),
				num_for_1nn(conf.num_for_1nn), num_cuts(conf.num_cuts),
				min_per_cut(conf.min_per_cut), cluster_size(conf.cluster_size),
				min_cluster_membership(conf.min_cluster_membership),
				quick_search(conf.quick_search), bumping(conf.bumping),
				quick_build(conf.quick_build) {}
	std::vector<vec<T>> all_entries;
	std::vector<layer> layers;
	std::vector<size_t> top_layers; // data_index -> top layer it appears on
	std::vector<std::vector<bool>>
			e_labels; // data index -> cut labels [*num_cuts]
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k_at_layer(const vec<T>& v, size_t k,
																							size_t starting_point,
																							const layer& ly);
	void add_edge(layer& lr, size_t i, size_t j);
	void add_edge_directional_to_cluster(layer& lr, size_t data_index,
																			 size_t cluster_index);
	void add_edge_directional(layer& lr, size_t data_index_from,
														size_t data_index_to);
	const std::vector<std::vector<size_t>>
	_query_k_internal(const vec<T>& v, size_t k, size_t full_search_top_layer);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Clustered EHNSW Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, edge_count_mult);
		add_param(pl, num_for_1nn);
		add_param(pl, num_cuts);
		add_param(pl, min_per_cut);
		add_param(pl, cluster_size);
		add_param(pl, quick_search);
		add_param(pl, bumping);
		add_param(pl, quick_build);
		return pl;
	}
};

template <typename T>
void clustered_ehnsw_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
	// choose cut bits
	e_labels.emplace_back();
	for (size_t cut = 0; cut < num_cuts; ++cut)
		e_labels[e_labels.size() - 1].emplace_back(int_distribution(gen));
	// choose top layer
	size_t cur_layer_ub =
			floor(-log(distribution(gen)) * 1 / log(double(edge_count_mult)));
	size_t cur_layer = std::min(cur_layer_ub, max_depth);
	top_layers.emplace_back(cur_layer);
}

template <typename T>
void clustered_ehnsw_engine<T>::add_edge_directional(layer& lr,
																										 size_t data_index_from,
																										 size_t data_index_to) {
	// TODO check that no cluster containing data_index_to is in the adjacent set
	// already (mabye also containing set) (and that data_index_from !=
	// data_index_to)
	for (size_t cluster_index : lr.ccont_inv[data_index_to])
		add_edge_directional_to_cluster(lr, data_index_from, cluster_index);
}

template <typename T>
void clustered_ehnsw_engine<T>::add_edge_directional_to_cluster(
		layer& lr, size_t data_index, size_t cluster_index) {
	// TODO make this use the clustered structure
	// TODO check that cluster_index does not contain data_index and cluster_index
	// is not in the adjacent set already
	//
	// currently: stub version that assumes cluster sizes of 1 (currently true)
	size_t i = data_index;
	size_t j = lr.ccont[cluster_index][0];

	// TODO check that j != i and j is not in the adjacent set already

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
	int deleted_cluster_index =
			-1; // nothing deleted if deleted_cluster_index == -1
	if (lr.cadj[i].size() + 1 > edge_count_mult) {
		for (size_t cut : cuts) {
			size_t sz = lr.edge_ranks[i][cut].size();
			if (cut == found_cut)
				sz++;
			if (sz > min_per_cut) {
				T cur_cut_val = lr.edge_ranks[i][cut].begin()->first;
				if (cur_cut_val < max_cut_val) {
					max_cut_val = cur_cut_val;
					max_cut = cut;
				}
			}
		}
		// if that longest edge is longer than our candidate edge, make the swap
		if (-max_cut_val > d) {
			auto iter = lr.edge_ranks[i][max_cut].begin();
			size_t edge_index = iter->second;
			deleted_cluster_index = lr.cadj[i][edge_index];
			lr.cadj[i][edge_index] = cluster_index;
			// deleted_j = hadj[layer][i][edge_index];
			// hadj[layer][i][edge_index] = j;
			lr.edge_ranks[i][max_cut].erase(iter);
			lr.edge_ranks[i][found_cut].emplace(-d, edge_index);
		}
	} else {
		lr.cadj[i].emplace_back(cluster_index);
		lr.edge_ranks[i][found_cut].emplace(-d, lr.cadj[i].size() - 1);
	}
	// try to bump a bigger edge until stability is reached (if bumping is
	// enabled)
	if (deleted_cluster_index >= 0 && bumping)
		add_edge_directional(lr, i, size_t(deleted_cluster_index));
}

template <typename T>
void clustered_ehnsw_engine<T>::add_edge(layer& lr, size_t i, size_t j) {
	add_edge_directional(lr, i, j);
	add_edge_directional(lr, j, i);
}

template <typename T> void clustered_ehnsw_engine<T>::_build() {
	assert(all_entries.size() > 0);
	// get highest layer, some highest vertex on that layer becomes start point
	size_t highest_vertex_index =
			std::distance(top_layers.begin(),
										std::max_element(top_layers.begin(), top_layers.end()));
	starting_vertex = highest_vertex_index;
	max_depth = top_layers[highest_vertex_index];
	for (size_t layer_index = 0; layer_index <= max_depth; ++layer_index) {
		layers.emplace_back();
	}

	// TODO do clustering on each layer here
	assert(cluster_size == 1);
	assert(min_cluster_membership == 1);
	size_t num_bins = num_cuts + 1;
	for (size_t vind = 0; vind < all_entries.size(); ++vind) {
		for (size_t layer_index = 0; layer_index <= max_depth; ++layer_index) {
			layers[layer_index].add_vertex(vind, num_bins);
			// as a stub routine (for testing), make clusters of size 1
			size_t cluster_index = layers[layer_index].add_cluster();
			layers[layer_index].add_to_cluster(vind, cluster_index);
		}
	}

	std::vector<size_t> data_indeces_to_process;
	for (size_t i = 0; i < all_entries.size(); ++i) {
		if (i != highest_vertex_index)
			data_indeces_to_process.emplace_back(i);
	}
	for (size_t cdi : data_indeces_to_process) {
		// if (cdi % 5000 == 0)
		//	std::cerr << "Built " << double(cdi) / double(all_entries.size()) * 100
		//						<< "%" << std::endl;

		// TODO extract out this process
		// use the extracted process for improvements, and also for filter edges
		// (including expander bins)

		// get the layer this entry will go up to
		size_t cur_layer = top_layers[cdi];
		// get kNN at each layer
		size_t full_search_top_layer = layers.size() - 1;
		if (quick_build)
			full_search_top_layer = cur_layer;
		std::vector<std::vector<size_t>> kNN = _query_k_internal(
				all_entries[cdi], edge_count_mult, full_search_top_layer);
		// add all the neighbours as edges
		for (size_t layer_index = 0;
				 layer_index <= cur_layer && layer_index < kNN.size(); ++layer_index) {
			for (size_t j : kNN[layer_index]) {
				add_edge(layers[layer_index], cdi, j);
			}
		}
	}
}
template <typename T>
const std::vector<size_t> clustered_ehnsw_engine<T>::_query_k_at_layer(
		const vec<T>& v, size_t k, size_t starting_point, const layer& lr) {
	// TODO this should be taking a vector of starting points
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
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		// TODO also iterate through lr.ccont_inv maybe
		for (const auto& cluster_index : lr.cadj.at(cur))
			for (const auto& next_di : lr.ccont[cluster_index]) {
				T d_next = dist(v, all_entries[next_di]);
				visit(d_next, next_di);
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
clustered_ehnsw_engine<T>::_query_k_internal(const vec<T>& v, size_t k,
																						 size_t full_search_top_layer) {
	// TODO use a vector of starting vertices, replaced with the top layer_k each
	// time
	// TODO add a search_k param for before the top full_search_top_layer (e.g. 3)
	auto current = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer_index = layers.size() - 1; layer_index >= 0; --layer_index) {
		size_t layer_k = k;
		if (layer_index > int(full_search_top_layer))
			layer_k = 1;
		ret.push_back(_query_k_at_layer(v, layer_k, current, layers[layer_index]));
		current = ret.back().front();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> clustered_ehnsw_engine<T>::_query_k(const vec<T>& v,
																												size_t k) {
	auto ret = _query_k_internal(v, k * num_for_1nn, 0)[0];
	ret.resize(std::min(k, ret.size()));
	return ret;
}
