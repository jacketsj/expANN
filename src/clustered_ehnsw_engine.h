#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "ehnsw_engine_2.h"
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

typedef size_t VIndex;
typedef size_t CIndex;
template <typename T> struct layer {
	// VIndex -> adjacent CIndex[]
	std::vector<std::vector<CIndex>> v_to_adj_c;
	// data_index -> VIndex
	robin_hood::unordered_flat_map<size_t, VIndex> di_to_v;
	std::vector<size_t> v_to_di;
	VIndex get_vindex(size_t data_index) const { return di_to_v.at(data_index); }
	size_t get_data_index(VIndex vind) const { return v_to_di.at(size_t(vind)); }
	// data index -> adjacent cluster indeces
	const std::vector<CIndex>& cadj(VIndex v) const {
		return v_to_adj_c.at(size_t(v));
	}
	std::vector<CIndex>& cadj_mut(VIndex v) { return v_to_adj_c[size_t(v)]; }
	// robin_hood::unordered_flat_map<size_t, std::vector<size_t>> cadj;
	//  cluster index -> contained vertex indeces
	std::vector<std::vector<VIndex>> ccont;
	//  cluster index -> contained vertex indeces + internal data
	std::vector<std::vector<std::pair<VIndex, vec<T>>>> ccont_data;
	// data index -> containing cluster indeces
	robin_hood::unordered_flat_map<size_t, std::vector<CIndex>> ccont_inv;
	// data index -> cut bin (or num_cuts for no cut) -> edges in that bin
	robin_hood::unordered_flat_map<size_t, std::vector<std::vector<size_t>>>
			edge_bins;
	// TODO remove edge_ranks maybe (stub based on ehnsw2 for initial testing)
	robin_hood::unordered_flat_map<
			size_t, std::vector<std::set<std::pair<float, size_t>>>>
			edge_ranks;
	layer() {}
	void add_vertex(size_t data_index, size_t num_bins) {
		VIndex new_vind = v_to_adj_c.size();
		v_to_adj_c.emplace_back();
		v_to_di.emplace_back(data_index);
		di_to_v[data_index] = new_vind;
		// cadj[data_index] = std::vector<size_t>();
		ccont_inv[data_index] = std::vector<CIndex>();
		edge_bins[data_index] = std::vector<std::vector<size_t>>(num_bins);
		edge_ranks[data_index] =
				std::vector<std::set<std::pair<float, size_t>>>(num_bins);
	}
	size_t add_cluster() {
		size_t ret = ccont.size();
		ccont.emplace_back();
		ccont_data.emplace_back();
		return ret;
	}
	void add_to_cluster(size_t data_index, const vec<T>& data,
											size_t cluster_index) {
		ccont_inv[data_index].emplace_back(cluster_index);
		ccont[cluster_index].emplace_back(get_vindex(data_index));
		ccont_data[cluster_index].emplace_back(get_vindex(data_index), data);
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
	std::vector<layer<T>> layers;
	std::vector<size_t> top_layers; // data_index -> top layer it appears on
	std::vector<std::vector<bool>>
			e_labels; // data index -> cut labels [*num_cuts]
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t>
	_query_k_at_layer_di(const vec<T>& v, size_t k,
											 const std::vector<size_t>& starting_points,
											 const layer<T>& ly);
	const std::vector<VIndex>
	_query_k_at_layer(const vec<T>& v, size_t k,
										const std::vector<VIndex>& starting_points,
										const layer<T>& ly);
	void add_edge(layer<T>& lr, size_t i, size_t j);
	void add_edge_directional_to_cluster(layer<T>& lr, size_t data_index,
																			 CIndex cluster_index);
	void add_edge_directional(layer<T>& lr, size_t data_index_from,
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
		add_param(pl, min_cluster_membership);
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
void clustered_ehnsw_engine<T>::add_edge_directional(layer<T>& lr,
																										 size_t data_index_from,
																										 size_t data_index_to) {
	// TODO check that no cluster containing data_index_to is in the adjacent set
	// already (mabye also containing set) (and that data_index_from !=
	// data_index_to)
	for (CIndex cluster_index : lr.ccont_inv[data_index_to])
		add_edge_directional_to_cluster(lr, data_index_from, cluster_index);
}

template <typename T>
void clustered_ehnsw_engine<T>::add_edge_directional_to_cluster(
		layer<T>& lr, size_t data_index, CIndex cluster_index) {
	// TODO check that cluster_index is not in the adjacent set already
	// TODO consider other distance metrics
	size_t i = lr.get_vindex(data_index);
	// j is the closest element among all elements of the cluster to i, not
	// including i itself
	std::vector<T> cluster_distances;
	for (auto& cluster_elem : lr.ccont[cluster_index]) {
		if (i == cluster_elem)
			cluster_distances.emplace_back(std::numeric_limits<T>::max());
		else
			cluster_distances.emplace_back(
					dist(all_entries[lr.get_data_index(i)],
							 all_entries[lr.get_data_index(cluster_elem)]));
	}
	size_t best_member_index = std::distance(
			cluster_distances.begin(),
			std::min_element(cluster_distances.begin(), cluster_distances.end()));
	size_t j = lr.ccont[cluster_index][best_member_index];

	T d = dist(all_entries[lr.get_data_index(i)],
						 all_entries[lr.get_data_index(j)]);
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
	if (lr.cadj(i).size() + 1 > edge_count_mult) {
		for (size_t cut : cuts) {
			size_t sz = lr.edge_ranks[data_index][cut].size();
			if (cut == found_cut)
				sz++;
			if (sz > min_per_cut) {
				T cur_cut_val = lr.edge_ranks[data_index][cut].begin()->first;
				if (cur_cut_val < max_cut_val) {
					max_cut_val = cur_cut_val;
					max_cut = cut;
				}
			}
		}
		// if that longest edge is longer than our candidate edge, make the swap
		if (-max_cut_val > d) {
			auto iter = lr.edge_ranks[data_index][max_cut].begin();
			size_t edge_index = iter->second;
			deleted_cluster_index = lr.cadj(i)[edge_index];
			lr.cadj_mut(i)[edge_index] = cluster_index;
			// deleted_j = hadj[layer][i][edge_index];
			// hadj[layer][i][edge_index] = j;
			lr.edge_ranks[data_index][max_cut].erase(iter);
			lr.edge_ranks[data_index][found_cut].emplace(-d, edge_index);
		}
	} else {
		lr.cadj_mut(i).emplace_back(cluster_index);
		lr.edge_ranks[data_index][found_cut].emplace(-d, lr.cadj(i).size() - 1);
	}
	// try to bump a bigger edge until stability is reached (if bumping is
	// enabled)
	if (deleted_cluster_index >= 0 && bumping)
		add_edge_directional_to_cluster(lr, data_index,
																		size_t(deleted_cluster_index));
}

template <typename T>
void clustered_ehnsw_engine<T>::add_edge(layer<T>& lr, size_t i, size_t j) {
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

	// assert(cluster_size == 1);
	// assert(min_cluster_membership == 1);
	size_t num_bins = num_cuts + 1;
	for (size_t data_index = 0; data_index < all_entries.size(); ++data_index) {
		for (size_t layer_index = 0; layer_index <= top_layers[data_index];
				 ++layer_index) {
			// for (size_t layer_index = 0; layer_index <= max_depth; ++layer_index) {
			layers[layer_index].add_vertex(data_index, num_bins);
			// TODO remove this commented code
			// as a stub routine (for testing), make clusters of size 1
			// size_t cluster_index = layers[layer_index].add_cluster();
			// layers[layer_index].add_to_cluster(data_index, all_entries[data_index],
			// 																	 cluster_index);
		}
	}

	// TODO do clustering on each layer here
	auto do_clustering = [&](const std::vector<vec<T>>& entries) {
		// build another ANN engine to do clustering
		ehnsw_engine_2<T> clustering_engine(
				ehnsw_engine_2_config(max_depth, edge_count_mult, num_for_1nn, num_cuts,
															min_per_cut, quick_search, bumping, quick_build));
		for (const auto& v : entries)
			clustering_engine.store_vector(v);
		clustering_engine.build();
		// each vector should be present in at least min_cluster_membership clusters
		std::vector<std::vector<size_t>> clusters;
		std::vector<size_t> presence_counts(entries.size());
		for (size_t m_i = 0; m_i < min_cluster_membership; ++m_i) {
			for (size_t local_data_index = 0; local_data_index < entries.size();
					 ++local_data_index) {
				if (presence_counts[local_data_index] < min_cluster_membership) {
					auto cluster = clustering_engine.query_k(entries.at(local_data_index),
																									 cluster_size);
					// add local_data_index itself to the cluster, if not already present
					bool cur_in_cluster = false;
					for (auto entry_index : cluster) {
						if (entry_index == local_data_index) {
							cur_in_cluster = true;
						}
					}
					if (!cur_in_cluster) {
						cluster.pop_back(); // remove furthest element
						cluster.push_back(local_data_index);
					}
					clusters.emplace_back(cluster);
					for (auto entry_index : cluster)
						presence_counts[entry_index]++;
				}
			}
		}
		return clusters;
	};
	for (size_t layer_index = 0; layer_index <= max_depth; ++layer_index) {
		std::vector<vec<T>> layer_entries;
		robin_hood::unordered_flat_map<size_t, size_t> local_to_global_index;
		for (size_t data_index = 0; data_index < all_entries.size(); ++data_index) {
			if (top_layers[data_index] >= layer_index) {
				local_to_global_index[layer_entries.size()] = data_index;
				layer_entries.emplace_back(all_entries[data_index]);
			}
		}
		std::vector<std::vector<size_t>> layer_clusters =
				do_clustering(layer_entries);
		for (auto& cluster : layer_clusters) {
			size_t cluster_index = layers[layer_index].add_cluster();
			for (size_t local_index : cluster) {
				size_t global_data_index = local_to_global_index[local_index];
				layers[layer_index].add_to_cluster(
						global_data_index, all_entries[global_data_index], cluster_index);
			}
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
const std::vector<size_t> clustered_ehnsw_engine<T>::_query_k_at_layer_di(
		const vec<T>& v, size_t k, const std::vector<size_t>& starting_points,
		const layer<T>& lr) {
	std::vector<VIndex> starting_vindeces;
	for (auto& data_index : starting_points)
		starting_vindeces.emplace_back(lr.get_vindex(data_index));
	auto ret_vind = _query_k_at_layer(v, k, starting_vindeces, lr);
	std::vector<size_t> ret;
	for (auto& vind : ret_vind)
		ret.emplace_back(lr.get_data_index(vind));
	return ret;
}
template <typename T>
const std::vector<VIndex> clustered_ehnsw_engine<T>::_query_k_at_layer(
		const vec<T>& v, size_t k, const std::vector<VIndex>& starting_vindeces,
		const layer<T>& lr) {
	// TODO this should be taking a vector of starting points
	std::priority_queue<std::pair<T, VIndex>> top_k;
	std::priority_queue<std::pair<T, VIndex>> to_visit;
	robin_hood::unordered_flat_set<VIndex> visited;
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
	for (const auto& starting_vindex : starting_vindeces)
		visit(dist(v, all_entries[lr.get_data_index(starting_vindex)]),
					starting_vindex);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		// TODO also iterate through lr.ccont_inv maybe (after making it use VIndex)
		for (const CIndex& cluster_index : lr.cadj(cur))
			for (const VIndex& next_vind : lr.ccont.at(cluster_index)) {
				T d_next = dist(v, all_entries[lr.get_data_index(next_vind)]);
				visit(d_next, next_vind);
			}
		// TODO consider the following alternative implementation using ccont_data
		// for (const CIndex& cluster_index : lr.cadj(cur))
		// for (const auto& [next_vind, next_data] :
		//		 lr.ccont_data.at(cluster_index)) {
		//	T d_next = dist(v, next_data);
		//	visit(d_next, next_vind);
		// }
	}
	std::vector<VIndex> ret;
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
	// TODO add a search_k param for before the top full_search_top_layer (e.g. 3)
	size_t max_current_size = 1; // TODO make this 1, or some param, or something
	std::vector<size_t> current = {starting_vertex};
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer_index = layers.size() - 1; layer_index >= 0; --layer_index) {
		size_t layer_k = k;
		if (layer_index > int(full_search_top_layer))
			layer_k = 1;
		ret.push_back(
				_query_k_at_layer_di(v, layer_k, current, layers[layer_index]));
		current = ret.back();
		if (current.size() > max_current_size)
			current.resize(max_current_size);
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
