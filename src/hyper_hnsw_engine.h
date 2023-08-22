#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"

struct hyper_hnsw_engine_config {
	size_t max_depth;
	// For n vertices in a layer, total number of clusters will be approximately
	// n*degree_node/degree_cluster
	size_t degree_cluster; // number of vertices indexed by each cluster
	size_t degree_node;		 // number of clusters indexed by each node
	// double cluster_count_constant;
	// // cluster_count_constant is a value >0 which gives the number of clusters
	// as
	// // max(1, size_t(cluster_count_constant * n*degree_node/degree_cluster))
	size_t num_for_1nn;
	hyper_hnsw_engine_config(size_t _max_depth, size_t _degree_cluster,
													 size_t _degree_node,
													 // double _cluster_count_constant,
													 size_t _num_for_1nn)
			: max_depth(_max_depth), degree_cluster(_degree_cluster),
				degree_node(_degree_node),
				// cluster_count_constant(_cluster_count_constant),
				num_for_1nn(_num_for_1nn) {}
};

template <typename T> struct simple_hypergraph {
	struct vertex {
		size_t data_index; // TODO don't store this for clusters, only nodes
		std::vector<size_t> adj;
		vertex(size_t _data_index) : data_index(_data_index) {}
	};
	struct vertex_with_ranks : vertex {
		// allow for dynamic changes to neighbours, keeping only the best
		using vertex::adj;
		using vertex::data_index;
		vertex_with_ranks(size_t _data_index) : vertex::vertex(_data_index) {}
		std::priority_queue<std::pair<T, size_t>> ranks;
		void add(size_t new_neighbour, T rank_val, size_t max_degree) {
			bool should_add = adj.size() < max_degree;
			should_add = should_add || ranks.top().first > rank_val;
			if (should_add) {
				size_t location = adj.size();
				if (adj.size() >= max_degree) {
					location = ranks.top().second;
					adj[location] = new_neighbour;
					ranks.pop();
				} else {
					adj.emplace_back(new_neighbour);
				}
				ranks.emplace(rank_val, location);
			}
		}
	};
	// hypergraph represented as bipartite graph between
	// clusters(hyperedges)/nodes(vertices)
	std::vector<vertex_with_ranks> clusters;
	std::vector<vertex_with_ranks> nodes;
	const size_t degree_cluster;
	const size_t degree_node;
	simple_hypergraph(size_t _degree_cluster, size_t _degree_node)
			: degree_cluster(_degree_cluster), degree_node(_degree_node) {}
	std::vector<size_t> get_neighbours(const size_t& node_index) {
		std::vector<size_t> ret;
		for (size_t cluster_index : nodes[node_index].adj) {
			for (size_t neighbour_node_index : clusters[cluster_index].adj)
				ret.emplace_back(neighbour_node_index);
		}
		return ret;
	}
	size_t get_data_index(const size_t& node_index) {
		return nodes[node_index].data_index;
	}
	std::vector<size_t> get_cluster_contents(const size_t& cluster_index) {
		std::vector<size_t> ret;
		for (size_t node_index : clusters[cluster_index])
			ret.emplace_back(node_index);
		return ret;
	}
	// need to be able to translate between different hypergraphs
	std::vector<size_t> next_layer_node_indexes;
	void add_next_layer_node_index(size_t node_index,
																 size_t next_layer_node_index) {
		next_layer_node_indexes[node_index] = next_layer_node_index;
	}
	size_t get_next_layer_node_index(size_t node_index) {
		return next_layer_node_indexes[node_index];
	}
	size_t add_node(const size_t& data_index) {
		size_t ret = nodes.size();
		nodes.emplace_back(data_index);
		next_layer_node_indexes.emplace_back(
				data_index); // initialize this to the data index
		return ret;
	}
	// TODO don't require data to be stored in clusters
	size_t add_cluster() {
		size_t ret = clusters.size();
		clusters.emplace_back(0);
		return ret;
	}
	void add_to_cluster(size_t node_index, size_t cluster_index, T rank_val) {
		if (node_index >= nodes.size()) {
			// intentionally cause a segfault to get stack trace since abort isn't
			// working
			static int* thingy = NULL;
			thingy[4] = 12;
		}
		clusters[cluster_index].add(node_index, rank_val, degree_cluster);
		nodes[node_index].add(cluster_index, rank_val, degree_node);
	}
};

template <typename T>
struct hyper_hnsw_engine : public ann_engine<T, hyper_hnsw_engine<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_node_index;
	size_t max_depth;
	size_t degree_cluster;
	size_t degree_node;
	// double cluster_count_constant;
	size_t num_for_1nn;
	hyper_hnsw_engine(hyper_hnsw_engine_config conf)
			: rd(), gen(rd()), distribution(0, 1), max_depth(conf.max_depth),
				degree_cluster(conf.degree_cluster), degree_node(conf.degree_node),
				// cluster_count_constant(conf.cluster_count_constant),
				num_for_1nn(conf.num_for_1nn) {}
	std::vector<vec<T>> all_entries;
	std::vector<simple_hypergraph<T>> hypergraphs; // layer -> hypergraph
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k_at_layer(const vec<T>& v, size_t k,
																							size_t starting_point,
																							size_t layer);
	const std::vector<std::vector<size_t>>
	_query_k_internal(const vec<T>& v, size_t k, bool fill_all_layers = false);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Hyper HNSW Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, degree_cluster);
		add_param(pl, degree_node);
		// add_param(pl, cluster_count_constant);
		add_param(pl, num_for_1nn);
		return pl;
	}
};

template <typename T>
void hyper_hnsw_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

// initialize the clusters that would be consistent with HNSW:
// choose some (random) subset of the nodes. Get the
// degree_node*(degree_cluster-1) nearest neighbours for each. Randomly
// partition this into degree_node different clusters.
// Then do K-means training iterations basically:
// For each cluster, get its mean, then get the degree_cluster nearest
// neighbours to it, and replace the elements of the cluster by calling
// add_to_cluster (which updates both directions of edges)
template <typename T> void hyper_hnsw_engine<T>::_build() {
	auto add_layer = [&](size_t v) {
		hypergraphs.emplace_back(degree_cluster, degree_node);
	};
	// add one layer to start, with first vertex
	add_layer(0);
	starting_node_index = hypergraphs[0].add_node(0);
	for (size_t i = 1; i < all_entries.size(); ++i) {
		if (i % 5000 == 0)
			std::cerr << "Built " << double(i) / double(all_entries.size()) * 100
								<< "%" << std::endl;
		// get kNN at each layer
		size_t k = degree_node * (degree_cluster - 1);
		// TODO should also be getting some "k nearest clusters" or similar
		// (also suffices to try adding to every cluster visited with the distance
		// to that cluster's mean)
		std::vector<std::vector<size_t>> kNN =
				_query_k_internal(all_entries[i], k, true);
		// get the layer this entry will go up to
		size_t cur_layer_ub =
				floor(-log(distribution(gen)) * 1 / log(double(degree_node)));
		// floor(-log(distribution(gen)) * 1 /
		// log(double(degree_node*degree_cluster)));
		size_t cur_layer = std::min(cur_layer_ub, max_depth);
		// if it is a new layer, add a layer
		bool is_new_top = false;
		while (cur_layer >= hypergraphs.size()) {
			add_layer(i);
			is_new_top = true;
		}
		std::vector<size_t> layer_to_node_index;
		size_t next_layer_node_index =
				i; // for layer=0, this should be the data_index
		for (size_t layer = 0; layer <= cur_layer; ++layer) {
			size_t node_index = hypergraphs[layer].add_node(i);
			hypergraphs[layer].add_next_layer_node_index(node_index,
																									 next_layer_node_index);
			next_layer_node_index = node_index;
			layer_to_node_index.emplace_back(node_index);
		}
		if (is_new_top)
			starting_node_index = next_layer_node_index;
		// add all the neighbours as edges
		for (size_t layer = 0; layer <= cur_layer && layer < kNN.size(); ++layer) {
			size_t node_index = layer_to_node_index[layer];
			//  TODO do k-means or something like it (b-matching?) instead of
			//  random_shuffle
			random_shuffle(kNN[layer].begin(), kNN[layer].end());
			size_t neighbours_i = 0;
			for (size_t local_cluster_index = 0; local_cluster_index < degree_node;
					 ++local_cluster_index) {
				if (neighbours_i >= kNN[layer].size())
					break;
				// create new cluster
				std::vector<size_t>
						cluster_elems; // elems to add to cluster (node indexes)
				cluster_elems.push_back(node_index);
				for (size_t cur_neighbours_i = neighbours_i;
						 cur_neighbours_i < kNN[layer].size() &&
						 cur_neighbours_i < degree_cluster + neighbours_i;
						 ++cur_neighbours_i) {
					cluster_elems.emplace_back(kNN[layer][cur_neighbours_i]);
				}
				neighbours_i += degree_cluster;
				// compute mean
				vec<T> mean;
				mean.set_dim(all_entries[0].dim());
				for (size_t cur_node_index : cluster_elems)
					mean +=
							all_entries[hypergraphs[layer].get_data_index(cur_node_index)];
				mean /= T(cluster_elems.size());
				size_t cluster_index = hypergraphs[layer].add_cluster();
				for (size_t cur_node_index : cluster_elems) {
					hypergraphs[layer].add_to_cluster(
							cur_node_index, cluster_index,
							dist2(mean, all_entries[hypergraphs[layer].get_data_index(
															cur_node_index)]));
				}
			}
		}
		// TODO do a training run now (just on the new clusters), or maybe a few
		// even, for each new cluster:
		// - re-compute means
		// 	   (add a function which takes a vec<T> and computes distance from it)
		// - discard existing ranks in clusters and recompute
		// - compute kNN again (from the mean location, for k=degree_cluster)
		// - call add_to_cluster with new ranks
	}
}

// TODO do the stuff below

template <typename T>
const std::vector<size_t>
hyper_hnsw_engine<T>::_query_k_at_layer(const vec<T>& v, size_t k,
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
	visit(dist(v, all_entries[hypergraphs[layer].get_data_index(starting_point)]),
				starting_point);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (auto& u : hypergraphs[layer].get_neighbours(cur)) {
			T d_next = dist(v, all_entries[hypergraphs[layer].get_data_index(u)]);
			visit(d_next, u);
		}
	}
	std::vector<size_t> ret;
	while (!top_k.empty()) {
		ret.push_back(top_k.top().second);
		// hypergraphs[layer].get_next_layer_node_index(top_k.top().second));
		top_k.pop();
	}
	reverse(ret.begin(), ret.end()); // sort from closest to furthest
	return ret;
}

template <typename T>
const std::vector<std::vector<size_t>>
hyper_hnsw_engine<T>::_query_k_internal(const vec<T>& v, size_t k,
																				bool fill_all_layers) {
	auto current = starting_node_index;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer = hypergraphs.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (!fill_all_layers && layer > 0)
			layer_k = 1;
		ret.push_back(_query_k_at_layer(v, layer_k, current, layer));
		current = hypergraphs[layer].get_next_layer_node_index(ret.back().front());
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> hyper_hnsw_engine<T>::_query_k(const vec<T>& v, size_t k) {
	auto ret = _query_k_internal(v, k * num_for_1nn, false)[0];
	ret.resize(std::min(k, ret.size()));
	for (size_t& index : ret) {
		// convert from final layer's node indices to data indices
		index = hypergraphs[0].get_data_index(index);
	}
	return ret;
}
