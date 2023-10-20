#pragma once

#include <algorithm>
#include <cassert>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "hnsw_engine_basic_4.h"
#include "robin_hood.h"
#include "topk_t.h"

struct static_rcg_engine_config {
	size_t M;
	size_t cluster_overlap;
	size_t C;
	size_t brute_force_size;
	size_t ef_search_mult;
	size_t ef_construction;
	static_rcg_engine_config(size_t _M, size_t _cluster_overlap, size_t _C,
													 size_t _brute_force_size, size_t _ef_search_mult,
													 size_t _ef_construction)
			: M(_M), cluster_overlap(_cluster_overlap), C(_C),
				brute_force_size(_brute_force_size), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction) {
		assert(C >
					 M * cluster_overlap); // if this doesn't hold, recursion might break
	}
};

template <typename T>
struct static_rcg_engine : public ann_engine<T, static_rcg_engine<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_int_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t cluster_overlap;
	size_t C;
	size_t brute_force_size;
	size_t ef_search_mult;
	size_t ef_construction;
	static_rcg_engine(static_rcg_engine_config conf)
			: rd(), gen(0), distribution(0, conf.C - 1), M(conf.M),
				cluster_overlap(conf.cluster_overlap), C(conf.C),
				brute_force_size(conf.brute_force_size),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction) {}
	std::vector<vec<T>> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	struct metanode {
		robin_hood::unordered_flat_map<size_t, size_t>
				to_local_index;									 // global index -> metanode local index
		std::vector<size_t> to_global_index; // metanode local index -> global index
		// build other ann engine here, indexing local indexing, dont need to save
		// it
		std::vector<metanode>
				clusters; // metanode centre index (approximatley n/C) -> metanode
									// indexing cluster with M neighbours per element
		std::vector<std::vector<size_t>>
				to_cluster_indices; // metanode local index ->
														// metanode centre indices (membership)
	};
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Static RCG Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, cluster_overlap);
		add_param(pl, C);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		return pl;
	}
	metanode root;
};

template <typename T>
void static_rcg_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void static_rcg_engine<T>::_build() {
	assert(all_entries.size() > 0);

	std::queue<std::reference_wrapper<metanode>> to_build;
	{
		for (size_t i = 0; i < all_entries.size(); ++i)
			root.to_global_index.emplace_back(i);
		to_build.emplace(root);
	}
	while (!to_build.empty()) {
		metanode& mn = to_build.front().get();
		const std::vector<size_t>& contents = mn.to_global_index;
		to_build.pop();

		std::vector<size_t> cluster_centres;
		for (size_t i = 0; i < contents.size(); ++i) {
			mn.to_local_index[contents[i]] = i;
			if (distribution(gen) == 0)
				cluster_centres.emplace_back(i);
		}
		mn.clusters.resize(cluster_centres.size());

		if (contents.size() <= brute_force_size) {
			continue;
		}

		// find the nearest clusters for each point, and store them
		// TODO do this recursively with static rcg engine
		hnsw_engine_basic_4_config cluster_conf(M, 2 * M, ef_search_mult,
																						ef_construction);
		hnsw_engine_basic_4<T> cluster_engine(cluster_conf);
		for (size_t i = 0; i < cluster_centres.size(); ++i) {
			cluster_engine.store_vector(
					all_entries[mn.to_global_index[cluster_centres[i]]]);
		}
		cluster_engine.build();
		for (size_t i = 0; i < contents.size(); ++i) {
			mn.to_cluster_indices.emplace_back(cluster_engine.query_k(
					all_entries[mn.to_global_index[i]], cluster_overlap));
			for (size_t cluster_index : mn.to_cluster_indices[i])
				mn.clusters[cluster_index].to_global_index.emplace_back(contents[i]);
		}

		// enqueue each cluster for building/indexing
		for (size_t i = 0; i < mn.clusters.size(); ++i)
			to_build.emplace(mn.clusters[i]);
	}
}

template <typename T>
std::vector<size_t> static_rcg_engine<T>::_query_k(const vec<T>& q, size_t k) {
	// TODO
	return {};
}
