#pragma once

#include <algorithm>
#include <cassert>
#include <queue>
#include <set>
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
		// if these don't hold, recursion might break
		assert(C > M * cluster_overlap);
		assert(brute_force_size > M * cluster_overlap);
	}
};

template <typename T>
struct static_rcg_engine : public ann_engine<T, static_rcg_engine<T>> {
	struct metanode {
		// TODO use another engine if under brute force size, instead of actual
		// brute forcing
		size_t starting_vertex;
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
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_int_distribution<> distribution;
	size_t M;
	size_t cluster_overlap;
	size_t C;
	size_t brute_force_size;
	size_t ef_search_mult;
	size_t ef_construction;
	metanode root;
	static_rcg_engine(static_rcg_engine_config conf)
			: rd(), gen(0), distribution(0, conf.C - 1), M(conf.M),
				cluster_overlap(conf.cluster_overlap), C(conf.C),
				brute_force_size(conf.brute_force_size),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction), root() {}
	std::vector<vec<T>> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> query_k_at_metanode(metanode& mn, const vec<T>& q,
																					size_t k);
	std::vector<size_t> _query_k(const vec<T>& q, size_t k);
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
		root.starting_vertex = 0; // arbitrary, should be random (later)
		to_build.emplace(root);
	}
	size_t num_built = 0;
	while (!to_build.empty()) {
		metanode& mn = to_build.front().get();
		const std::vector<size_t>& contents = mn.to_global_index;
		to_build.pop();

		// remove duplicates from to_global_index (they WILL occur)
		{
			std::set<size_t> contents_set(mn.to_global_index.begin(),
																		mn.to_global_index.end());
			mn.to_global_index.assign(contents_set.begin(), contents_set.end());
		}

		std::cerr << "Building a new node: num_built=" << num_built++
							<< ", to_build.size()=" << to_build.size()
							<< ", size=" << contents.size() << std::endl;

		std::vector<size_t> cluster_centres;
		for (size_t i = 0; i < contents.size(); ++i) {
			mn.to_local_index[contents[i]] = i;
			if (distribution(gen) == 0)
				cluster_centres.emplace_back(i);
		}
		if (cluster_centres.empty()) { // if centres are empty (unlikely),
																	 // arbitrarily add the first element
			cluster_centres.emplace_back(0);
		}
		mn.clusters.resize(cluster_centres.size());

		// convert starting vertex to a local index (assumed to be global index on
		// entry)
		mn.starting_vertex = mn.to_local_index[mn.starting_vertex];

		if (contents.size() <= brute_force_size) {
			continue;
		}

		// find the nearest clusters for each point, and store them
		// TODO do this recursively with static rcg engine
		hnsw_engine_basic_4_config other_conf(M, 2 * M, ef_search_mult,
																					ef_construction);
		hnsw_engine_basic_4<T> cluster_engine(other_conf);
		for (size_t i = 0; i < cluster_centres.size(); ++i) {
			cluster_engine.store_vector(
					all_entries[mn.to_global_index[cluster_centres[i]]]);
		}
		cluster_engine.build();
		for (size_t i = 0; i < contents.size(); ++i) {
			mn.to_cluster_indices.emplace_back(cluster_engine.query_k(
					all_entries[mn.to_global_index[i]], cluster_overlap));
		}

		// ensure each cluster centre is in its own cluster, just in case (since
		// this is only approximately guaranteed)
		for (size_t i = 0; i < cluster_centres.size(); ++i) {
			bool in_clusters = false;
			for (size_t cluster_index : mn.to_cluster_indices[cluster_centres[i]]) {
				if (cluster_index == i) {
					in_clusters = true;
					break;
				}
			}
			if (!in_clusters) {
				mn.to_cluster_indices[cluster_centres[i]]
						.pop_back(); // get rid of the worst one
				mn.to_cluster_indices[cluster_centres[i]].emplace_back(
						i); // replace it with i
			}
			// initialize cluster starting vertex to a global index
			mn.clusters[i].starting_vertex = contents[cluster_centres[i]];
		}

		// index everything to figure out which cluster to use
		hnsw_engine_basic_4<T> entire_engine(other_conf);
		for (size_t i = 0; i < contents.size(); ++i) {
			entire_engine.store_vector(all_entries[mn.to_global_index[i]]);
		}
		entire_engine.build();

		// actually put everything in the clusters
		for (size_t i = 0; i < contents.size(); ++i) {
			for (size_t cluster_index : mn.to_cluster_indices[i]) {
				mn.clusters[cluster_index].to_global_index.emplace_back(contents[i]);
				// put all approx nearest neighbours in too
				for (size_t local_neighbour_index :
						 entire_engine.query_k(all_entries[mn.to_global_index[i]], M)) {
					if (i != local_neighbour_index)
						mn.clusters[cluster_index].to_global_index.emplace_back(
								contents[local_neighbour_index]);
				}
			}
		}

		// enqueue each cluster for building/indexing
		for (size_t i = 0; i < mn.clusters.size(); ++i) {
			to_build.emplace(mn.clusters[i]);
		}
	}
}

template <typename T>
std::vector<size_t> static_rcg_engine<T>::query_k_at_metanode(metanode& mn,
																															const vec<T>& q,
																															size_t k) {
	topk_t<T> tk(k);
	if (mn.to_global_index.size() <= brute_force_size) {
		for (size_t global_index : mn.to_global_index) {
			tk.consider(dist2(q, all_entries[global_index]), global_index);
		}
		return tk.to_vector();
	}

	// do traversal to get knn
	using measured_data = std::pair<T, size_t>;
	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};

	// TODO entry points should be determined by a recursive metanode
	// be careful with global/local indices

	std::vector<size_t> entry_points = {mn.starting_vertex};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points)
		entry_points_with_dist.emplace_back(dist2(q, all_entries[entry_point]),
																				entry_point);
	// candidates stores local indices (so to_cluster_indices can be used)
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points_with_dist.begin(), entry_points_with_dist.end(),
								 best_elem);
	// nearest stores global indices (so it can be returned easily)
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(worst_elem);
	//(entry_points_with_dist.begin(), entry_points_with_dist.end(),
	// worst_elem);
	for (auto [d, local_index] : entry_points_with_dist) {
		nearest.emplace(d, mn.to_global_index[local_index]);
	}

	while (nearest.size() > k)
		nearest.pop();

	robin_hood::unordered_flat_set<size_t> visited;
	robin_hood::unordered_flat_set<size_t> visited_clusters;
	for (auto& entry_point : entry_points) {
		visited.insert(entry_point);
	}

	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		for (size_t cluster_index : mn.to_cluster_indices[cur.second]) {
			if (visited_clusters.contains(cluster_index))
				continue;
			visited_clusters.insert(cluster_index);
			std::cerr << "Entering neighbour queries for " << cur.second
								<< "(d=" << cur.first << ")" << std::endl;
			for (size_t global_next :
					 query_k_at_metanode(mn.clusters[cluster_index], q, k)) {
				if (!visited.contains(global_next)) {
					visited.insert(global_next);
					T d_next = dist2(q, all_entries[global_next]);
					if (nearest.size() < k || d_next < nearest.top().first) {
						candidates.emplace(d_next, mn.to_local_index[global_next]);
						nearest.emplace(d_next, global_next);
						if (nearest.size() > k)
							nearest.pop();
					}
				}
			}
			std::cerr << "Exiting neighbour queries for " << cur.second
								<< "(d=" << cur.first << ")" << std::endl;
		}
	}
	std::vector<size_t> ret;
	std::cerr << "Returning from a metanode: ret=";
	while (!nearest.empty()) {
		std::cerr << nearest.top().second << "(d=" << nearest.top().first << "),";
		ret.emplace_back(nearest.top().second);
		nearest.pop();
	}
	std::cerr << std::endl;
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> static_rcg_engine<T>::_query_k(const vec<T>& q, size_t k) {
	return query_k_at_metanode(root, q, k * ef_search_mult);
}
