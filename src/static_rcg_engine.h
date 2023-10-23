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
	size_t rC;
	size_t brute_force_size;
	size_t ef_search_mult;
	size_t ef_construction;
	static_rcg_engine_config(size_t _M, size_t _cluster_overlap, size_t _C,
													 size_t _rC, size_t _brute_force_size,
													 size_t _ef_search_mult, size_t _ef_construction)
			: M(_M), cluster_overlap(_cluster_overlap), C(_C), rC(_rC),
				brute_force_size(_brute_force_size), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction) {
		// TODO figure out what these should actually be
		// if these don't hold, recursion might break
		// assert(C > M * cluster_overlap);
		// assert(brute_force_size > M * cluster_overlap);
	}
};

template <typename T>
struct static_rcg_engine : public ann_engine<T, static_rcg_engine<T>> {
	struct metanode {
		// TODO use another engine if under brute force size, instead of actual
		// brute forcing
		std::unique_ptr<metanode>
				recursed_elems; // 1 in rC clusters get indexed here
		robin_hood::unordered_flat_map<size_t, size_t>
				to_local_index;									 // global index -> metanode local index
		std::vector<size_t> to_global_index; // metanode local index -> global index
		// build other ann engine here, indexing local indexing, dont need to save
		// it
		std::vector<metanode>
				clusters; // metanode centre index (approximatley n/C) -> metanode
									// indexing cluster with M neighbours per element
		std::vector<std::vector<size_t>>
				to_cluster_indices; // metanode local index -> metanode centre indices
														// (membership)
		size_t depth;
	};
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_int_distribution<> distribution, distribution_r;
	size_t M;
	size_t cluster_overlap;
	size_t C;
	size_t rC;
	size_t brute_force_size;
	size_t ef_search_mult;
	size_t ef_construction;
#ifdef RECORD_STATS
	size_t num_distcomps;
#endif
	metanode root;
	static_rcg_engine(static_rcg_engine_config conf)
			: rd(), gen(0), distribution(0, conf.C - 1),
				distribution_r(0, conf.rC - 1), M(conf.M),
				cluster_overlap(conf.cluster_overlap), C(conf.C), rC(conf.rC),
				brute_force_size(conf.brute_force_size),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction), root() {}
	std::vector<vec<T>> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<std::pair<T, size_t>>
	query_k_at_metanode(metanode& mn, const vec<T>& q, size_t k);
	std::vector<size_t> _query_k(const vec<T>& q, size_t k);
	const std::string _name() { return "Static RCG Engine"; }
	size_t num_clusters(const metanode& mn);
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, cluster_overlap);
		add_param(pl, C);
		add_param(pl, rC);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
#ifdef RECORD_STATS
		add_param(pl, num_distcomps);
#endif
		return pl;
	}
};

template <typename T>
void static_rcg_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T>
size_t static_rcg_engine<T>::num_clusters(const metanode& mn) {
	auto num_elems = mn.to_global_index.size();
	auto depth = mn.depth;
	size_t divisor = C;
	if (depth >= 1)
		divisor = 1;
	// while (depth > 0) {
	//	divisor /= 2;
	//	--depth;
	// }
	// if (divisor < 1)
	//	divisor = 1;
	return num_elems / divisor;
	// return std::floor(std::sqrt(num_elems)) / C;
	// return std::max(num_elems / C, C);
	// if (num_elems / C <= C)
	// return num_elems;
	// return num_elems / C; // C;
}

template <typename T> void static_rcg_engine<T>::_build() {
	assert(all_entries.size() > 0);

	std::stack<std::reference_wrapper<metanode>> to_build;
	{
		for (size_t i = 0; i < all_entries.size(); ++i)
			root.to_global_index.emplace_back(i);
		root.depth = 0;
		to_build.emplace(root);
	}
	auto emplace_to_build = [&](metanode& mn) {
		// remove duplicates from to_global_index (they WILL occur)
		{
			std::set<size_t> contents_set(mn.to_global_index.begin(),
																		mn.to_global_index.end());
			mn.to_global_index.assign(contents_set.begin(), contents_set.end());
		}
		to_build.emplace(mn);
	};
	size_t num_built = 0;
	size_t total_builds = 0;
	while (!to_build.empty()) {
		metanode& mn = to_build.top().get();
		const std::vector<size_t>& contents = mn.to_global_index;
		to_build.pop();

		for (size_t i = 0; i < contents.size(); ++i) {
			mn.to_local_index[contents[i]] = i;
		}

		if ((total_builds + contents.size()) / 500000 > (total_builds / 500000) ||
				contents.size() >= 5000) {
			//  if ((num_built++) % 5000 == 0) {
			std::cerr << "Building a new metanode: num_built=" << num_built
								<< ", to_build.size()=" << to_build.size()
								<< ", size=" << contents.size() << ", depth=" << mn.depth
								<< ", accumulated_buildsize=" << total_builds << std::endl;
		}
		total_builds += contents.size();
		num_built++;

		if (contents.size() <= brute_force_size) {
			continue;
		}

		std::vector<size_t> cluster_centres;
		for (size_t i = 0; i < contents.size(); ++i)
			cluster_centres.emplace_back(i);
		std::shuffle(cluster_centres.begin(), cluster_centres.end(), gen);
		cluster_centres.resize(
				std::min(cluster_centres.size(), // in case there is only 1 element
								 std::max(size_t(2), num_clusters(mn))));
		mn.clusters.resize(cluster_centres.size());

		// figure out which elements to put in the "higher level"
		std::vector<size_t> recursed_elems;
		for (size_t i = 0; i < contents.size(); ++i) {
			if (distribution_r(gen) == 0) {
				recursed_elems.emplace_back(i);
			}
		}
		// if recursed elems are empty (unlikely), arbitrarily add the first
		// element
		if (recursed_elems.empty()) {
			recursed_elems.emplace_back(0);
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
				size_t num_used = 0;
				for (size_t local_neighbour_index : entire_engine.hadj_bottom[i]) {
					if (i != local_neighbour_index) {
						if (num_used++ < M / (mn.depth + 1))
							mn.clusters[cluster_index].to_global_index.emplace_back(
									contents[local_neighbour_index]);
					}
				}
				/*
				for (size_t local_neighbour_index : entire_engine.query_k(
								 all_entries[mn.to_global_index[i]], ef_construction)) {
					if (i != local_neighbour_index)
						mn.clusters[cluster_index].to_global_index.emplace_back(
								contents[local_neighbour_index]);
				}
				*/
			}
		}

		// enqueue each cluster for building/indexing
		for (size_t i = 0; i < mn.clusters.size(); ++i) {
			mn.clusters[i].depth = mn.depth + 1;
			emplace_to_build(mn.clusters[i]);
		}

		// enqueue the recursed elements
		mn.recursed_elems = std::make_unique<metanode>();
		for (size_t i : recursed_elems) {
			mn.recursed_elems->depth = mn.depth + 1;
			mn.recursed_elems->to_global_index.emplace_back(mn.to_global_index[i]);
		}
		emplace_to_build(*mn.recursed_elems);
	}
#ifdef RECORD_STATS
	// reset before queries
	num_distcomps = 0;
#endif
}

template <typename T>
std::vector<std::pair<T, size_t>>
static_rcg_engine<T>::query_k_at_metanode(metanode& mn, const vec<T>& q,
																					size_t k) {
	topk_t<T> tk(k);
	if (mn.to_global_index.size() <= brute_force_size) {
		for (size_t global_index : mn.to_global_index) {
#ifdef RECORD_STATS
			++num_distcomps;
#endif
			tk.consider(dist2(q, all_entries[global_index]), global_index);
		}
		return tk.to_combined_vector();
	}

	// do traversal to get knn
	using measured_data = std::pair<T, size_t>;
	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};

	std::vector<std::pair<T, size_t>> entry_points_with_dist =
			query_k_at_metanode(*mn.recursed_elems, q, 1);
	for (auto& [_, index] : entry_points_with_dist) {
		index = mn.to_local_index[index];
	}
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
	for (auto& [_, entry_point] : entry_points_with_dist) {
		visited.insert(entry_point);
	}

	// TODO use a stack to avoid the recursion, or something
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
			// std::cerr << "Entering neighbour queries for " << cur.second
			//					<< "(d=" << cur.first << ")" << std::endl;
			for (const auto& [d_next, global_next] :
					 query_k_at_metanode(mn.clusters[cluster_index], q, k)) {
				if (!visited.contains(global_next)) {
					visited.insert(global_next);
					if (nearest.size() < k || d_next < nearest.top().first) {
						candidates.emplace(d_next, mn.to_local_index[global_next]);
						nearest.emplace(d_next, global_next);
						if (nearest.size() > k)
							nearest.pop();
					}
				}
			}
			// std::cerr << "Exiting neighbour queries for " << cur.second
			//					<< "(d=" << cur.first << ")" << std::endl;
		}
	}
	std::vector<std::pair<T, size_t>> ret;
	// std::cerr << "Returning from a metanode: ret=";
	while (!nearest.empty()) {
		// std::cerr << nearest.top().second << "(d=" << nearest.top().first <<
		// "),";
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	// std::cerr << std::endl;
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> static_rcg_engine<T>::_query_k(const vec<T>& q, size_t k) {
	std::vector<size_t> ret;
	auto ret_combined = query_k_at_metanode(root, q, k * ef_search_mult);
	ret.reserve(ret_combined.size());
	for (auto& [_, ind] : ret_combined)
		ret.emplace_back(ind);
	return ret;
}
