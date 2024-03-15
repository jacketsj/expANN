#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "clustering.h"
#include "robin_hood.h"
#include "topk_t.h"

#include "antitopo_engine.h"

struct line_pruning_exact_engine_config {
	size_t brute_force_size;
	size_t line_count;
	antitopo_engine_config sub_conf;
	line_pruning_exact_engine_config(size_t _brute_force_size, size_t _line_count,
																	 antitopo_engine_config _sub_conf)
			: brute_force_size(_brute_force_size), line_count(_line_count),
				sub_conf(_sub_conf) {}
};

template <typename T>
struct line_pruning_exact_engine
		: public ann_engine<T, line_pruning_exact_engine<T>> {
private:
	template <typename A, typename B> auto dist2(const A& a, const B& b) {
		return (a - b).squaredNorm();
	}

public:
	using fvec = typename vec<T>::Underlying;
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	struct line {
		// TODO a is completely unnecessary I think
		fvec a, dir;
		line(const fvec& a, const fvec& b) : a(a), dir((b - a).normalized()) {}
		T proj(fvec v) const { return dir.dot(v - a); }
	};
	struct line_with_points {
		line l;
		std::set<std::pair<T, size_t>> sorted_elems;
		line_with_points(const line& l, const std::vector<size_t>& elems,
										 const std::vector<fvec>& entries)
				: l(l) {
			for (size_t i : elems) {
				sorted_elems.emplace(l.proj(entries[i]), i);
			}
		}
		float get_lb(const fvec& v) const {
			// TODO allow filtering here, so that the clusters containing the best
			// known elements don't inherently need to be recursed in
			// Could even filter an entire visited list from preceeding ANN search
			// (possibly too slow)
			auto projected_v = l.proj(v);
			auto it = sorted_elems.lower_bound(std::make_pair(projected_v, 0));
			auto it_dist = [&]() { return std::abs(it->first - projected_v); };
			if (it == sorted_elems.begin())
				return it_dist();
			auto pit = std::prev(it);
			auto pit_dist = [&]() { return std::abs(pit->first - projected_v); };
			if (it == sorted_elems.end())
				return pit_dist();
			return std::min(it_dist(), pit_dist());
		}
	};
	struct cluster_tree_node {
		std::vector<size_t> elems;
		std::vector<fvec> entries;
		std::vector<line_with_points> lines;
		std::vector<size_t> children;
		bool leaf;
		cluster_tree_node(const std::vector<size_t>& elems,
											const std::vector<fvec>& entries)
				: elems(elems), entries(entries) {}
		float get_lb(const fvec& v) const {
			// TODO when getting a lb, also return the corresponding index, so that it
			// can be tested to see if it's a new best approximate nearest neighbour
			// (and a new graph search can be initiated with it as a seed if so)
			float best = 0;
			for (const auto& lp : lines) {
				best = std::max(best, lp.get_lb(v));
			}
			return best * best; // return as squared dist
		}
		void add_line(const fvec& a, const fvec& b) {
			line l(a, b);
			lines.emplace_back(l, elems, entries);
		}
	};
	size_t brute_force_size;
	size_t line_count;
	antitopo_engine_config sub_conf;
	antitopo_engine<T> sub_engine;
	std::vector<cluster_tree_node> cluster_tree;
	line_pruning_exact_engine(line_pruning_exact_engine_config conf)
			: rd(), gen(0), distribution(0, 1),
				brute_force_size(conf.brute_force_size), line_count(conf.line_count),
				sub_conf(sub_conf), sub_engine(conf.sub_conf) {}
	using config = line_pruning_exact_engine_config;
	std::vector<fvec> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Line-Pruning Exact Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, brute_force_size);
		add_param(pl, line_count);
		return pl;
	}
};

template <typename T>
void line_pruning_exact_engine<T>::_store_vector(const vec<T>& v0) {
	auto v = v0.internal;
	all_entries.emplace_back(v);
}

template <typename T> void line_pruning_exact_engine<T>::_build() {
	// TODO-critical step 1a make a hierarchical clustering of the data
	std::queue<size_t> to_build;
	auto make_cluster_tree_node = [&](const std::vector<size_t>& elems) {
		std::vector<fvec> entries;
		for (const auto& e : elems)
			entries.emplace_back(all_entries[e]);
		cluster_tree_node cluster(elems, entries);
		if (elems.size() <= brute_force_size)
			cluster.leaf = true;
		size_t new_index = cluster_tree.size();
		cluster_tree.emplace_back(cluster);
		to_build.emplace(new_index);
		return new_index;
	};
	{
		std::vector<size_t> all_elems;
		for (size_t i = 0; i < all_entries.size(); ++i)
			all_elems.emplace_back(i);
		make_cluster_tree_node(all_elems);
	}
	while (!to_build.empty()) {
		size_t cindex = to_build.front();
		to_build.pop();
		if (cluster_tree[cindex].leaf) {
			continue;
		}
		auto clustering_local_indices =
				accel_k_means(cluster_tree[cindex].entries, 16, sub_conf);
		for (const auto& sub_cluster : clustering_local_indices) {
			std::vector<size_t> sub_cluster_contents_global;
			for (const auto& local_elem : sub_cluster)
				sub_cluster_contents_global.emplace_back(
						cluster_tree[cindex].elems[local_elem]);
			cluster_tree[cindex].children.emplace_back(
					make_cluster_tree_node(sub_cluster_contents_global));
		}
	}

	for (auto& node : cluster_tree) {
		if (node.leaf)
			continue; // always scan through everything in a leaf
		std::uniform_int_distribution<> int_distribution(0, node.elems.size() - 1);
		for (size_t i = 0; i < line_count; ++i) {
			auto v1 = all_entries[node.elems[int_distribution(gen)]];
			auto v2 = all_entries[node.elems[int_distribution(gen)]];
			node.add_line(v1, v2);
			// TODO consider adding lines through the centroid too
		}
		// TODO consider adding the lines corresponding to edges in a mesh graph
	}
	for (const auto& v : all_entries) {
		sub_engine.store_vector(vec<T>(v));
	}
	sub_engine.build();
}

template <typename T>
std::vector<size_t> line_pruning_exact_engine<T>::_query_k(const vec<T>& q0,
																													 size_t k) {
	const auto& q = q0.internal;
	std::unordered_set<size_t> nearest_data; // TODO replace with a visited list,
																					 // similar to antitopo engine/hnsw
																					 // (share it with sub_engine even)
	for (const auto& data_index : sub_engine.query_k(q0, k)) {
		nearest_data.emplace(data_index);
	}
	using measured_data = std::pair<T, size_t>;
	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(worst_elem);
	for (const auto& data_index : nearest_data) {
		nearest.emplace(dist2(q, all_entries[data_index]), data_index);
	}
	while (nearest.size() > k) {
		nearest_data.erase(nearest.top().second);
		nearest.pop();
	}
	std::queue<size_t> clusters_to_traverse;
	clusters_to_traverse.emplace(0);
	// TODO figure out best candidate clusters (by lower bound?), and traverse
	// those first. Save new NN value in the to_traverse priority_queue in order
	// to prune better.
	while (!clusters_to_traverse.empty()) {
		size_t current_cluster = clusters_to_traverse.front();
		clusters_to_traverse.pop();
		const auto& cluster_node = cluster_tree[current_cluster];
		if (cluster_node.get_lb(q) < nearest.top().first) {
			if (cluster_node.leaf) {
				for (const size_t& elem_index : cluster_node.elems) {
					T cur_dist = dist2(all_entries[elem_index], q);
					if (cur_dist < nearest.top().first &&
							!nearest_data.contains(elem_index)) {
						// TODO continue ANN search with elem_index as an additional seed
						// point. Call query_k_at_layer with the previous visited list (add
						// a setting to not clear it, or to feed a special one), and
						// everything in nearest as entry_points
						nearest.emplace(cur_dist, elem_index);
						nearest_data.emplace(elem_index);
						if (nearest.size() > k) {
							nearest_data.erase(nearest.top().second);
							nearest.pop();
						}
					}
				}
			} else {
				for (const auto& child : cluster_node.children)
					clusters_to_traverse.emplace(child);
			}
		}
	}
	std::vector<size_t> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top().second);
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}
