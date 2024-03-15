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
#include "robin_hood.h"
#include "topk_t.h"

namespace {

template <typename A, typename B> auto dist2(const A& a, const B& b) {
	return (a - b).squaredNorm();
}
} // namespace

struct line_pruning_exact_engine_config {
	size_t brute_force_size;
	size_t line_count;
	line_pruning_exact_engine_config(size_t _brute_force_size, size_t _line_count)
			: brute_force_size(_brute_force_size), line_count(_line_count) {}
};

template <typename T>
struct line_pruning_exact_engine
		: public ann_engine<T, line_pruning_exact_engine<T>> {
	using fvec = typename vec<T>::Underlying;
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	struct line {
		// TODO a is completely unnecessary I think
		fvec a, dir;
		line(const fvec& a, const fvec& b) : a(a), dir((b - a).normalized()) {}
		T proj(fvec v) const { return dot(v - a, dir); }
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
		std::vector<line_with_points> lines;
		std::vector<size_t> children;
		bool leaf;
		float get_lb(const fvec& v) {
			// TODO when getting a lb, also return the corresponding index, so that it
			// can be tested to see if it's a new best approximate nearest neighbour
			// (and a new graph search can be initiated with it as a seed if so)
			float best = 0;
			for (const auto& lp : lines) {
				best = std::max(best, lp.get_lb(v));
			}
			return best;
		}
	};
	size_t brute_force_size;
	size_t line_count;
	std::vector<cluster_tree_node> cluster_tree;
	line_pruning_exact_engine(line_pruning_exact_engine_config conf)
			: rd(), gen(0), distribution(0, 1),
				brute_force_size(conf.brute_force_size), line_count(conf.line_count) {}
	using config = line_pruning_exact_engine_config;
	std::vector<fvec> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
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
	// TODO step 1a make a hierarchical clustering of the data
	// TODO step 1b sample lines within each cluster
	// TODO step 2 build a sub-engine
}

template <typename T>
std::vector<size_t> line_pruning_exact_engine<T>::_query_k(const vec<T>& q0,
																													 size_t k) {
	const auto& q = q0.internal;
	// TODO step 1 query the sub-engine to get an approximate nearest neighbour
	// with high likelyhood of being the best
	using measured_data = std::pair<T, size_t>;
	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(worst_elem);
	// TODO step 2 traverse the hierarchical clustering stored in cluster_tree
	// (starting from index 0):
	// - Check if the lower bound of a cluster precludes it from being a possible
	// choice
	// - If it doesn't, recurse into it
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
					if (cur_dist < nearest.top().first()) {
						nearest.emplace(cur_dist, elem_index);
						if (nearest.size() > k)
							nearest.pop();
					}
				}
			} else {
				for (const auto& child : cluster_node.children)
					clusters_to_traverse.emplace(child);
			}
		}
	}
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}
