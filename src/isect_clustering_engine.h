#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "randomgeometry.h"
#include "topk_t.h"
#include "vecset.h"

struct isect_clustering_engine_config {
	size_t tree_copies, max_leaf_size, search_count_per_copy, num_clusters,
			num_isect, max_depth, cluster_overlap;
	isect_clustering_engine_config(size_t _tree_copies, size_t _max_leaf_size,
																 size_t _search_count_per_copy,
																 size_t _num_clusters = 8,
																 size_t _num_isect = 8, size_t _max_depth = 40,
																 size_t _cluster_overlap = 2)
			: tree_copies(_tree_copies), max_leaf_size(_max_leaf_size),
				search_count_per_copy(_search_count_per_copy),
				num_clusters(_num_clusters), num_isect(_num_isect),
				max_depth(_max_depth), cluster_overlap(_cluster_overlap) {}
};

typedef gch::small_vector<unsigned short> clustering_location;

// idea: create a tree of intersection clusterings. Each cell with too many
// points becomes a child. should have multiple copies of the tree (with
// different seeds) to avoid edge problems this is similar to using random
// quadtrees
template <typename T>
struct isect_clustering_engine
		: public ann_engine<T, isect_clustering_engine<T>> {
	size_t tree_copies, max_leaf_size, search_count_per_copy, num_clusters,
			num_isect, max_depth, cluster_overlap;
	isect_clustering_engine(isect_clustering_engine_config conf)
			: tree_copies(conf.tree_copies), max_leaf_size(conf.max_leaf_size),
				search_count_per_copy(conf.search_count_per_copy),
				num_clusters(conf.num_clusters), num_isect(conf.num_isect),
				max_depth(conf.max_depth), cluster_overlap(conf.cluster_overlap) {}
	std::vector<vec<T>> all_entries;
	struct tree_node {
		std::vector<std::vector<vec<T>>> centres;
		struct VectorHasher {
			size_t operator()(const clustering_location& V) const {
				int concat_val = 0;
				size_t num_vals_possible = 8; // TODO this needs to match up with
																			// num_clusters, should generalize code
				for (size_t i = 0; i < V.size(); ++i) {
					concat_val = concat_val * num_vals_possible + V[i];
				}
				return std::hash<int>{}(concat_val);
				// int hash = V.size();
				// for (auto& i : V) {
				//	hash ^= i + 0x9e3779b9 + (int(hash) << 6) + (int(hash) >> 2);
				// }
				// return hash;
			}
		};
		clustering_location compute_multiindex(const vec<T>& v) const {
			clustering_location ans;
			for (size_t isect_index = 0; isect_index < centres.size();
					 ++isect_index) {
				size_t best_cluster = 0;
				T best_dist = dist2(v, centres[isect_index][0]);
				for (size_t cluster_index = 0;
						 cluster_index < centres[isect_index].size(); ++cluster_index) {
					T cur_dist = dist2(v, centres[isect_index][cluster_index]);
					if (best_dist > cur_dist) {
						best_cluster = cluster_index;
						best_dist = cur_dist;
					}
				}
				ans.emplace_back(best_cluster);
			}
			return ans;
		}
		robin_hood::unordered_flat_map<clustering_location, std::vector<size_t>,
																	 VectorHasher>
				tables;
		robin_hood::unordered_flat_map<clustering_location, size_t, VectorHasher>
				subtree_tables;
	};
	struct tree {
		std::vector<tree_node> nodes;
	};
	std::vector<tree> trees;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "ISect Clustering Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, tree_copies);
		add_param(pl, max_leaf_size);
		add_param(pl, search_count_per_copy);
		add_param(pl, num_clusters);
		add_param(pl, num_isect);
		add_param(pl, max_depth);
		add_param(pl, cluster_overlap);
		return pl;
	}
};

template <typename T>
void isect_clustering_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void isect_clustering_engine<T>::_build() {
	assert(all_entries.size() > 0);
	std::random_device rd;
	// std::shared_ptr<std::mt19937> gen = std::make_shared<std::mt19937>(rd());
	std::mt19937 gen(rd());
	for (size_t copy = 0; copy < tree_copies; ++copy) {
		std::cerr << "Built " << double(copy) / double(tree_copies) * 100 << "%"
							<< std::endl;
		trees.emplace_back();
		auto& t = trees.back();
		t.nodes.emplace_back();
		std::vector<size_t> all_indices;
		for (size_t i = 0; i < all_entries.size(); ++i)
			all_indices.push_back(i);
		struct buildable {
			size_t node_i;
			const std::vector<size_t> ivals;
			size_t depth;
			buildable(size_t _node_i, const std::vector<size_t> _ivals, size_t _depth)
					: node_i(_node_i), ivals(_ivals), depth(_depth) {}
		};
		std::stack<buildable> to_build;
		to_build.emplace(t.nodes.size() - 1, all_indices, 0);
		while (!to_build.empty()) {
			auto b = to_build.top();
			to_build.pop();

			std::vector<vec<T>> entries;
			for (auto& i : b.ivals)
				entries.push_back(all_entries[i]);

			size_t local_cluster_count = std::min(num_clusters, entries.size());
			size_t local_cluster_overlap =
					std::min(cluster_overlap, local_cluster_count);

			// entry_index -> isect_index -> rank -> cluster_index
			std::vector<std::vector<std::vector<size_t>>> multimultiindex(
					entries.size(),
					std::vector<std::vector<size_t>>(
							num_isect, std::vector<size_t>(local_cluster_overlap, 0)));
			std::vector<std::vector<vec<T>>> all_centres;
			for (size_t isect_index = 0; isect_index < num_isect; ++isect_index) {
				std::vector<vec<T>> centres;
				for (size_t i = 0; i < local_cluster_count; ++i) {
					std::uniform_int_distribution<> distribution(i, entries.size());
					size_t k = distribution(gen);
					std::swap(entries[i], entries[k]);
					centres.emplace_back(entries[i]);
				}
				all_centres.emplace_back(centres);
				for (size_t entry_index = 0; entry_index < entries.size();
						 ++entry_index) {
					std::vector<std::pair<T, size_t>> ranked_clusters;
					for (size_t cluster_index = 0; cluster_index < centres.size();
							 ++cluster_index) {
						ranked_clusters.emplace_back(
								dist2(entries[entry_index], centres[cluster_index]),
								cluster_index);
					}
					std::sort(ranked_clusters.begin(), ranked_clusters.end());
					for (size_t cluster_index = 0; cluster_index < local_cluster_overlap;
							 ++cluster_index) {
						multimultiindex[entry_index][isect_index][cluster_index] =
								ranked_clusters[cluster_index].second;
					}
				}
				// TODO
				// cluster centres are entries[0,...,local_cluster_count-1]
				// for everything in entries:
				//   vector<pair<T, i>> ranked_clusters;
				//   for cluster_i in cluster_indices:
				//     ranked_clusters.emplace_back(distance to cluster cluster_i,
				//     cluster_i)
				//   sort(ranked_clusters)
				//   // multiindex[current entry][isect_index] =
				//   ranked_clusters[0].second for cluster_i from 0 to
				//   cluster_overlap-1: 	 multimultiindex[current
				//   entry][isect_index][cluster_i] = ranked_clusters[cluster_i].second
			}
			// entry_index -> list of isect_index lists
			std::vector<std::vector<clustering_location>> multiindexlist(
					entries.size());
			// computing this explicitly is technically unnecessary (we could fill the
			// tables here) recursively generate multiindexlist from multimultiindex
			for (size_t entry_index = 0; entry_index < entries.size();
					 ++entry_index) {
				std::function<void(clustering_location, size_t)> recurser =
						[&](clustering_location partial_list, size_t isect_index) {
							if (isect_index >= num_isect) {
								multiindexlist[entry_index].emplace_back(partial_list);
							} else {
								for (size_t cluster_i :
										 multimultiindex[entry_index][isect_index]) {
									clustering_location next_partial_list = partial_list;
									next_partial_list.emplace_back(cluster_i);
									recurser(next_partial_list, isect_index + 1);
								}
							}
						};
				recurser(clustering_location(), 0);
			}

			t.nodes[b.node_i].centres = all_centres;
			std::vector<clustering_location> to_be_children;
			for (size_t entry_index = 0; entry_index < entries.size();
					 ++entry_index) {
				size_t i = b.ivals[entry_index];
				for (const auto& mi : multiindexlist[entry_index]) {
					auto& table = t.nodes[b.node_i].tables[mi];
					table.push_back(i);
					if (table.size() == max_leaf_size + 1)
						to_be_children.push_back(mi); // triggers at most once per table
				}
			}
			if (b.depth < max_depth)
				for (auto mi : to_be_children) {
					t.nodes.emplace_back();
					t.nodes[b.node_i].subtree_tables[mi] = t.nodes.size() - 1;
					// if num_probes were to be used (instead of overlapping clusters),
					// would do it right here
					to_build.emplace(t.nodes.size() - 1, t.nodes[b.node_i].tables[mi],
													 b.depth + 1);
				}

			/*
			arragement_generator<T> arrange_gen(all_entries[0].size(), num_clusters,
																					num_isect, gen);
			vecset vs(entries);
			t.nodes[b.node_i].arrange = arrange_gen(vs);
			std::vector<clustering_location> to_be_children;
			for (auto& i : b.ivals) {
				auto mi = t.nodes[b.node_i].arrange.compute_multiindex(all_entries[i]);
				auto& table = t.nodes[b.node_i].tables[mi];
				table.push_back(i);
				if (table.size() == max_leaf_size + 1)
					to_be_children.push_back(mi); // triggers at most once per table
			}
			if (b.depth < max_depth)
				for (auto mi : to_be_children) {
					t.nodes.emplace_back();
					t.nodes[b.node_i].subtree_tables[mi] = t.nodes.size() - 1;
					// get cluster_overlap close cells and add them to the set to be
					// propogated
					std::vector<clustering_location> close_cells =
							t.nodes[b.node_i].arrange.random_probes(mi, cluster_overlap,
																											*gen);
					std::vector<size_t> close_cells_contents;
					for (auto& mi_close : close_cells)
						for (auto& vi : t.nodes[b.node_i].tables[mi_close])
							close_cells_contents.emplace_back(vi);
					to_build.emplace(t.nodes.size() - 1, close_cells_contents,
													 b.depth + 1);
				}
			*/
		}
	}
}
template <typename T>
std::vector<size_t> isect_clustering_engine<T>::_query_k(const vec<T>& v,
																												 size_t k) {
	topk_t<T> tk(k);
	for (auto& t : trees) {
		size_t visited = 0;
		std::vector<std::reference_wrapper<std::vector<size_t>>> tables_to_check;
		// populate all nodes to check
		{
			size_t cur = 0;
			auto mi = t.nodes[cur].compute_multiindex(v);
			tables_to_check.push_back(t.nodes[cur].tables[mi]);
			while (t.nodes[cur].subtree_tables.count(mi) > 0) {
				if (t.nodes[cur].subtree_tables.count(mi) == 0)
					break;
				cur = t.nodes[cur].subtree_tables[mi];
				mi = t.nodes[cur].compute_multiindex(v);
				tables_to_check.push_back(t.nodes[cur].tables[mi]);
			}
			reverse(tables_to_check.begin(), tables_to_check.end());
			for (auto& table : tables_to_check) {
				for (size_t ui : table.get()) {
					auto& u = all_entries[ui];
					tk.consider(dist(v, u), ui);
					// if (dist2(v, u) < dist2(v, all_entries[ret])) {
					//	ret = ui;
					// }
				}
				visited += table.get().size();
				if (visited >= search_count_per_copy)
					break;
			}
		}
	}
	return tk.to_vector();
	// return {ret};
}
