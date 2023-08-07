#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "randomgeometry.h"
#include "vecset.h"

// idea: create a tree of arrangements. Each cell with too many points becomes a
// child.
// should have multiple copies of the tree (with different seeds) to avoid edge
// problems
// this is similar to using random quadtrees. Turns out it's also quite similar
// to the known-optimal data-dependent ANN algorithm (but a lot more practical)
//
// this is the version using an inverted file for each node. Should result in
// higher build time for better query time/cache hits
template <typename T>
struct tree_arrangement_engine_if
		: public ann_engine<T, tree_arrangement_engine_if<T>> {
	size_t tree_copies, max_leaf_size, search_count_per_copy, affine_copies,
			num_orientations, max_depth;
	tree_arrangement_engine_if(size_t _tree_copies, size_t _max_leaf_size,
														 size_t _search_count_per_copy,
														 size_t _affine_copies = 3,
														 size_t _num_orientations = 8,
														 size_t _max_depth = 40)
			: tree_copies(_tree_copies), max_leaf_size(_max_leaf_size),
				search_count_per_copy(_search_count_per_copy),
				affine_copies(_affine_copies), num_orientations(_num_orientations),
				max_depth(_max_depth) {}
	std::vector<vec<T>> all_entries;
	struct tree_node {
		arrangement<T> arrange;
		struct VectorHasher {
			size_t operator()(const std::vector<unsigned short>& V) const {
				int concat_val = 0;
				size_t num_vals_possible = 4; // TODO this needs to match up with
																			// affine_copies, should generalize code
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
		std::unordered_map<std::vector<unsigned short>,
											 std::vector<std::pair<vec<T>, size_t>>, VectorHasher>
				tables;
		std::unordered_map<std::vector<unsigned short>, size_t, VectorHasher>
				subtree_tables;
	};
	struct tree {
		std::vector<tree_node> nodes;
	};
	std::vector<tree> trees;
	void _store_vector(const vec<T>& v);
	void _build();
	const vec<T>& _query(const vec<T>& v);
	const std::string _name() {
		return "Tree Arrangement Engine (with simple IF)";
	}
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, tree_copies);
		add_param(pl, max_leaf_size);
		add_param(pl, search_count_per_copy);
		add_param(pl, affine_copies);
		add_param(pl, num_orientations);
		add_param(pl, max_depth);
		return pl;
	}
};

template <typename T>
void tree_arrangement_engine_if<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void tree_arrangement_engine_if<T>::_build() {
	assert(all_entries.size() > 0);
	std::random_device rd;
	std::shared_ptr<std::mt19937> gen = std::make_shared<std::mt19937>(rd());
	for (size_t copy = 0; copy < tree_copies; ++copy) {
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

			arragement_generator<T> arrange_gen(all_entries[0].size(), affine_copies,
																					num_orientations, gen);
			std::vector<vec<T>> entries;
			for (auto& i : b.ivals)
				entries.push_back(all_entries[i]);
			vecset vs(entries);
			t.nodes[b.node_i].arrange = arrange_gen(vs);
			std::vector<std::vector<unsigned short>> to_be_children;
			for (auto& i : b.ivals) {
				auto mi = t.nodes[b.node_i].arrange.compute_multiindex(all_entries[i]);
				auto& table = t.nodes[b.node_i].tables[mi];
				table.emplace_back(all_entries[i], i);
				if (table.size() == max_leaf_size + 1)
					to_be_children.push_back(mi); // triggers at most once per table
			}
			if (b.depth < max_depth)
				for (auto mi : to_be_children) {
					t.nodes.emplace_back();
					t.nodes[b.node_i].subtree_tables[mi] = t.nodes.size() - 1;
					std::vector<size_t> ivals_only;
					for (auto& [_, i] : t.nodes[b.node_i].tables[mi])
						ivals_only.push_back(i);
					to_build.emplace(t.nodes.size() - 1, ivals_only, b.depth + 1);
				}
		}
	}
}
template <typename T>
const vec<T>& tree_arrangement_engine_if<T>::_query(const vec<T>& v) {
	vec<T>& ret = all_entries[0];
	for (auto& t : trees) {
		size_t visited = 0;
		std::vector<std::reference_wrapper<std::vector<std::pair<vec<T>, size_t>>>>
				tables_to_check;
		// populate all nodes to check
		{
			size_t cur = 0;
			auto mi = t.nodes[cur].arrange.compute_multiindex(v);
			tables_to_check.push_back(t.nodes[cur].tables[mi]);
			while (t.nodes[cur].subtree_tables.count(mi) > 0) {
				if (t.nodes[cur].subtree_tables.count(mi) == 0)
					break;
				cur = t.nodes[cur].subtree_tables[mi];
				mi = t.nodes[cur].arrange.compute_multiindex(v);
				tables_to_check.push_back(t.nodes[cur].tables[mi]);
			}
			reverse(tables_to_check.begin(), tables_to_check.end());
			for (auto& table : tables_to_check) {
				size_t iters = std::min(table.get().size(), search_count_per_copy);
				for (size_t table_ind = 0; table_ind < iters; ++table_ind) {
					// for (auto& [u, _] : table.get()) {
					auto& u = table.get()[table_ind].first;
					if (dist2(v, u) < dist2(v, ret)) {
						ret = u; // all_entries[iu];
					}
				}
				visited += table.get().size();
				if (visited >= search_count_per_copy)
					break;
			}
		}
	}
	return ret;
}
