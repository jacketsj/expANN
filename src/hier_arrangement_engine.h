#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "randomgeometry.h"
#include "vecset.h"

struct hier_arrangement_engine_config {
	size_t affine_copies, num_arranges, num_levels, level_mult,
			starting_orientations, search_count;
	hier_arrangement_engine_config(size_t _num_arranges, size_t _num_levels,
																 size_t _search_count,
																 size_t _affine_copies = 3,
																 size_t _level_mult = 2,
																 size_t _starting_orientations = 1)
			: affine_copies(_affine_copies), num_arranges(_num_arranges),
				num_levels(_num_levels), level_mult(_level_mult),
				starting_orientations(_starting_orientations),
				search_count(_search_count) {}
};

// basic lsh method
template <typename T>
struct hier_arrangement_engine
		: public ann_engine<T, hier_arrangement_engine<T>> {
	size_t affine_copies, num_arranges, num_levels, level_mult,
			starting_orientations, search_count;
	hier_arrangement_engine(hier_arrangement_engine_config conf)
			: affine_copies(conf.affine_copies), num_arranges(conf.num_arranges),
				num_levels(conf.num_levels), level_mult(conf.level_mult),
				starting_orientations(conf.starting_orientations),
				search_count(conf.search_count) {}
	std::vector<vec<T>> all_entries;
	struct arrangement_level {
		std::vector<arrangement<T>> arranges;
		std::vector<std::map<arrangement_location, std::vector<size_t>>> tables;
	};
	std::vector<arrangement_level> levels;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Hierarchical Arrangement Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, num_arranges);
		add_param(pl, num_levels);
		add_param(pl, search_count);
		add_param(pl, affine_copies);
		add_param(pl, level_mult);
		add_param(pl, starting_orientations);
		return pl;
	}
};

template <typename T>
void hier_arrangement_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void hier_arrangement_engine<T>::_build() {
	assert(all_entries.size() > 0);
	std::random_device rd;
	std::shared_ptr<std::mt19937> gen = std::make_shared<std::mt19937>(rd());
	size_t num_orientations = starting_orientations;
	for (size_t level = 0; level < num_levels; ++level) {
		levels.emplace_back();
		for (size_t arrange_no = 0; arrange_no < num_arranges; ++arrange_no) {
			arragement_generator<T> arrange_gen(all_entries[0].size(), affine_copies,
																					num_orientations, gen);
			vecset vs(all_entries);
			levels.back().arranges.push_back(arrange_gen(vs));
			levels.back().tables.emplace_back();
			for (size_t vi = 0; vi < all_entries.size(); ++vi)
				levels.back()
						.tables
						.back()[levels.back().arranges.back().compute_multiindex(
								all_entries[vi])]
						.push_back(vi);
		}
		starting_orientations *= level_mult;
	}
}
template <typename T>
std::vector<size_t> hier_arrangement_engine<T>::_query_k(const vec<T>& v,
																												 size_t k) {
	std::cout << "Using query (MISSING _query_k impl)" << std::endl; // TODO
	size_t ret = 0;
	size_t visited = 0;
	for (int level = num_levels - 1; level >= 0; --level) {
		for (size_t a = 0; a < num_arranges; ++a) {
			auto arrange =
					levels[level]
							.tables[a][levels[level].arranges[a].compute_multiindex(v)];
			for (const auto& ei : arrange) {
				if (dist2(v, all_entries[ei]) < dist2(v, all_entries[ret])) {
					ret = ei;
				}
			}
			visited += arrange.size();
			// TODO figure out if I should put this further in? How to avoid simd
			// problems though...
			if (visited >= search_count)
				break;
		}
	}
	return {ret};
}
