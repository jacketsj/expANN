#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "randomgeometry.h"
#include "topk_t.h"
#include "vecset.h"

// basic lsh method
template <typename T>
struct arrangement_engine : public ann_engine<T, arrangement_engine<T>> {
	size_t affine_copies, num_orientations, num_arranges;
	arrangement_engine(size_t _affine_copies, size_t _num_orientations,
										 size_t _num_arranges)
			: affine_copies(_affine_copies), num_orientations(_num_orientations),
				num_arranges(_num_arranges) {}
	std::vector<vec<T>> all_entries;
	std::vector<arrangement<T>> arranges;
	std::vector<std::map<std::vector<unsigned short>,
											 std::vector<std::pair<vec<T>, size_t>>>>
			tables;
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k(const vec<T>& v);
	const std::string _name() { return "Arrangement Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, num_orientations);
		add_param(pl, affine_copies);
		add_param(pl, num_arranges);
		return pl;
	}
};

template <typename T>
void arrangement_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void arrangement_engine<T>::_build() {
	assert(all_entries.size() > 0);
	std::random_device rd;
	std::shared_ptr<std::mt19937> gen = std::make_shared<std::mt19937>(rd());
	for (size_t arrange_no = 0; arrange_no < num_arranges; ++arrange_no) {
		arragement_generator<T> arrange_gen(all_entries[0].size(), affine_copies,
																				num_orientations, gen);
		vecset vs(all_entries);
		arranges.push_back(arrange_gen(vs));
		tables.emplace_back();
		for (size_t vi = 0; vi < all_entires.size(); ++vi)
			tables.back()[arranges.back().compute_multiindex(all_entries[vi])]
					.emplace_back(all_entries[vi], vi);
	}
}
template <typename T>
const std::vector<size_t> arrangement_engine<T>::_query_k(const vec<T>& v,
																													size_t k) {
	topk_t<T> ret;
	for (size_t a = 0; a < num_arranges; ++a)
		for (const auto& [e, ei] : tables[a][arranges[a].compute_multiindex(v)]) {
			ret.consider(dist2(v, e), ei);
		}
	return ret.to_vector();
}
