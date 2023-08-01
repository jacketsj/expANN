#pragma once

#include <map>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "randomgeometry.h"
#include "vecset.h"

// basic lsh method
template <typename T>
struct arrangement_engine : public ann_engine<T, arrangement_engine<T>> {
	size_t affine_copies, num_orientations;
	arrangement_engine(size_t _affine_copies, size_t _num_orientations)
			: affine_copies(_affine_copies), num_orientations(_num_orientations) {}
	std::vector<vec<T>> all_entries;
	arrangement<T> arrange;
	std::map<std::vector<unsigned short>, std::vector<vec<T>>> table;
	void _store_vector(const vec<T>& v);
	void _build();
	const vec<T>& _query(const vec<T>& v);
	const std::string _name() { return "Arrangement Engine"; }
	const std::string _name_long() {
		return "Arrangement Engine (num_orientations=" +
					 std::to_string(num_orientations) +
					 ",affine_copies=" + std::to_string(affine_copies) + ")";
	}
};

template <typename T>
void arrangement_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void arrangement_engine<T>::_build() {
	assert(all_entries.size() > 0);
	arragement_generator<T> arrange_gen(all_entries[0].size(), affine_copies,
																			num_orientations);
	vecset vs(all_entries);
	arrange = arrange_gen(vs);
	for (const auto& v : all_entries)
		table[arrange.compute_multiindex(v)].push_back(v);
}
template <typename T>
const vec<T>& arrangement_engine<T>::_query(const vec<T>& v) {
	vec<T>& ret = all_entries[0];
	for (const auto& e : table[arrange.compute_multiindex(v)]) {
		if (dist2(v, e) < dist2(v, ret)) {
			ret = e;
		}
	}
	return ret;
}
