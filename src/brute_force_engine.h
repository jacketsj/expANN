#pragma once

#include <string>
#include <vector>

#include "ann_engine.h"

template <typename T>
struct brute_force_engine : public ann_engine<T, brute_force_engine<T>> {
	std::vector<vec<T>> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	const vec<T>& _query(const vec<T>& v);
	const std::string _name() { return "Brute-Force Engine"; }
	const std::string _name_long() { return "Brute-Force Engine"; }
};

template <typename T>
void brute_force_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void brute_force_engine<T>::_build() {
	assert(all_entries.size() > 0);
}
template <typename T>
const vec<T>& brute_force_engine<T>::_query(const vec<T>& v) {
	vec<T>& ret = all_entries[0];
	for (const auto& e : all_entries) {
		if (dist2(v, e) < dist2(v, ret)) {
			ret = e;
		}
	}
	return ret;
}
