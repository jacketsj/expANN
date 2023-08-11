#pragma once

#include <queue>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "topk_t.h"

template <typename T>
struct brute_force_engine : public ann_engine<T, brute_force_engine<T>> {
	std::vector<vec<T>> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	const vec<T>& _query(const vec<T>& v);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Brute-Force Engine"; }
	const param_list_t _param_list() { return param_list_t(); }
};

template <typename T>
void brute_force_engine<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void brute_force_engine<T>::_build() {
	assert(all_entries.size() > 0);
}
template <typename T>
std::vector<size_t> brute_force_engine<T>::_query_k(const vec<T>& v, size_t k) {
	topk_t<T> tk(k);
	for (size_t i = 0; i < all_entries.size(); ++i) {
		tk.consider(dist2(v, all_entries[i]), i);
	}
	return tk.to_vector();
}
