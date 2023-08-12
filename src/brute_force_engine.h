#pragma once

#include <queue>
#include <string>
#include <vector>

#include "ann_engine.h"

template <typename T>
struct brute_force_engine : public ann_engine<T, brute_force_engine<T>> {
	std::vector<vec<T>> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
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
	std::priority_queue<std::pair<T, size_t>> top_k;
	for (size_t i = 0; i < all_entries.size(); ++i) {
		T d = dist2(v, all_entries[i]);
		if (top_k.size() < k || top_k.top().first > d) {
			top_k.emplace(d, i);
		}
		if (top_k.size() > k)
			top_k.pop();
	}
	std::vector<size_t> ret;
	while (!top_k.empty()) {
		ret.push_back(top_k.top().second);
		top_k.pop();
	}
	reverse(ret.begin(), ret.end()); // sort from closest to furthest
	return ret;
}
