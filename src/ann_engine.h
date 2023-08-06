#pragma once

#include <map>
#include <string>

#include "vec.h"

typedef std::map<std::string, std::string> param_list_t;
#define add_param(pl, p) pl[#p] = std::to_string(p)

template <typename T, class Derived> struct ann_engine {
	ann_engine() = default;
	std::string name() { return static_cast<Derived*>(this)->_name(); }
	param_list_t param_list() {
		return static_cast<Derived*>(this)->_param_list();
	}
	void store_vector(const vec<T>& v) {
		static_cast<Derived*>(this)->_store_vector(v);
	}
	void build() { static_cast<Derived*>(this)->_build(); }
	// TODO add query_k and make the return values size_t and std::vector<size_t>
	const vec<T>& query(const vec<T>& v) {
		return static_cast<Derived*>(this)->_query(v);
	}
};
