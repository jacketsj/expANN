#pragma once

#include <algorithm>
#include <execution>
#include <map>
#include <string>
#include <vector>

#include "vec.h"

typedef std::map<std::string, std::string> param_list_t;
#define add_param(pl, p) pl[#p] = std::to_string(p)
#define add_sub_param(pl, prefix_name, pname, p)                               \
	pl[std::string(prefix_name) + pname] = p

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
	const std::vector<size_t> query_k(const vec<T>& v, size_t k) {
		return static_cast<Derived*>(this)->_query_k(v, k);
	}
	const std::vector<std::vector<size_t>>
	query_k_batch(const std::vector<vec<T>>& vs, size_t k) {
		std::vector<std::vector<size_t>> ret;
		for (const auto& v : vs) {
			ret.emplace_back(query_k(v, k));
		}
		// std::transform(std::execution::par_unseq, vs.begin(), vs.end(),
		// ret.begin(),
		//							 [&](const vec<T>& v) { return query_k(v, k); });
		return ret;
	}
};
