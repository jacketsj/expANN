#pragma once

#include <string>

#include "vec.h"

template <typename T, class Derived> struct ann_engine {
	ann_engine() = default;
	std::string name() { return static_cast<Derived*>(this)->_name(); }
	void store_vector(const vec<T>& v) {
		static_cast<Derived*>(this)->_store_vector(v);
	}
	void build() { static_cast<Derived*>(this)->_build(); }
	const vec<T>& query(const vec<T>& v) {
		return static_cast<Derived*>(this)->_query(v);
	}
};
