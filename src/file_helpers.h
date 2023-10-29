#pragma once

#include "file_allocator.h"

#define MAX_LAYERS 12
#define MAX_NEIGHBOURS 200

template <typename T> using filevec = std::vector<T, file_allocator<T>>;
template <typename T, unsigned N> using sv = gch::small_vector<T, N>;
template <typename T> using svl = sv<T, MAX_LAYERS>;
template <typename T> using svn = sv<T, MAX_NEIGHBOURS>;
template <typename T> filevec<T> tofv(const filevec<T>& v) { return v; }
template <typename T> svn<T> tosvn(const svn<T>& v) { return v; }
template <typename T> svn<T> tosvn(const std::vector<T>& v) {
	svn<T> result;
	result.reserve(v.size());
	for (const auto& item : v) {
		result.push_back(item);
	}
	return result;
}
template <typename T> filevec<T> tofv(const std::vector<T>& v) {
	filevec<T> result;
	result.reserve(v.size());
	for (const auto& item : v) {
		result.push_back(item);
	}
	return result;
}
