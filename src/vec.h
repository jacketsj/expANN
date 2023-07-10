#pragma once

#include <cmath>
#include <vector>

template <typename T> class vec {
	std::vector<T> internal;

public:
	vec() = default;
	vec(size_t dim) : internal(dim) {}
	size_t dim() const { return internal.size(); }
	void operator+=(const vec<T>& oth) {
		assert(dim() == oth.dim());
		for (size_t i = 0; i < internal.size(); ++i)
			internal[i] += oth.internal[i];
	}
	void operator-=(const vec<T>& oth) {
		assert(dim() == oth.dim());
		for (size_t i = 0; i < internal.size(); ++i)
			internal[i] -= oth.internal[i];
	}
	void operator*=(const T& val) {
		for (size_t i = 0; i < internal.size(); ++i)
			internal[i] *= val;
	}
	void operator/=(const T& val) {
		for (size_t i = 0; i < internal.size(); ++i)
			internal[i] /= val;
	}
	vec<T> operator+(const vec<T>& oth) const {
		vec<T> ans(*this);
		ans += oth;
		return ans;
	}
	vec<T> operator-(const vec<T>& oth) const {
		vec<T> ans(*this);
		ans -= oth;
		return ans;
	}
	T dot(const vec<T>& oth) const {
		assert(dim() == oth.dim());
		T ans = 0;
		for (size_t i = 0; i < internal.size(); ++i)
			ans += internal[i] * oth.internal[i];
	}
	T norm2() const {
		T ans = 0;
		for (T& val : internal)
			ans += val * val;
		return ans;
	}
	T norm() const { return sqrt(norm2()); }
	void normalize() { (*this) /= norm(); }
	friend T dist2(const vec<T>& a, const vec<T>& b) { return (a - b).norm2(); }
	friend T dist(const vec<T>& a, const vec<T>& b) { return sqrt(dist2(a, b)); }
	friend T dot(const vec<T>& a, const vec<T>& b) { return a.dot(b); }
	friend T cosinedist2(const vec<T>& a, const vec<T>& b) {
		T dotres = dot(a, b);
		return dotres * dotres / (a.norm2() * b.norm2());
	}
	friend T cosinedist(const vec<T>& a, const vec<T>& b) {
		return sqrt(cosinedist2(a, b));
	}
};
