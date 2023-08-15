#pragma once

#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "small_vector.hpp"

template <typename> struct ChooseUnderlying;

template <> struct ChooseUnderlying<float> {
	using Underlying = Eigen::VectorXf;
};

template <> struct ChooseUnderlying<double> {
	using Underlying = Eigen::VectorXd;
};

template <typename T> struct ChooseUnderlying {
	using Underlying = gch::small_vector<T>;
};

template <typename T> class vec {
	// std::vector<T> internal;
	// gch::small_vector<T> internal;

	// Eigen::VectorXf internal;
	using Underlying = ChooseUnderlying<T>::Underlying;
	Underlying internal;

public:
	vec() = default;
	// vec(size_t dim) : internal(dim) {}
	vec(const std::vector<T>& v) {
		internal.resize(v.size());
		for (size_t i = 0; i < size_t(v.size()); ++i)
			internal[i] = v[i];
		// for (const auto& val : v)
		//	internal.emplace_back(val);
	}
	void set_dim(size_t dim) { internal.resize(dim); }
	size_t size() const { return size_t(internal.size()); }
	size_t dim() const { return size(); }
	std::string to_string() const {
		std::string ret;
		for (const auto& val : internal)
			ret += std::to_string(val) + ' ';
		return ret;
	}
	T& operator[](size_t i) { return internal[i]; }
	void operator+=(const vec<T>& oth) {
		// assert(dim() == oth.dim());
		for (size_t i = 0; i < size(); ++i)
			internal[i] += oth.internal[i];
	}
	void operator-=(const vec<T>& oth) {
		// assert(dim() == oth.dim());
		for (size_t i = 0; i < size(); ++i)
			internal[i] -= oth.internal[i];
	}
	void operator*=(const T& val) {
		for (size_t i = 0; i < size(); ++i)
			internal[i] *= val;
	}
	void operator/=(const T& val) {
		for (size_t i = 0; i < size(); ++i)
			internal[i] /= val;
	}
	vec<T> operator+(const vec<T>& oth) const {
		vec<T> ans(*this);
		ans += oth;
		return ans;
	}
	vec<T> operator-(const vec<T>& oth) const {
		// assert(dim() == oth.dim());
		vec<T> ans(*this);
		ans -= oth;
		return ans;
	}
	T dot(const vec<T>& oth) const {
		// assert(dim() == oth.dim());
		T ans = 0;
		for (size_t i = 0; i < size(); ++i)
			ans += internal[i] * oth.internal[i];
		return ans;
	}
	T norm2() const {
		T ans = 0;
		for (const T& val : internal)
			ans += val * val;
		return ans;
	}
	T norm() const { return sqrt(norm2()); }
	void normalize() { (*this) /= norm(); }
	vec<T> normalized() const {
		vec<T> ret = (*this);
		ret.normalize();
		return ret;
	}
	friend T dist2(const vec<T>& a, const vec<T>& b) { return (a - b).norm2(); }
	friend T dist(const vec<T>& a, const vec<T>& b) {
		// assert(a.dim() == b.dim());
		return sqrt(dist2(a, b));
	}
	friend T dot(const vec<T>& a, const vec<T>& b) { return a.dot(b); }
	friend T cosinedist2(const vec<T>& a, const vec<T>& b) {
		T dotres = dot(a, b);
		return dotres * dotres / (a.norm2() * b.norm2());
	}
	friend T cosinedist(const vec<T>& a, const vec<T>& b) {
		return sqrt(cosinedist2(a, b));
	}
	friend void to_json(nlohmann::json& j, const vec<T>& v) {
		j = nlohmann::json(v.internal);
	}
	friend void from_json(const nlohmann::json& j, vec<T>& v) {
		// v.internal = j.get<std::vector<T>>();
		auto full_vector = j.get<std::vector<T>>();
		v = vec<T>(full_vector);
	}
};
