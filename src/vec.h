#pragma once

#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

//#include "distance.h"

#include <Eigen/Dense>

#include "small_vector.hpp"

template <typename> struct ChooseUnderlying;

template <> struct ChooseUnderlying<float> {
#ifdef DIM
	using Underlying = Eigen::Matrix<float, DIM, 1>;
#else
	using Underlying = Eigen::VectorXf;
#endif
};

template <> struct ChooseUnderlying<double> {
	using Underlying = Eigen::VectorXd;
};

// template <typename T> struct ChooseUnderlying {
//	using Underlying = gch::small_vector<T>;
// };

template <typename T> class vec {
	// std::vector<T> internal;
	// gch::small_vector<T> internal;

	// Eigen::VectorXf internal;
	using Underlying = typename ChooseUnderlying<T>::Underlying;
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
	vec(const Underlying& _internal) : internal(_internal) {}
	vec(Underlying&& _internal) : internal(_internal) {}
	T* data() { return internal.data(); }
	void set_dim(size_t dim) { internal.resize(dim); }
	size_t size() const { return size_t(internal.size()); }
	size_t dim() const { return size(); }
	std::string to_string() const {
		std::string ret;
		for (const auto& val : internal)
			ret += std::to_string(val) + ' ';
		return ret;
	}
	const Underlying& get_underlying() const { return internal; }
	T& operator[](size_t i) { return internal[i]; }
	void operator+=(const vec<T>& oth) {
		internal += oth.internal;
		// assert(dim() == oth.dim());
		// for (size_t i = 0; i < size(); ++i)
		//	internal[i] += oth.internal[i];
	}
	void operator-=(const vec<T>& oth) {
		internal -= oth.internal;
		// assert(dim() == oth.dim());
		// for (size_t i = 0; i < size(); ++i)
		//	internal[i] -= oth.internal[i];
	}
	void operator*=(const T& val) {
		internal *= val;
		// for (size_t i = 0; i < size(); ++i)
		//	internal[i] *= val;
	}
	void operator/=(const T& val) {
		internal /= val;
		// for (size_t i = 0; i < size(); ++i)
		//	internal[i] /= val;
	}
	vec<T> operator+(const vec<T>& oth) const {
		return vec<T>(internal + oth.internal);
		// assert(dim() == oth.dim());
		// vec<T> ans(*this);
		// ans += oth;
		// return ans;
	}
	vec<T> operator-(const vec<T>& oth) const {
		return vec<T>(internal - oth.internal);
		// assert(dim() == oth.dim());
		// vec<T> ans(*this);
		// ans -= oth;
		// return ans;
	}
	T dot(const vec<T>& oth) const {
		return internal.dot(oth.internal);

		// assert(dim() == oth.dim());
		// T ans = 0;
		// for (size_t i = 0; i < size(); ++i)
		//	ans += internal[i] * oth.internal[i];
		// return ans;
	}
	T norm2() const {
		return internal.squaredNorm();
		// T ans = 0;
		// for (const T& val : internal)
		//	ans += val * val;
		// return ans;
	}
	T norm() const {
		return internal.norm();
		// return sqrt(norm2());
	}
	vec<T> normalized() const {
		return vec<T>(internal.normalized());
		// vec<T> ret = (*this);
		// ret.normalize();
		// return ret;
	}
	void normalize() {
		(*this) = normalized();
		// (*this) /= norm();
	}
	friend T dist2(const vec<T>& a, const vec<T>& b) {
		return (a.internal - b.internal).squaredNorm();
		// return (a - b).norm2();
	}
	friend T dist2fast(const vec<T>& a, const vec<T>& b) {
		return (a.internal - b.internal).squaredNorm();
		// return distance_compare_avx512f_f16(a.internal.data(), b.internal.data(),
		//																		a.size());
		//  return distance_compare_avx512f_f16((unsigned char*)a.internal.data(),
		//																		(unsigned char*)b.internal.data(),
		//																		a.size());
	}
	friend T dist(const vec<T>& a, const vec<T>& b) {
		// assert(a.dim() == b.dim());
		return (a.internal - b.internal).norm();
		// return sqrt(dist2(a, b));
	}
	friend T dot(const vec<T>& a, const vec<T>& b) { return a.dot(b); }
	// cosine dist not currently used anywhere in the codebase, commented for now
	// friend T cosinedist2(const vec<T>& a, const vec<T>& b) {
	// 	T dotres = dot(a, b);
	// 	return dotres * dotres / (a.norm2() * b.norm2());
	// }
	// friend T cosinedist(const vec<T>& a, const vec<T>& b) {
	// 	return sqrt(cosinedist2(a, b));
	// }
	std::vector<T> to_vector() const {
		std::vector<T> ret(dim());
		for (size_t i = 0; i < dim(); ++i) {
			ret[i] = internal[i];
		}
		return ret;
	}
	friend void to_json(nlohmann::json& j, const vec<T>& v) {
		j = nlohmann::json(v.to_vector());
	}
	friend void from_json(const nlohmann::json& j, vec<T>& v) {
		// v.internal = j.get<std::vector<T>>();
		auto full_vector = j.get<std::vector<T>>();
		v = vec<T>(full_vector);
	}
};
