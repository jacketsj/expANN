#pragma once

#include "vec.h"

/// TODO is this even a good representation of a hyperplane? Could use a vector
/// + a scalar instead
template <typename T> class hyperplane {
	vec<T> normvec;
	vec<T> origin;

public:
	hyperplane() = default;
	hyperplane(const vec<T>& _normvec)
			: normvec(_normvec), origin(_normvec.size()) {}
	hyperplane(const vec<T>& _normvec, const vec<T>& _origin)
			: normvec(_normvec), origin(_origin) {}

	// T signeddist2(const vec<T>& test) const {
	//	return dot(normvec, test - origin);
	// }
	// T signeddist(const vec<T>& test) const {
	//	T d2 = signeddist2
	//	return sqrt(signeddist2);
	// }
	//
	//  bool side(const vec<T>& test) const { return signeddist2(test) > 0; }
};
