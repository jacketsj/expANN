#pragma once

#include "vec.h"

template <typename T> class hyperplane {
	vec<T> normvec;

public:
	hyperplane() = default;
	hyperplane(const vec<T>& _normvec) : normvec(_normvec.normalize()) {}

	T signeddist(const vec<T>& test) const { return dot(normvec, test); }
	// bool side(const vec<T>& test) const { return signeddist2(test) > 0; }
};
