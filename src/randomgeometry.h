#pragma once

#include "vec.h"

template <typename T> vec<T> random_vec() {
	// TODO
}

template <typename T> vec<T> random_vec() { return hyperplane(random_vec()); }
