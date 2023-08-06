#pragma once

#include <vector>

#include "dataset.h"
#include "vec.h"

template <typename T>
struct in_memory_dataset : public dataset<T, in_memory_dataset<T>> {
	std::vector<vec<T>> all_vecs;
	vec<T> _get_vec(size_t i) const { return all_vecs[i]; };
};

template <typename T>
struct in_memory_test_dataset
		: public test_dataset<T, in_memory_test_dataset<T>, in_memory_dataset<T>> {
	std::vector<vec<T>> all_query_vecs;
	std::vector<std::vector<size_t>> all_query_ans;
	vec<T> _get_query(size_t i) const { return all_query_vecs[i]; }
	std::vector<size_t> _get_query_ans(size_t i) const {
		return all_query_ans[i];
	}
};

template <typename T>
void to_json(nlohmann::json& j, const in_memory_test_dataset<T>& imtd) {
	j = nlohmann::json{{"name", imtd.name},
										 {"n", imtd.n},
										 {"dim", imtd.dim},
										 {"m", imtd.m},
										 {"k", imtd.k},
										 {"all_vecs", nlohmann::json(imtd.all_vecs)},
										 {"all_query_vecs", nlohmann::json(imtd.all_vecs)},
										 {"all_query_ans", nlohmann::json(imtd.all_vecs)}};
}

template <typename T>
void from_json(const nlohmann::json& j, in_memory_test_dataset<T>& imtd) {
	imtd.name = j.at("name");
	imtd.n = j.at("n");
	imtd.dim = j.at("dim");
	imtd.m = j.at("m");
	imtd.k = j.at("k");
	imtd.all_vecs = j.at("all_vecs");
	imtd.all_query_vecs = j.at("all_query_vecs");
	imtd.all_query_ans = j.at("all_query_ans");
}
