#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "ehnsw_engine_basic_fast.h"
#include "robin_hood.h"
#include "topk_t.h"

struct multi_engine_partition_config {
	ehnsw_engine_basic_fast_config subconf;
	size_t num_splits;
	size_t oversample_factor;
	multi_engine_partition_config(ehnsw_engine_basic_fast_config _subconf,
																size_t _num_splits, size_t _oversample_factor)
			: subconf(_subconf), num_splits(_num_splits),
				oversample_factor(_oversample_factor) {}
};

template <typename T>
struct multi_engine_partition
		: public ann_engine<T, multi_engine_partition<T>> {
	std::random_device rd;
	std::mt19937 gen;
	ehnsw_engine_basic_fast_config subconf;
	size_t num_splits;
	size_t total_inserted;
	size_t oversample_factor;
	std::vector<std::unique_ptr<ehnsw_engine_basic_fast<T>>> subengines;
	std::vector<std::vector<size_t>> to_original_index;
	multi_engine_partition(multi_engine_partition_config conf)
			: rd(), gen(0), subconf(conf.subconf), num_splits(conf.num_splits),
				total_inserted(0), oversample_factor(conf.oversample_factor),
				subengines() {
		for (size_t e = 0; e < num_splits; ++e) {
			subengines.emplace_back(
					std::make_unique<ehnsw_engine_basic_fast<T>>(subconf));
			to_original_index.emplace_back();
		}
	}
	using config = multi_engine_partition_config;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Multi Engine Partition"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, subconf.M);
		add_param(pl, subconf.M0);
		add_param(pl, subconf.ef_search_mult);
		add_param(pl, subconf.ef_construction);
		add_param(pl, subconf.use_cuts);
		add_param(pl, num_splits);
		add_param(pl, oversample_factor);
		return pl;
	}
};

template <typename T>
void multi_engine_partition<T>::_store_vector(const vec<T>& v) {
	std::uniform_int_distribution<size_t> distribution(0, num_splits - 1);
	size_t split = distribution(gen);
	to_original_index[split].emplace_back(total_inserted++);
	subengines[split]->store_vector(v);
}

template <typename T> void multi_engine_partition<T>::_build() {
	for (size_t e = 0; e < num_splits; ++e) {
		subengines[e]->build();
	}
}

template <typename T>
std::vector<size_t> multi_engine_partition<T>::_query_k(const vec<T>& q,
																												size_t k) {
	std::vector<std::pair<T, size_t>> ret_combined;
	size_t k_per = std::max((oversample_factor * k + num_splits - 1) / num_splits,
													size_t(1));
	for (size_t e = 0; e < std::min(num_splits, total_inserted); ++e) {
		for (const auto& [dist, local_index] :
				 subengines[e]->query_k_combined(q, k_per)) {
			size_t global_index = to_original_index[e][local_index];
			ret_combined.emplace_back(dist, global_index);
		}
	}
	std::sort(ret_combined.begin(), ret_combined.end());
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (const auto& [_, i] : ret_combined)
		ret.emplace_back(i);
	return ret;
}
