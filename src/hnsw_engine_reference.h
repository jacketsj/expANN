#pragma once

#include "ann_engine.h"

#include "../hnswlib/hnswlib/hnswlib.h"

struct hnsw_engine_reference_config {
	int M;
	int ef_construction;
	hnsw_engine_reference_config(int _M, int _ef_construction)
			: M(_M), ef_construction(_ef_construction) {}
};

template <typename T>
struct hnsw_engine_reference : public ann_engine<T, hnsw_engine_reference<T>> {
	int M;
	int ef_construction;
	hnswlib::HierarchicalNSW<T>* alg_hnsw;
	hnsw_engine_reference(hnsw_engine_reference_config conf)
			: M(conf.M), ef_construction(conf.ef_construction), alg_hnsw(nullptr) {}
	std::vector<vec<T>> all_entries;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(vec<T> v, size_t k);
	const std::string _name() { return "HNSW Reference Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, ef_construction);
		return pl;
	}
};

template <typename T>
void hnsw_engine_reference<T>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T> void hnsw_engine_reference<T>::_build() {
	assert(all_entries.size() > 0);

	// init index
	hnswlib::L2Space space(all_entries[0].size());
	alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, all_entries.size(), M,
																								 ef_construction);
	// Add data to index
	int i = 0;
	for (auto& v : all_entries) {
		alg_hnsw->addPoint(v.data(), i++);
	}
}

template <typename T>
std::vector<size_t> hnsw_engine_reference<T>::_query_k(vec<T> v, size_t k) {
	std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
			alg_hnsw->searchKnn(v.data(), k);

	std::vector<size_t> ret;
	while (!result.empty()) {
		ret.push_back(result.top().second);
		result.pop();
	}
	return ret;
}
