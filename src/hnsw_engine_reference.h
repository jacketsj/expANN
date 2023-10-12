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
	hnswlib::L2Space* space;
	hnsw_engine_reference(hnsw_engine_reference_config conf)
			: M(conf.M), ef_construction(conf.ef_construction), alg_hnsw(nullptr),
				space(nullptr) {}
	// std::vector<vec<T>> all_entries;
	size_t dim;
	std::vector<T> all_entries_f;
	void _store_vector(vec<T> v);
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

template <typename T> void hnsw_engine_reference<T>::_store_vector(vec<T> v) {
	// all_entries.push_back(v);
	dim = v.size();
	for (size_t i = 0; i < v.size(); ++i)
		all_entries_f.push_back(v[i]);
}

template <typename T> void hnsw_engine_reference<T>::_build() {
	assert(all_entries_f.size() > 0);

	// init index
	space = new hnswlib::L2Space(dim);
	alg_hnsw = new hnswlib::HierarchicalNSW<float>(
			space, all_entries_f.size() / dim, M, ef_construction);
	// Add data to index
	int j = 0;
	for (size_t i = 0; i < all_entries_f.size(); i += dim) {
		if (j % 5000 == 0)
			std::cerr << "Built "
								<< double(j) / double(all_entries_f.size() / dim) * 100 << "%"
								<< std::endl;
		alg_hnsw->addPoint(&all_entries_f[i], j++);
	}
	// for (auto& v : all_entries) {
	//}
}

template <typename T>
std::vector<size_t> hnsw_engine_reference<T>::_query_k(vec<T> v, size_t k) {
	size_t ref_size = all_entries_f.size();
	for (size_t i = 0; i < v.size(); ++i)
		all_entries_f.push_back(v[i]);
	std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
			alg_hnsw->searchKnn(&all_entries_f[ref_size], k);
	// alg_hnsw->searchKnn(v.data(), k);

	std::vector<size_t> ret;
	while (!result.empty()) {
		ret.push_back(result.top().second);
		result.pop();
	}
	return ret;
}
