#pragma once

#include "ann_engine.h"

#include "../hnswlib/hnswlib/hnswlib.h"

struct hnsw_engine_reference_config {
	int M;
	int ef_construction;
	bool use_ecuts;
	hnsw_engine_reference_config(int _M, int _ef_construction,
															 bool _use_ecuts = false)
			: M(_M), ef_construction(_ef_construction), use_ecuts(_use_ecuts) {}
};

template <typename T>
struct hnsw_engine_reference : public ann_engine<T, hnsw_engine_reference<T>> {
	int M;
	int ef_construction;
	bool use_ecuts;
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_int_distribution<> int_distribution;
	std::set<size_t>
			available_bins; // bins available (stateful value, for current checks)
	hnswlib::HierarchicalNSW<T>* alg_hnsw;	 // TODO make this a unique ptr
	hnswlib::L2Space* space;								 // TODO make this a unique ptr
	std::vector<std::vector<char>> e_labels; // vertex -> cut labels (*num_cuts)
	hnsw_engine_reference(hnsw_engine_reference_config conf)
			: M(conf.M), ef_construction(conf.ef_construction),
				use_ecuts(conf.use_ecuts), rd(), gen(rd()), int_distribution(0, 1),
				alg_hnsw(nullptr), space(nullptr) {}
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
		add_param(pl, use_ecuts);
		return pl;
	}
	bool generate_elabel() { return int_distribution(gen); }
	bool edge_filter(size_t v, size_t u);
	void edge_filter_refresh(size_t M) {
		available_bins.clear();
		for (size_t i = 0; i + 1 < M; ++i) {
			available_bins.insert(i);
		}
	}
};

template <typename T> void hnsw_engine_reference<T>::_store_vector(vec<T> v) {
	// all_entries.push_back(v);
	dim = v.size();
	for (size_t i = 0; i < v.size(); ++i)
		all_entries_f.push_back(v[i]);

	e_labels.emplace_back();
	for (size_t cut = 0; cut + 1 < M; ++cut)
		e_labels.back().emplace_back(char(bool(generate_elabel())));
}

template <typename T>
bool hnsw_engine_reference<T>::edge_filter(size_t v, size_t u) {
	for (size_t bin : available_bins) {
		// an edge is permitted in a bin if it crosses the cut for that bin
		if (bin >= e_labels[v].size() || bin >= e_labels[u].size() ||
				e_labels[v][bin] != e_labels[u][bin]) {
			available_bins.erase(bin);
			return true;
		}
	}
	return false;
}

template <typename T> void hnsw_engine_reference<T>::_build() {
	assert(all_entries_f.size() > 0);

	// init index
	space = new hnswlib::L2Space(dim);
	// TODO do this in a memory safe way
	std::function<void(size_t)> efr = [&](size_t M) { edge_filter_refresh(M); };
	std::function<bool(size_t, size_t)> ef = [&](size_t u, size_t v) {
		if (!use_ecuts)
			return true;
		return edge_filter(u, v);
	};
	// if (use_ecuts) {
	alg_hnsw = new hnswlib::HierarchicalNSW<float>(
			space, all_entries_f.size() / dim, M, ef_construction, ef, efr);
	//} else {
	// TODO fix this
	// alg_hnsw = new hnswlib::HierarchicalNSW<float>(
	//		space, all_entries_f.size() / dim, M, ef_construction);
	//}
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
