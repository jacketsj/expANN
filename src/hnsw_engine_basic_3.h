#pragma once

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"
#include "topk_t.h"

struct hnsw_engine_basic_3_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	hnsw_engine_basic_3_config(size_t _M, size_t _M0, size_t _ef_search_mult,
														 size_t _ef_construction)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction) {}
};

template <typename T>
struct hnsw_engine_basic_3 : public ann_engine<T, hnsw_engine_basic_3<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	hnsw_engine_basic_3(hnsw_engine_basic_3_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction) {}
	std::vector<vec<T>> all_entries;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<
			robin_hood::unordered_flat_map<size_t, std::vector<std::pair<T, size_t>>>>
			hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<bool> visited;
	std::vector<size_t> visited_recent;
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const vec<T>& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k);
	std::vector<std::pair<T, size_t>>
	prune_edges(size_t layer, std::vector<std::pair<T, size_t>> to);
	std::vector<size_t> query_k_alt(const vec<T>& v, size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "HNSW Engine Basic 3"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		return pl;
	}
};

template <typename T>
std::vector<std::pair<T, size_t>>
hnsw_engine_basic_3<T>::prune_edges(size_t layer,
																		std::vector<std::pair<T, size_t>> to) {
	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

	// reference impl vs paper difference
	if (to.size() <= edge_count_mult) {
		return to;
	}

	sort(to.begin(), to.end());
	std::vector<std::pair<T, size_t>> ret;
	for (const auto& md : to) {
		_mm_prefetch(&all_entries[md.second], _MM_HINT_T0);
		if (ret.size() >= edge_count_mult)
			break;
		bool choose = true;
		for (const auto& md_chosen : ret) {
			if (md.first == md_chosen.first ||
					dist2(all_entries[md.second], all_entries[md_chosen.second]) <=
							md.first) {
				choose = false;
				break;
			}
		}
		if (choose) {
			ret.emplace_back(md);
		}
	}

	return ret;
}

template <typename T>
void hnsw_engine_basic_3<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.push_back(v);

	// get kNN for each layer
	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));
	// std::cerr << "v_index=" << v_index << " at layer=" << new_max_layer
	//					<< std::endl;
	std::vector<std::vector<std::pair<T, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		for (int layer = hadj.size() - 1; layer > int(new_max_layer); --layer) {
			kNN_per_layer.emplace_back(query_k_at_layer(v, layer, cur, 1));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
		}
		for (int layer = std::min(new_max_layer, hadj.size() - 1); layer >= 0;
				 --layer) {
			kNN_per_layer.emplace_back(
					query_k_at_layer(v, layer, cur, ef_construction));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
			cur.resize(1); // present in reference impl, but not in hnsw paper
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}

	// add the found edges to the graph
	for (size_t layer = 0; layer < std::min(hadj.size(), new_max_layer + 1);
			 ++layer) {
		hadj[layer][v_index] = prune_edges(layer, kNN_per_layer[layer]);
		// std::cerr << "Just set outgoing edges for " << v_index
		//					<< " at layer=" << layer << ": ";
		// for (auto& [_, u] : hadj[layer][v_index]) {
		//	std::cerr << u << ' ';
		// }
		// std::cerr << '\n';
		//  add bidirectional connections, prune if necessary
		for (auto& md : kNN_per_layer[layer]) {
			bool edge_exists = false;
			for (auto& md_other : hadj[layer][md.second]) {
				if (md_other.second == v_index) {
					edge_exists = true;
				}
			}
			if (!edge_exists) {
				// std::cerr << "(Inside) About to re-set outgoing edges for " <<
				// md.second
				//					<< " at layer=" << layer << " and v_index=" << v_index
				//					<< ": ";
				// for (auto& [_, u] : hadj[layer][md.second]) {
				//	std::cerr << u << ' ';
				// }
				// std::cerr << '\n';
				hadj[layer][md.second].emplace_back(md.first, v_index);
				hadj[layer][md.second] = prune_edges(layer, hadj[layer][md.second]);
				// std::cerr << "(Inside) just re-set outgoing edges for " << md.second
				//					<< " at layer=" << layer << ": ";
				// for (auto& [_, u] : hadj[layer][md.second]) {
				//	std::cerr << u << ' ';
				// }
				// std::cerr << '\n';
			}
		}
	}

	// add new layers if necessary
	while (new_max_layer >= hadj.size()) {
		hadj.emplace_back();
		hadj.back()[v_index] = std::vector<std::pair<T, size_t>>();
		starting_vertex = v_index;
	}
}

template <typename T> void hnsw_engine_basic_3<T>::_build() {
	assert(all_entries.size() > 0);

	auto convert_el = [](std::vector<std::pair<T, size_t>> el) {
		std::vector<size_t> ret;
		for (auto& [_, val] : el) {
			ret.emplace_back(val);
		}
		return ret;
	};

	for (size_t v_index = 0; v_index < all_entries.size(); ++v_index) {
		hadj_flat.emplace_back();
		hadj_bottom.emplace_back();
		hadj_bottom[v_index] = convert_el(hadj[0][v_index]);
		for (size_t layer = 0; layer < hadj.size(); ++layer) {
			hadj_flat[v_index].emplace_back();
			if (hadj[layer].contains(v_index)) {
				hadj_flat[v_index][layer] = convert_el(hadj[layer][v_index]);
			} else {
				break;
			}
		}
	}

	visited.resize(all_entries.size());
	visited_recent.reserve(all_entries.size());

	/*
	for (size_t layer = 0; layer < hadj.size(); ++layer) {
		std::cerr << "layer: " << layer << " (num nodes=" << hadj[layer].size()
							<< ")\n";
		for (size_t v_index = 0; v_index < all_entries.size(); ++v_index) {
			if (hadj[layer].contains(v_index)) {
				std::cerr << v_index << "->[";
				for (auto [_, u] : hadj[layer][v_index])
					std::cerr << u << " ";
				std::cerr << "]\n";
			}
		}
	}
	*/
}

// bool querying = false;

template <typename T>
std::vector<std::pair<T, size_t>> hnsw_engine_basic_3<T>::query_k_at_layer(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;
	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (const auto& ep : entry_points) {
		entry_points_with_dist.emplace_back(dist2(q, all_entries[ep]), ep);
	}
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points_with_dist.begin(), entry_points_with_dist.end(),
								 best_elem);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points_with_dist.begin(), entry_points_with_dist.end(),
							worst_elem);
	while (nearest.size() > k)
		nearest.pop();

	robin_hood::unordered_flat_set<size_t> visited;
	for (const auto& ep : entry_points)
		visited.insert(ep);

	while (!candidates.empty()) {
		auto cur = candidates.top();
		// if (querying)
		//	std::cerr << "Looking at candidate (layer=" << layer << ") " <<
		// cur.second
		//						<< std::endl;
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			// TODO second condition should be unnecessary as written
			break;
		}
		// TODO this might affect things positively or negatively
		for (const auto& next : hadj[layer][cur.second]) {
			_mm_prefetch(&next, _MM_HINT_T0);
		}
		for (const auto& [_, next] : hadj[layer][cur.second]) {
			_mm_prefetch(&all_entries[next], _MM_HINT_T0);
		}
		for (const auto& [_, next] : hadj[layer][cur.second]) {
			//_mm_prefetch(&next, _MM_HINT_T0);
			//_mm_prefetch(&all_entries[next], _MM_HINT_T0);
			if (!visited.contains(next)) {
				//_mm_prefetch(&all_entries[next], _MM_HINT_T0);
				visited.insert(next);
				T d_next = dist2(q, all_entries[next]);
				if (nearest.size() < k || d_next < nearest.top().first) {
					// if (querying)
					//	std::cerr << "Looking at edge to " << next << std::endl;
					candidates.emplace(d_next, next);
					nearest.emplace(d_next, next);
					if (nearest.size() > k)
						nearest.pop();
				}
			}
		}
	}
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> hnsw_engine_basic_3<T>::query_k_alt(const vec<T>& q,
																												size_t k) {
	using measured_data = std::pair<T, size_t>;
	size_t entry_point = starting_vertex;
	T ep_dist = dist2(all_entries[entry_point], q);
	for (size_t layer = hadj.size() - 1; layer > 0; --layer) {
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto& neighbour : hadj_flat[entry_point][layer]) {
				_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
				T neighbour_dist = dist2(q, all_entries[neighbour]);
				if (neighbour_dist < ep_dist) {
					entry_point = neighbour;
					ep_dist = neighbour_dist;
					changed = true;
				}
			}
		}
	}

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	entry_points_with_dist.emplace_back(ep_dist, entry_point);

	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points_with_dist.begin(), entry_points_with_dist.end(),
								 best_elem);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points_with_dist.begin(), entry_points_with_dist.end(),
							worst_elem);
	while (nearest.size() > k)
		nearest.pop();

	// TODO make visited a (global-ish) array as well
	// robin_hood::unordered_flat_set<size_t> visited;
	// visited.insert(entry_point);
	visited[entry_point] = true;

	while (!candidates.empty()) {
		auto cur = candidates.top();
		// if (querying)
		//	std::cerr << "Looking at candidate (layer=" << layer << ") " <<
		// cur.second
		//						<< std::endl;
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		// TODO figure out the exact prefetch pattern in reference impl, use it
		_mm_prefetch(&hadj_bottom[cur.second], _MM_HINT_T0);
		for (const auto& next : hadj_bottom[cur.second]) {
			_mm_prefetch(&next, _MM_HINT_T0);
		}
		for (const auto& next : hadj_bottom[cur.second]) {
			_mm_prefetch(&all_entries[next], _MM_HINT_T0);
		}
		for (const auto& next : hadj_bottom[cur.second]) {
			//_mm_prefetch(&next, _MM_HINT_T0);
			_mm_prefetch(&all_entries[next], _MM_HINT_T0);
			if (!visited[next]) {
				// if (!visited.contains(next)) {
				//_mm_prefetch(&all_entries[next], _MM_HINT_T0);
				visited[next] = true;
				// TODO visited_recent can avoid bounds checks since it is always
				// sufficiently reserved
				visited_recent.emplace_back(next);
				// visited.insert(next);
				T d_next = dist2(q, all_entries[next]);
				if (nearest.size() < k || d_next < nearest.top().first) {
					// if (querying)
					//	std::cerr << "Looking at edge to " << next << std::endl;
					candidates.emplace(d_next, next);
					nearest.emplace(d_next, next);
					if (nearest.size() > k)
						nearest.pop();
				}
			}
		}
	}
	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<size_t> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top().second);
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> hnsw_engine_basic_3<T>::_query_k(const vec<T>& q,
																										 size_t k) {
	auto ret = query_k_alt(q, k * ef_search_mult);
	ret.resize(k);
	return ret;

	/*
	// querying = true;
	size_t cur_vert = starting_vertex;
	int layer;
	for (layer = hadj.size() - 1; layer > 0; --layer)
		cur_vert = query_k_at_layer(q, layer, {cur_vert}, 1)[0].second;
	auto ret_combined = query_k_at_layer(q, 0, {cur_vert}, k * ef_search_mult);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
		// std::cerr << "Returning " << ret.back() << std::endl;
	}
	return ret;
	*/
}
