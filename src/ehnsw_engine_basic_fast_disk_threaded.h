#pragma once

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "ann_engine.h"
#include "file_helpers.h"
#include "robin_hood.h"
#include "topk_t.h"

#include "small_vector.hpp"

#include "file_allocator.h"

//#include "mmappable_vector.h"
// using namespace mmap_allocator_namespace;

struct ehnsw_engine_basic_fast_disk_threaded_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	size_t num_worker_threads;
	size_t chunk_size;
	ehnsw_engine_basic_fast_disk_threaded_config(size_t _M, size_t _M0,
																							 size_t _ef_search_mult,
																							 size_t _ef_construction,
																							 size_t _num_worker_threads,
																							 size_t _chunk_size)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction),
				num_worker_threads(_num_worker_threads), chunk_size(_chunk_size) {}
	static ehnsw_engine_basic_fast_disk_threaded_config default_conf() {
		return ehnsw_engine_basic_fast_disk_threaded_config(40, 80, 2, 100, 8, 100);
	}
};

template <typename T>
struct ehnsw_engine_basic_fast_disk_threaded
		: public ann_engine<T, ehnsw_engine_basic_fast_disk_threaded<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	size_t num_worker_threads;
	size_t chunk_size;
	size_t max_layer;
	ehnsw_engine_basic_fast_disk_threaded()
			: rd(), gen(0), distribution(0, 1),
				M(ehnsw_engine_basic_fast_disk_threaded_config::default_conf().M),
				M0(ehnsw_engine_basic_fast_disk_threaded_config::default_conf().M0),
				ef_search_mult(
						ehnsw_engine_basic_fast_disk_threaded_config::default_conf()
								.ef_search_mult),
				ef_construction(
						ehnsw_engine_basic_fast_disk_threaded_config::default_conf()
								.ef_construction),
				num_worker_threads(
						ehnsw_engine_basic_fast_disk_threaded_config::default_conf()
								.num_worker_threads),
				chunk_size(ehnsw_engine_basic_fast_disk_threaded_config::default_conf()
											 .chunk_size),
				max_layer(0) {}
	ehnsw_engine_basic_fast_disk_threaded(
			ehnsw_engine_basic_fast_disk_threaded_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction),
				num_worker_threads(conf.num_worker_threads),
				chunk_size(conf.chunk_size), max_layer(0) {}
	filevec<vec<T>> all_entries;
	filevec<svl<svn<size_t>>> hadj_flat; // vector -> layer -> edges
	filevec<svn<size_t>> hadj_bottom;		 // vector -> edges in bottom layer
	filevec<svl<svn<std::pair<T, size_t>>>>
			hadj_flat_with_lengths; // vector -> layer -> edges with lengths
	void _store_vector(const vec<T>& v);
	void _build();
	filevec<char> visited; // booleans
	std::vector<size_t> visited_recent;
	std::vector<std::vector<char>> e_labels; // vertex -> cut labels (*num_cuts)
	size_t num_cuts() { return e_labels[0].size(); }
	template <typename container>
	svn<std::pair<T, size_t>> prune_edges(size_t layer, size_t from,
																				container to);
	template <bool use_bottomlayer>
	std::vector<std::pair<T, size_t>>
	query_k_at_layer_threaded(const vec<T>& q, size_t layer,
														const std::vector<size_t>& entry_points, size_t k);
	template <bool use_bottomlayer>
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const vec<T>& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	std::vector<std::pair<T, size_t>> query_k_combined(const vec<T>& v, size_t k);
	const std::string _name() { return "EHNSW Engine Basic Fast Disk Threaded"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		add_param(pl, num_worker_threads);
		add_param(pl, chunk_size);
		return pl;
	}
	bool generate_elabel() {
		std::uniform_int_distribution<> int_distribution(0, 1);
		return int_distribution(gen);
	}
};

template <typename T>
template <typename container>
svn<std::pair<T, size_t>>
ehnsw_engine_basic_fast_disk_threaded<T>::prune_edges(size_t layer, size_t from,
																											container to) {
	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

	// reference impl vs paper difference
	if (to.size() <= edge_count_mult) {
		svn<std::pair<T, size_t>> ret = tosvn(to);
		return ret;
		// return to;
	}

	sort(to.begin(), to.end());
	svn<std::pair<T, size_t>> ret;
	std::vector<bool> bins(layer > 0 ? 0 : edge_count_mult - num_cuts());
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
		if (layer == 0 && ret.size() + num_cuts() >= edge_count_mult) {
			bool found_bin = false;
			for (size_t bin = 0; bin < bins.size(); ++bin) {
				if (!bins[bin] && e_labels[md.second][bin] != e_labels[from][bin]) {
					bins[bin] = true;
					found_bin = true;
				}
			}
			if (!found_bin) {
				choose = false;
			}
		}
		if (choose) {
			ret.emplace_back(md);
		}
	}

	return ret;
}

template <typename T>
void ehnsw_engine_basic_fast_disk_threaded<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.push_back(v);

	e_labels.emplace_back();
	for (size_t cut = 0; cut < M0 - 2 * 10; ++cut)
		e_labels.back().emplace_back(generate_elabel());

	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));

	hadj_flat_with_lengths.emplace_back();
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat_with_lengths[v_index].emplace_back();
	}

	auto convert_el = [](svn<std::pair<T, size_t>> el) constexpr {
		svn<size_t> ret;
		ret.reserve(el.size());
		for (auto& [_, val] : el) {
			ret.emplace_back(val);
		}
		return ret;
	};

	// get kNN for each layer
	std::vector<std::vector<std::pair<T, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		{
			size_t entry_point = starting_vertex;
			T ep_dist = dist2(v, all_entries[entry_point]);
			for (size_t layer = max_layer - 1; layer > new_max_layer; --layer) {
				bool changed = true;
				while (changed) {
					changed = false;
					for (auto& neighbour : hadj_flat[entry_point][layer]) {
						_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
						T neighbour_dist = dist2(v, all_entries[neighbour]);
						if (neighbour_dist < ep_dist) {
							entry_point = neighbour;
							ep_dist = neighbour_dist;
							changed = true;
						}
					}
				}
			}
			cur = {entry_point};
		}
		for (int layer = std::min(new_max_layer, max_layer - 1); layer >= 0;
				 --layer) {
			kNN_per_layer.emplace_back(
					query_k_at_layer<false>(v, layer, cur, ef_construction));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
			cur.resize(1); // present in reference impl, but not in hnsw paper
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}

	// add the found edges to the graph
	for (size_t layer = 0; layer < std::min(max_layer, new_max_layer + 1);
			 ++layer) {
		hadj_flat_with_lengths[v_index][layer] =
				prune_edges(layer, v_index, kNN_per_layer[layer]);
		//  add bidirectional connections, prune if necessary
		for (auto& md : kNN_per_layer[layer]) {
			bool edge_exists = false;
			for (auto& md_other : hadj_flat_with_lengths[md.second][layer]) {
				if (md_other.second == v_index) {
					edge_exists = true;
				}
			}
			if (!edge_exists) {
				hadj_flat_with_lengths[md.second][layer].emplace_back(md.first,
																															v_index);
				hadj_flat_with_lengths[md.second][layer] = prune_edges(
						layer, md.second, hadj_flat_with_lengths[md.second][layer]);
				hadj_flat[md.second][layer] =
						convert_el(hadj_flat_with_lengths[md.second][layer]);
				if (layer == 0)
					hadj_bottom[md.second] = hadj_flat[md.second][layer];
			}
		}
	}

	// add new layers if necessary
	while (new_max_layer >= max_layer) {
		++max_layer;
		starting_vertex = v_index;
	}

	visited.emplace_back();
	hadj_flat.emplace_back();
	hadj_bottom.emplace_back();
	hadj_bottom[v_index] = convert_el(hadj_flat_with_lengths[v_index][0]);
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat[v_index].emplace_back();
		hadj_flat[v_index][layer] =
				convert_el(hadj_flat_with_lengths[v_index][layer]);
	}
}

template <typename T> void ehnsw_engine_basic_fast_disk_threaded<T>::_build() {
	assert(all_entries.size() > 0);
}

template <typename T>
template <bool use_bottomlayer>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_disk_threaded<T>::query_k_at_layer_threaded(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;

	auto get_vertex = [&](const size_t& index) constexpr
												->decltype(hadj_bottom[0])& {
		if constexpr (use_bottomlayer) {
			return hadj_bottom[index];
		} else {
			return hadj_flat[index][layer];
		}
	};

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points) {
		entry_points_with_dist.emplace_back(dist2(q, all_entries[entry_point]),
																				entry_point);
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

	for (auto& entry_point : entry_points) {
		visited[entry_point] = true;
		visited_recent.emplace_back(entry_point);
	}

	tbb::global_control gcontrol(tbb::global_control::max_allowed_parallelism,
															 num_worker_threads);

	while (!candidates.empty()) {
		std::vector<measured_data> next_candidates;
		while (!candidates.empty() && next_candidates.size() < chunk_size) {
			// while (!candidates.empty() && next_candidates.size() < 1) {
			auto cur = candidates.top();
			candidates.pop();
			if (next_candidates.empty() && cur.first > nearest.top().first &&
					nearest.size() == k) {
				break;
			}
			for (size_t neighbour : get_vertex(cur.second))
				if (!visited[neighbour]) {
					// prefetch from disk
					for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
						_mm_prefetch(((char*)&all_entries[neighbour]) + mult * 64,
												 _MM_HINT_T0);
					next_candidates.emplace_back(0, neighbour);
					visited[neighbour] = true;
					visited_recent.emplace_back(neighbour);
				}
		}
		// for (auto& [d_next, next] : next_candidates) {
		//	d_next = dist2(q, all_entries[next]);
		//}
		tbb::parallel_for(
				tbb::blocked_range<size_t>(0, next_candidates.size()),
				[&](const tbb::blocked_range<size_t>& range) {
					for (size_t i = range.begin(); i != range.end(); ++i) {
						auto& [d_next, next] = next_candidates[i];
						d_next = dist2(q, all_entries[next]);
					}
				},
				tbb::simple_partitioner());
		for (auto& [d_next, next] : next_candidates) {
			if (nearest.size() < k || d_next < nearest.top().first) {
				candidates.emplace(d_next, next);
				nearest.emplace(d_next, next);
				if (nearest.size() > k)
					nearest.pop();
			}
		}
	}
	/*
	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		std::vector<size_t> neighbour_list; // = get_vertex(cur.second);
		for (size_t neighbour : get_vertex(cur.second))
			if (!visited[neighbour]) {
				neighbour_list.emplace_back(neighbour);
				visited[neighbour] = true;
				visited_recent.emplace_back(neighbour);
			}
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
			for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
				_mm_prefetch(((char*)&all_entries[neighbour_list[i]]) + mult * 64,
										 _MM_HINT_T0);
#endif
			_mm_prefetch(&visited[neighbour_list[i]], _MM_HINT_T0);
		};
		for (size_t next_i_pre = 0;
				 next_i_pre < std::min(in_advance, neighbour_list.size());
				 ++next_i_pre) {
			do_loop_prefetch(next_i_pre);
		}
		auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
				size_t next_i) constexpr {
			if constexpr (inAdvanceIterExtra) {
				_mm_prefetch(&neighbour_list[next_i + in_advance + in_advance_extra],
										 _MM_HINT_T0);
			}
			if constexpr (inAdvanceIter) {
				do_loop_prefetch(next_i + in_advance);
			}
			const auto& next = neighbour_list[next_i];
			// if (!visited[next]) {
			// visited[next] = true;
			// visited_recent.emplace_back(next);
			T d_next = dist2(q, all_entries[next]);
			if (nearest.size() < k || d_next < nearest.top().first) {
				candidates.emplace(d_next, next);
				nearest.emplace(d_next, next);
				if (nearest.size() > k)
					nearest.pop();
			}
			//}
		};
		size_t next_i = 0;
		for (; next_i + in_advance + in_advance_extra < neighbour_list.size();
				 ++next_i) {
			loop_iter.template operator()<true, true>(next_i);
		}
		for (; next_i + in_advance < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<true, false>(next_i);
		}
		for (; next_i < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<false, false>(next_i);
		}
	}
	*/

	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
template <bool use_bottomlayer>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_disk_threaded<T>::query_k_at_layer(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	return query_k_at_layer_threaded<use_bottomlayer>(q, layer, entry_points, k);

	using measured_data = std::pair<T, size_t>;

	auto get_vertex = [&](const size_t& index) constexpr
												->decltype(hadj_bottom[0])& {
		if constexpr (use_bottomlayer) {
			return hadj_bottom[index];
		} else {
			return hadj_flat[index][layer];
		}
	};

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points) {
		entry_points_with_dist.emplace_back(dist2(q, all_entries[entry_point]),
																				entry_point);
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

	for (auto& entry_point : entry_points) {
		visited[entry_point] = true;
		visited_recent.emplace_back(entry_point);
	}

	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		std::vector<size_t> neighbour_list; // = get_vertex(cur.second);
		for (size_t neighbour : get_vertex(cur.second))
			if (!visited[neighbour]) {
				neighbour_list.emplace_back(neighbour);
				visited[neighbour] = true;
				visited_recent.emplace_back(neighbour);
			}
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
			for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
				_mm_prefetch(((char*)&all_entries[neighbour_list[i]]) + mult * 64,
										 _MM_HINT_T0);
#endif
			_mm_prefetch(&visited[neighbour_list[i]], _MM_HINT_T0);
		};
		for (size_t next_i_pre = 0;
				 next_i_pre < std::min(in_advance, neighbour_list.size());
				 ++next_i_pre) {
			do_loop_prefetch(next_i_pre);
		}
		auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
				size_t next_i) constexpr {
			if constexpr (inAdvanceIterExtra) {
				_mm_prefetch(&neighbour_list[next_i + in_advance + in_advance_extra],
										 _MM_HINT_T0);
			}
			if constexpr (inAdvanceIter) {
				do_loop_prefetch(next_i + in_advance);
			}
			const auto& next = neighbour_list[next_i];
			// if (!visited[next]) {
			// visited[next] = true;
			// visited_recent.emplace_back(next);
			T d_next = dist2(q, all_entries[next]);
			if (nearest.size() < k || d_next < nearest.top().first) {
				candidates.emplace(d_next, next);
				nearest.emplace(d_next, next);
				if (nearest.size() > k)
					nearest.pop();
			}
			//}
		};
		size_t next_i = 0;
		for (; next_i + in_advance + in_advance_extra < neighbour_list.size();
				 ++next_i) {
			loop_iter.template operator()<true, true>(next_i);
		}
		for (; next_i + in_advance < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<true, false>(next_i);
		}
		for (; next_i < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<false, false>(next_i);
		}
	}
	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t>
ehnsw_engine_basic_fast_disk_threaded<T>::_query_k(const vec<T>& q, size_t k) {
	size_t entry_point = starting_vertex;
	T ep_dist = dist2(all_entries[entry_point], q);
	for (size_t layer = max_layer - 1; layer > 0; --layer) {
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

	auto ret_combined =
			query_k_at_layer<true>(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}

template <typename T>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_disk_threaded<T>::query_k_combined(const vec<T>& q,
																													 size_t k) {
	size_t entry_point = starting_vertex;
	T ep_dist = dist2(all_entries[entry_point], q);
	for (int layer = max_layer - 1; layer >= 0; --layer) {
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

	auto ret_combined =
			query_k_at_layer<true>(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	return ret_combined;
}