#pragma once

#include <algorithm>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "randomgeometry.h"
#include "robin_hood.h"
#include "topk_t.h"

#include "ehnsw_engine_basic_fast.h"

struct ehnsw_engine_basic_fast_clusterchunks_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	bool use_cuts;
	size_t min_cluster_size;
	size_t max_cluster_size;
	ehnsw_engine_basic_fast_clusterchunks_config(
			size_t _M, size_t _M0, size_t _ef_search_mult, size_t _ef_construction,
			bool _use_cuts, size_t _min_cluster_size, size_t _max_cluster_size)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction), use_cuts(_use_cuts),
				min_cluster_size(_min_cluster_size),
				max_cluster_size(_max_cluster_size) {}
};

template <typename T>
struct ehnsw_engine_basic_fast_clusterchunks
		: public ann_engine<T, ehnsw_engine_basic_fast_clusterchunks<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	bool use_cuts;
	size_t min_cluster_size;
	size_t max_cluster_size;
	size_t max_layer;
#ifdef RECORD_STATS
	size_t num_distcomps;
	size_t total_projected_degree;
	size_t total_clusters_checked;
#endif
	ehnsw_engine_basic_fast_clusterchunks(
			ehnsw_engine_basic_fast_clusterchunks_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction), use_cuts(conf.use_cuts),
				min_cluster_size(conf.min_cluster_size),
				max_cluster_size(conf.max_cluster_size), max_layer(0) {}
	using config = ehnsw_engine_basic_fast_clusterchunks_config;
	std::vector<vec<T>> all_entries;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<std::vector<std::vector<std::pair<T, size_t>>>>
			hadj_flat_with_lengths; // vector -> layer -> edges with lengths
	std::vector<vec<T>> centroids;
	std::vector<std::vector<size_t>> clusters;
	std::vector<size_t> reverse_clusters;
	std::vector<std::vector<size_t>> hadj_bottom_projected;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<char> visited; // booleans
	std::vector<size_t> visited_recent;
	std::vector<std::vector<bool>> e_labels; // vertex -> cut labels (*num_cuts)
	size_t num_cuts() { return use_cuts ? e_labels[0].size() : 0; }
	std::vector<std::pair<T, size_t>>
	prune_edges(size_t layer, size_t from, std::vector<std::pair<T, size_t>> to);
	template <bool use_bottomlayer, bool use_clusters>
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const vec<T>& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k);
	std::vector<std::pair<T, size_t>>
	query_k_at_bottom_via_clusters(const vec<T>& q, size_t layer,
																 const std::vector<size_t>& entry_points,
																 size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() {
		return use_cuts ? "EHNSW Engine Basic Fast Cluster-Chunks"
										: "HNSW Engine Basic Fast Cluster-Chunks";
	}
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		add_param(pl, use_cuts);
		add_param(pl, min_cluster_size);
		add_param(pl, max_cluster_size);
#ifdef RECORD_STATS
		add_param(pl, num_distcomps);
		add_param(pl, total_projected_degree);
		add_param(pl, total_clusters_checked);
#endif
		return pl;
	}
	bool generate_elabel() {
		std::uniform_int_distribution<> int_distribution(0, 1);
		return int_distribution(gen);
	}
};

template <typename T>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_clusterchunks<T>::prune_edges(
		size_t layer, size_t from, std::vector<std::pair<T, size_t>> to) {
	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

	// reference impl vs paper difference
	if (to.size() <= edge_count_mult) {
		return to;
	}

	sort(to.begin(), to.end());
	std::vector<std::pair<T, size_t>> ret;
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
void ehnsw_engine_basic_fast_clusterchunks<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.push_back(v);

	e_labels.emplace_back();
	for (size_t cut = 0; cut < M0 - 2 * 10; ++cut)
		e_labels.back().emplace_back(generate_elabel());

	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));
	// size_t new_max_layer = 0;

	hadj_flat_with_lengths.emplace_back();
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat_with_lengths[v_index].emplace_back();
	}

	auto convert_el = [](std::vector<std::pair<T, size_t>> el) constexpr {
		std::vector<size_t> ret;
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
					query_k_at_layer<false, false>(v, layer, cur, ef_construction));
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

template <typename T> void ehnsw_engine_basic_fast_clusterchunks<T>::_build() {
	assert(all_entries.size() > 0);
#ifdef RECORD_STATS
	total_projected_degree = 0;
	total_clusters_checked = 0;
#endif

	// start by setting centroids to the values of the second-to-last layer
	for (size_t i = 0; i < all_entries.size(); ++i) {
		if (hadj_flat[i].size() > 1) {
			centroids.emplace_back(all_entries[i]);
		}
	}
	auto assign_to_clusters = [&]() {
		ehnsw_engine_basic_fast<T> sub_engine(ehnsw_engine_basic_fast_config(
				M, M0, ef_search_mult, ef_construction, use_cuts));
		for (auto& v : centroids)
			sub_engine.store_vector(v);
		sub_engine.build();
		clusters.clear();
		clusters.resize(centroids.size());
		for (size_t i = 0; i < all_entries.size(); ++i) {
			clusters[sub_engine.query_k(all_entries[i], 1)[0]].emplace_back(i);
		}
	};
	const size_t max_iters = 30;
	for (size_t iter = 0; iter < max_iters; ++iter) {
		assign_to_clusters();
		// loosely enforce min/max cluster sizes
		std::vector<std::vector<size_t>> clusters_new;
		vec_generator<T> rvgen(all_entries[0].size());
		for (size_t cluster_index = 0; cluster_index < clusters.size();
				 ++cluster_index) {
			// enforce min cluster size by deleting small clusters
			if (clusters[cluster_index].size() >= min_cluster_size) {
				// enforce max cluster size by splitting big clusters
				if (clusters[cluster_index].size() <= max_cluster_size) {
					clusters_new.emplace_back(clusters[cluster_index]);
				} else {
					// partition with a random hyperplane
					auto project_vec = rvgen.random_vec(); // normal vector of hyperplane
					clusters_new.emplace_back();
					clusters_new.emplace_back();
					for (size_t entry_index : clusters[cluster_index]) {
						clusters_new[clusters_new.size() - 1 -
												 ((all_entries[entry_index] - centroids[cluster_index])
															.dot(project_vec) > 0)]
								.emplace_back(entry_index);
					}
				}
			}
		}
		clusters = clusters_new;
		for (size_t centroid_index = 0; centroid_index < centroids.size();
				 ++centroid_index) {
			auto& centroid = centroids[centroid_index];
			centroid.clear();
			for (size_t elem_index : clusters[centroid_index]) {
				centroid += all_entries[elem_index];
			}
			if (!clusters[centroid_index].empty()) {
				centroid /= clusters[centroid_index].size();
			}
		}
	}
	assign_to_clusters();
	reverse_clusters.resize(all_entries.size());
	for (size_t i = 0; i < clusters.size(); ++i) {
		for (size_t j : clusters[i])
			reverse_clusters[j] = i;
	}

	hadj_bottom_projected.resize(all_entries.size());
	for (size_t i = 0; i < all_entries.size(); ++i) {
		robin_hood::unordered_flat_set<size_t> added;
		for (size_t j : hadj_bottom[i]) {
			size_t cluster_index = reverse_clusters[j];
			if (!added.contains(cluster_index)) {
				hadj_bottom_projected[i].emplace_back(cluster_index);
				added.insert(cluster_index);
			}
		}
#ifdef RECORD_STATS
		total_projected_degree += hadj_bottom_projected[i].size();
#endif
	}

#ifdef RECORD_STATS
	// reset before queries
	num_distcomps = 0;
#endif
}

template <typename T>
template <bool use_bottomlayer, bool use_clusters>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_clusterchunks<T>::query_k_at_layer(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;

	auto get_vertex = [&](const size_t& index) constexpr->std::vector<size_t>& {
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
#ifdef RECORD_STATS
		++num_distcomps;
#endif
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
		std::vector<size_t> neighbour_list;
		if constexpr (use_clusters && use_bottomlayer) {
			for (size_t cluster_neighbour : hadj_bottom_projected[cur.second]) {
				for (size_t neighbour : clusters[cluster_neighbour]) {
					if (!visited[neighbour]) {
						neighbour_list.emplace_back(neighbour);
						visited[neighbour] = true;
						visited_recent.emplace_back(neighbour);
					}
				}
			}
		} else
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
#ifdef RECORD_STATS
			++num_distcomps;
#endif
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
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_clusterchunks<T>::query_k_at_bottom_via_clusters(
		const vec<T>& q, size_t layer,
		const std::vector<size_t>& initial_entry_points, size_t k) {
	using measured_data = std::pair<T, size_t>;

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : initial_entry_points) {
#ifdef RECORD_STATS
		++num_distcomps;
#endif
		size_t cluster_index = reverse_clusters[entry_point];
		visited[cluster_index] = true;
		visited_recent.emplace_back(cluster_index);
		for (size_t entry_index : clusters[cluster_index]) {
			T d_entry_index = dist2(q, all_entries[entry_index]);
			entry_points_with_dist.emplace_back(d_entry_index, entry_index);
		}
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

	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		auto loop_with_prefetches =
				[&](const std::vector<size_t>& member_list,
						const std::vector<vec<T>>& vec_list,
						std::function<void(const size_t&)> inner_loop_fn) {
					constexpr size_t in_advance = 4;
					constexpr size_t in_advance_extra = 2;
					auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
						for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
							_mm_prefetch(((char*)&vec_list[member_list[i]]) + mult * 64,
													 _MM_HINT_T0);
#endif
					};
					for (size_t next_i_pre = 0;
							 next_i_pre < std::min(in_advance, member_list.size());
							 ++next_i_pre) {
						do_loop_prefetch(next_i_pre);
					}
					auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
							size_t next_i) constexpr {
						if constexpr (inAdvanceIterExtra) {
							_mm_prefetch(&member_list[next_i + in_advance + in_advance_extra],
													 _MM_HINT_T0);
						}
						if constexpr (inAdvanceIter) {
							do_loop_prefetch(next_i + in_advance);
						}
						const auto& next = member_list[next_i];
						inner_loop_fn(next);
					};
					size_t next_i = 0;
					for (; next_i + in_advance + in_advance_extra < member_list.size();
							 ++next_i) {
						loop_iter.template operator()<true, true>(next_i);
					}
					for (; next_i + in_advance < member_list.size(); ++next_i) {
						loop_iter.template operator()<true, false>(next_i);
					}
					for (; next_i < member_list.size(); ++next_i) {
						loop_iter.template operator()<false, false>(next_i);
					}
				};
		bool found_continuation = nearest.size() < k;
		for (size_t cluster_index : hadj_bottom_projected[cur.second]) {
			if (!visited[cluster_index]) {
				visited[cluster_index] = true;
				visited_recent.emplace_back(cluster_index);
#ifdef RECORD_STATS
				++total_clusters_checked;
#endif
				loop_with_prefetches(
						clusters[cluster_index], all_entries, [&](const size_t& next) {
#ifdef RECORD_STATS
							++num_distcomps;
#endif
							T d_next = dist2(q, all_entries[next]);
							if (nearest.size() < k || d_next < nearest.top().first) {
								nearest.emplace(d_next, next);
								if (nearest.size() > k)
									nearest.pop();
								found_continuation = true;
								candidates.emplace(d_next, next);
							}
						});
			}
		}
		if (!found_continuation) {
			break;
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
ehnsw_engine_basic_fast_clusterchunks<T>::_query_k(const vec<T>& q, size_t k) {
	size_t entry_point = starting_vertex;
#ifdef RECORD_STATS
	++num_distcomps;
#endif
	T ep_dist = dist2(all_entries[entry_point], q);
	for (size_t layer = max_layer - 1; layer > 0; --layer) {
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto& neighbour : hadj_flat[entry_point][layer]) {
				_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
#ifdef RECORD_STATS
				++num_distcomps;
#endif
				T neighbour_dist = dist2(q, all_entries[neighbour]);
				if (neighbour_dist < ep_dist) {
					entry_point = neighbour;
					ep_dist = neighbour_dist;
					changed = true;
				}
			}
		}
	}

	// auto ret_combined =
	//		query_k_at_layer<true, true>(q, 0, {entry_point}, k * ef_search_mult);
	auto ret_combined =
			query_k_at_bottom_via_clusters(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}
