#include "par_antitopo_engine.h"

namespace {
template <typename A, typename B> auto dist2(const A& a, const B& b) {
	return (a - b).squaredNorm();
}
} // namespace

void par_antitopo_engine::prune_edges(size_t layer, size_t from, bool lazy) {
	auto& to = hadj_flat_with_lengths[from][layer];

	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult *= 2;

	if (lazy && to.size() <= edge_count_mult) {
		add_new_edges(layer, from);
		return;
	}

	std::sort(to.begin(), to.end());
	std::vector<std::pair<dist_t, size_t>> ret;
	// std::vector<fvec> normalized_ret;
	auto origin = all_entries[from];
	for (const auto& md : to) {
		bool choose = true;
		// auto v1 = (all_entries[md.second] - origin).normalized();
		for (size_t next_i = 0; next_i < ret.size(); ++next_i) {
			// const auto& v2 = normalized_ret[next_i];
			if (dist2(all_entries[md.second], all_entries[ret[next_i].second]) <
					dist2(all_entries[md.second], origin)) {
				// if (dist2(v1, v2) <= 1.0) {
				choose = false;
				break;
			}
		}
		if (choose) {
			ret.emplace_back(md);
			// normalized_ret.emplace_back(v1);
			if (ret.size() >= edge_count_mult)
				break;
		}
	}
	to = ret;
	update_edges(layer, from);
}
void par_antitopo_engine::store_vector(const fvec& v) {
	size_t v_index = all_entries.size();
	all_entries.emplace_back(v);

	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));

	hadj_flat_with_lengths.emplace_back();
	hadj_flat.emplace_back();
	hadj_bottom.emplace_back();
	edge_list_locks.emplace_back();
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat_with_lengths[v_index].emplace_back();
		hadj_flat[v_index].emplace_back();
		edge_list_locks[v_index].emplace_back(std::make_unique<std::mutex>());
	}

	// add new layers if necessary
	while (new_max_layer >= max_layer) {
		++max_layer;
		starting_vertex = v_index;
	}
}

void par_antitopo_engine::edit_vector(size_t data_index, const fvec& v) {
	all_entries[data_index] = v;
}
void par_antitopo_engine::improve_entries(
		const std::vector<size_t>& data_indices) {
	size_t num_threads = data_indices.size();

	visit_manager.resize_visit_containers(num_threads, all_entries.size());

	std::barrier sync_point(num_threads);

	auto worker = [&](size_t v_index, size_t thread_index) {
		// if (v_index % 1000 == 0) {
		//	std::cout << "Improving v_index=" << v_index << std::endl;
		// }
		//  get kNN for each layer
		auto kNN_per_layer = get_knn_per_layer(v_index, thread_index);

		const auto local_max_layer = hadj_flat_with_lengths[v_index].size() - 1;
		sync_point.arrive_and_wait();

		// add the found edges to the graph
		size_t layer = 0;
		for (; layer < std::min(max_layer, local_max_layer + 1); ++layer) {
			add_edges_with_lock(layer, v_index, kNN_per_layer[layer], sync_point);
		}
		for (; layer < max_layer; ++layer)
			for (size_t i = 0; i < 3; ++i)
				sync_point.arrive_and_wait();
	};

	std::vector<std::jthread> threads;
	size_t thread_index = 0;
	for (size_t index : data_indices) {
		threads.emplace_back(worker, index, thread_index++);
	}
}

void par_antitopo_engine::_build() { build(); }
void par_antitopo_engine::build() {
	for (size_t block_start_index = 0; block_start_index < all_entries.size();
			 block_start_index += num_threads) {
		std::vector<size_t> data_indices;
		for (size_t data_index = block_start_index;
				 data_index < all_entries.size() &&
				 data_index < block_start_index + num_threads;
				 ++data_index) {
			if (data_index % 20000 == 0) {
				std::cout << "Adding data_index=" << data_index << " to improve block"
									<< std::endl;
			}
			data_indices.emplace_back(data_index);
		}
		improve_entries(data_indices);
	}
#ifdef RECORD_STATS
	// reset before queries
	num_distcomps = 0;
#endif
}

template <bool use_bottomlayer>
std::vector<std::pair<par_antitopo_engine::dist_t, size_t>>
par_antitopo_engine::query_k_at_layer(const fvec& q, size_t layer,
																			const std::vector<size_t>& entry_points,
																			size_t k,
																			std::optional<size_t> thread_index) {
	auto visitref = visit_manager.get_visitref(thread_index, all_entries.size());
	using measured_data = std::pair<dist_t, size_t>;

	auto get_vertex = [&](const size_t& index) constexpr -> std::vector<size_t>& {
		if constexpr (use_bottomlayer) {
			return hadj_bottom[index];
		} else {
			return hadj_flat[index][layer];
		}
	};
	auto get_data = [&](const size_t& data_index) constexpr -> auto& {
		return all_entries[data_index];
	};

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points) {
		entry_points_with_dist.emplace_back(dist2(q, get_data(entry_point)),
																				entry_point);
	}

	std::vector<measured_data> container_candidates, container_nearest;
	container_candidates.reserve(2 * k);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points_with_dist.begin(), entry_points_with_dist.end(),
								 best_elem, std::move(container_candidates));

	std::vector<measured_data> best_candidates;
	best_candidates.reserve(k);
	auto clean_candidates_size = [&]() {
		if (candidates.size() >= 2 * k) {
			best_candidates.clear();
			for (size_t i = 0; i < k; ++i) {
				best_candidates.emplace_back(candidates.top());
				candidates.pop();
			}
			while (!candidates.empty())
				candidates.pop();
			for (const auto& can : best_candidates)
				candidates.emplace(can);
		}
	};
	container_nearest.reserve(k + 1);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points_with_dist.begin(), entry_points_with_dist.end(),
							worst_elem, std::move(container_nearest));
	while (nearest.size() > k)
		nearest.pop();

	for (auto& entry_point : entry_points) {
		visitref.Visit(entry_point);
	}

	size_t min_iters = 2 * std::max(entry_points.size(), k) + 1;
	size_t iter = 0;
	std::vector<size_t> neighbour_list;
	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k &&
				iter >= min_iters) {
			break;
		}
		++iter;
		for (size_t neighbour : get_vertex(cur.second))
			if (visitref.Visit(neighbour)) {
				neighbour_list.emplace_back(neighbour);
			}
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
			for (size_t mult = 0; mult < DIM * sizeof(float) / 64; ++mult)
				_mm_prefetch(((char*)&get_data(neighbour_list[i])) + mult * 64,
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
			dist_t d_next = dist2(q, get_data(next));
			if (nearest.size() < k || d_next < nearest.top().first) {
				candidates.emplace(d_next, next);
				nearest.emplace(d_next, next);
				if (nearest.size() > k)
					nearest.pop();
				clean_candidates_size();
			}
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
		neighbour_list.clear();
	}
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}
std::vector<size_t>
par_antitopo_engine::query_k(const fvec& q, size_t k,
														 std::optional<size_t> thread_index) {
	size_t entry_point = starting_vertex;
	dist_t ep_dist = dist2(all_entries[entry_point], q);
	for (size_t layer = max_layer - 1; layer > 0; --layer) {
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto& neighbour : hadj_flat[entry_point][layer]) {
				_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
				dist_t neighbour_dist = dist2(q, all_entries[neighbour]);
				if (neighbour_dist < ep_dist) {
					entry_point = neighbour;
					ep_dist = neighbour_dist;
					changed = true;
				}
			}
		}
	}

	std::vector<std::pair<dist_t, size_t>> ret_combined = query_k_at_layer<true>(
			q, 0, {entry_point}, k * ef_search_mult, thread_index);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}
std::vector<std::vector<std::pair<par_antitopo_engine::dist_t, size_t>>>
par_antitopo_engine::get_knn_per_layer(size_t data_index,
																			 std::optional<size_t> thread_index) {
	const auto& v = all_entries[data_index];
	const auto local_max_layer = hadj_flat_with_lengths[data_index].size() - 1;
	std::vector<std::vector<std::pair<dist_t, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		{
			size_t entry_point = starting_vertex;
			dist_t ep_dist = dist2(v, all_entries[entry_point]);
			for (size_t layer = max_layer - 1; layer > local_max_layer; --layer) {
				bool changed = true;
				while (changed) {
					changed = false;
					for (auto& neighbour : hadj_flat[entry_point][layer]) {
						_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
						dist_t neighbour_dist = dist2(v, all_entries[neighbour]);
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
		for (int layer = std::min(local_max_layer, max_layer - 1); layer >= 0;
				 --layer) {
			// add old edges as candidates (if any)
			for (size_t old_neighbour : hadj_flat[data_index][layer]) {
				if (old_neighbour != cur[0])
					cur.emplace_back(old_neighbour);
			}
			if (layer == 0) {
				kNN_per_layer.emplace_back(query_k_at_layer<true>(
						v, layer, cur, ef_construction, thread_index));
			} else {
				kNN_per_layer.emplace_back(query_k_at_layer<false>(
						v, layer, cur, ef_construction, thread_index));
			}
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
			cur.resize(1);
			// Don't add data_index to its own neighbours (but do use it for next
			// layer of queries)
			if (!kNN_per_layer.back().empty() &&
					kNN_per_layer.back()[0].second == data_index) {
				kNN_per_layer.back().erase(kNN_per_layer.back().begin());
			}
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}
	return kNN_per_layer;
}
void par_antitopo_engine::add_edges_with_lock(
		size_t layer, size_t data_index,
		const std::vector<std::pair<dist_t, size_t>>& new_edges,
		std::barrier<>& sync_point) {
	std::vector<std::pair<dist_t, size_t>> pruned_edges_dupe;
	{
		// std::lock_guard<std::mutex> guard(*edge_list_locks[data_index][layer]);
		for (const auto& edge : new_edges)
			hadj_flat_with_lengths[data_index][layer].emplace_back(edge);
		prune_edges(layer, data_index, false);
		pruned_edges_dupe = hadj_flat_with_lengths[data_index][layer];
	}
	sync_point.arrive_and_wait();
	//  add bidirectional connections, prune if necessary
	//  three phases for multithreading: determine which to add to, add to the
	//  list, and then call prune if ownership achieved
	std::vector<std::pair<dist_t, size_t>> pruned_edges_dupe_filtered;
	size_t edge_count_mult = M;
	if (layer == 0)
		edge_count_mult *= 2;
	for (auto& md : pruned_edges_dupe) {
		bool same_edge_exists = false;
		bool smaller_edge_exists =
				hadj_flat_with_lengths[md.second][layer].size() < edge_count_mult;
		;
		for (auto& md_other : hadj_flat_with_lengths[md.second][layer]) {
			if (md_other.second == data_index) {
				same_edge_exists = true;
				break;
			}
			if (md_other.first < md.first) {
				smaller_edge_exists = true; // reduce contention for starting nodes
			}
		}
		if (!same_edge_exists && smaller_edge_exists) {
			pruned_edges_dupe_filtered.emplace_back(md);
		}
	}
	sync_point.arrive_and_wait();
	std::vector<std::pair<dist_t, size_t>> pruned_edges_dupe_refiltered;
	for (auto& md : pruned_edges_dupe_filtered) {
		std::lock_guard<std::mutex> guard(*edge_list_locks[md.second][layer]);
		if (hadj_flat_with_lengths[md.second][layer].size() ==
				hadj_flat[md.second][layer].size())
			pruned_edges_dupe_refiltered.emplace_back(md);
		hadj_flat_with_lengths[md.second][layer].emplace_back(md.first, data_index);
	}
	sync_point.arrive_and_wait();
	for (auto& md : pruned_edges_dupe_refiltered) {
		// std::lock_guard<std::mutex> guard(*edge_list_locks[md.second][layer]);
		prune_edges(layer, md.second, true);
	}
}
