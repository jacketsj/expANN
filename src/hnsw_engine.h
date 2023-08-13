#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "ann_engine.h"

struct hnsw_engine_config {
	size_t max_depth;
	size_t edge_count_mult;
	double coinflip_p;
	hnsw_engine_config(size_t _max_depth, size_t _edge_count_mult,
										 double _coinflip_p)
			: max_depth(_max_depth), edge_count_mult(_edge_count_mult),
				coinflip_p(_coinflip_p) {}
};

// basic lsh method
template <typename T, bool TAKE_FIRST>
struct hnsw_engine : public ann_engine<T, hnsw_engine<T, TAKE_FIRST>> {
	std::random_device rd;
	std::mt19937 gen;
	std::geometric_distribution<> d;
	size_t max_depth;
	size_t edge_count_mult;
	double coinflip_p;
	size_t starting_vertex;
	hnsw_engine(hnsw_engine_config conf)
			: rd(), gen(rd()), d(conf.coinflip_p), max_depth(conf.max_depth),
				edge_count_mult(conf.edge_count_mult), coinflip_p(conf.coinflip_p) {}
	std::vector<vec<T>> all_entries;
	std::vector<size_t> priority;
	std::vector<std::vector<std::vector<size_t>>> hadj;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	// const std::string _name() { return "HNSW Engine"; }
	const std::string _name() { return "HNSW Engine (Random Expanders)"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, max_depth);
		add_param(pl, edge_count_mult);
		add_param(pl, coinflip_p);
		return pl;
	}
};

template <typename T, bool TAKE_FIRST>
void hnsw_engine<T, TAKE_FIRST>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
	// size_t pri = 0;
	// while (std::uniform_int_distribution<>(0, 1)(gen) && pri <= max_depth) {
	// 	++pri;
	// }
	// priority.push_back(pri);
	priority.push_back(std::min(max_depth, size_t(d(gen))));
}

template <typename T, bool TAKE_FIRST>
void hnsw_engine<T, TAKE_FIRST>::_build() {
	assert(all_entries.size() > 0);
	for (size_t depth = 0; depth <= max_depth; ++depth) {
		// find all entries at depth
		std::vector<size_t> vertices;
		for (size_t i = 0; i < all_entries.size(); ++i) {
			if (priority[i] >= depth)
				vertices.push_back(i);
		}
		// if none, terminate
		if (vertices.size() <= 1)
			break;
		// for n vertices, build a graph with edge_count_mult*n random edges
		// (good+sparse expander in expectation)
		std::uniform_int_distribution<> d_verts(0, vertices.size() - 1);
		hadj.emplace_back(all_entries.size());
		for (size_t i = 0; i < vertices.size() * edge_count_mult; ++i) {
			// pick two random vertices
			size_t a = d_verts(gen), b;
			do {
				b = d_verts(gen);
			} while (b == a);
			// create an edge between them at depth, duplicates are fine
			hadj[depth][vertices[a]].push_back(vertices[b]);
			hadj[depth][vertices[b]].push_back(vertices[a]);
			// starting_vertex should be something random in the deepest layer
			starting_vertex = a;
		}
	}
}
template <typename T, bool TAKE_FIRST>
std::vector<size_t> hnsw_engine<T, TAKE_FIRST>::_query_k(const vec<T>& v,
																												 size_t k) {
	std::cout << "Using query (MISSING _query_k impl)" << std::endl; // TODO
	size_t cur = starting_vertex;
	// for each layer, in decreasing depth
	for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		// find the best vertex in the current layer by local search
		bool improvement_found = false;
		do {
			improvement_found = false;
			// otherwise look at all incident edges first
			T best_dist2 = dist2(all_entries[cur], v);
			size_t best = cur;
			for (size_t adj_vert : hadj[layer][cur]) {
				T next_dist2 = dist2(all_entries[adj_vert], v);
				if (next_dist2 < best_dist2) {
					improvement_found = true;
					best = adj_vert;
					// if TAKE_FIRST is active, take the improving first edge seen at each
					// step
					if constexpr (TAKE_FIRST) {
						break;
					} else {
						best_dist2 = next_dist2;
					}
				}
			}
			cur = best;
		} while (improvement_found);
	}
	return {cur};
}
