#pragma once

#include <cassert>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "ann_engine.h"
#include "topk_t.h"

template <typename SubEngineConfig> struct projection_engine_config {
	size_t num_projections;
	size_t sub_dim;
	bool random_projections; // TODO use this or get rid of it
	SubEngineConfig sub_conf;
	projection_engine_config(size_t _num_projections, size_t _sub_dim,
													 bool _random_projections, SubEngineConfig _sub_conf)
			: num_projections(_num_projections), sub_dim(_sub_dim),
				random_projections(_random_projections), sub_conf(_sub_conf) {
		assert(num_projections > 0);
	}
};

template <typename T, typename SubEngine>
struct projection_engine
		: public ann_engine<T, projection_engine<T, SubEngine>> {
	size_t num_projections;
	size_t sub_dim;
	bool random_projections;
	std::vector<std::unique_ptr<SubEngine>> sub_engines;
	// Eigen::Tensor<T, 3> projections;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	std::vector<Matrix> projections;
	template <typename SubEngineConfig>
	projection_engine(projection_engine_config<SubEngineConfig> conf)
			: num_projections(conf.num_projections), sub_dim(conf.sub_dim),
				random_projections(conf.random_projections)
	// ,sub_engines(conf.num_projections, SubEngine(conf.sub_conf))
	{
		for (size_t projection_i = 0; projection_i < num_projections;
				 ++projection_i)
			sub_engines.emplace_back(std::make_unique<SubEngine>(conf.sub_conf));
	}
	std::vector<vec<T>> all_entries;
	void compute_projections(size_t dim);
	std::vector<vec<T>> perform_projections(const vec<T>& v);
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() {
		return "Projection Engine<" + sub_engines[0]->name() + ">";
	}
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, num_projections);
		add_param(pl, sub_dim);
		add_param(pl, random_projections);
		for (auto& [pname, p] : sub_engines[0]->param_list())
			add_sub_param(pl, "subengine-", pname, p);
		return pl;
	}
};

template <typename T, typename SubEngine>
void projection_engine<T, SubEngine>::compute_projections(size_t dim) {
	projections = std::vector<Matrix>(num_projections, Matrix(sub_dim, dim));
	// Eigen::Tensor<T, 3>(num_projections, sub_dim, dim);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<T> distrib(0, 1);

	for (size_t projection_i = 0; projection_i < num_projections;
			 ++projection_i) {
		std::generate(projections[projection_i].data(),
									projections[projection_i].data() + dim * sub_dim,
									[&distrib, &gen]() { return distrib(gen); });
	}

	// make the bases orthogonal for each projection:
	for (size_t projection_i = 0; projection_i < num_projections;
			 ++projection_i) {
		// Use the unitary matrix from QR decomposition
		// Eigen::MatrixXd current_projection = projections.chip(projection_i, 0);
		// auto current_projection(projections.chip(projection_i, 0));
		projections[projection_i] =
				Eigen::HouseholderQR<Matrix>(projections[projection_i]).householderQ();
		// Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
		// qr( 		current_projection);

		// assert(qr.matrixQR().rows() == d && qr.matrixQR().cols() == n);
		assert(projections[projection_i].matrixQR().rows() == d &&
					 projections[projection_i].matrixQR().cols() == n);

		// projections.chip(projection_i, 0) = qr.householderQ();
	}
}

template <typename T, typename SubEngine>
std::vector<vec<T>>
projection_engine<T, SubEngine>::perform_projections(const vec<T>& v) {
	std::vector<vec<T>> ret;
	ret.reserve(num_projections);
	for (size_t projection_i = 0; projection_i < num_projections;
			 ++projection_i) {
		// ret.emplace_back(projections.chip(projection_i, 0) * v.get_underlying());
		ret.emplace_back(projections[projection_i] * v.get_underlying());
	}
	return ret;
}

template <typename T, typename SubEngine>
void projection_engine<T, SubEngine>::_store_vector(const vec<T>& v) {
	all_entries.push_back(v);
}

template <typename T, typename SubEngine>
void projection_engine<T, SubEngine>::_build() {
	compute_projections(all_entries[0].size());
	// project all vectors and store the projections in their corresponding
	// subengines
	for (const vec<T>& v : all_entries) {
		auto v_projected = perform_projections(v);
		for (size_t projection_i = 0; projection_i < num_projections;
				 ++projection_i) {
			sub_engines[projection_i]->store_vector(v_projected[projection_i]);
		}
	}
	for (size_t projection_i = 0; projection_i < num_projections; ++projection_i)
		sub_engines[projection_i]->build();
}

template <typename T, typename SubEngine>
std::vector<size_t> projection_engine<T, SubEngine>::_query_k(const vec<T>& v,
																															size_t k) {
	topk_t<T> tk(k);

	// TODO a larger number than k should be queried for in the subengines
	// (parametized by 'num_for_1nn')

	auto v_projected = perform_projections(v);
	for (size_t projection_i = 0; projection_i < num_projections;
			 ++projection_i) {
		for (size_t ui :
				 sub_engines[projection_i]->query_k(v_projected[projection_i], k)) {
			tk.consider(dist2(v, all_entries[ui]), ui);
		}
	}

	return tk.to_vector();
}
