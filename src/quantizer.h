#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "vec.h"

// TODO this should just be a normal scorer, it's more general
class quantized_scorer {
protected:
	quantized_scorer() = default;

public:
	virtual float score(size_t index) = 0;
	virtual void prefetch(size_t index) = 0;
	// TODO add a prefetched loop with a callback too
};

class quantizer {
protected:
	quantizer() = default;

public:
	// TODO this should just be the constructor
	virtual void
	build(const std::vector<vec<float>::Underlying>& unquantized) = 0;
	virtual std::unique_ptr<quantized_scorer>
	generate_scorer(const vec<float>::Underlying& query) = 0;
};

class quantized_scorer_simple : public quantized_scorer {
private:
	vec<float>::Underlying query;
	const std::vector<vec<float>::Underlying>& stored;
	size_t dimension;

public:
	quantized_scorer_simple(const vec<float>::Underlying& _query,
													const std::vector<vec<float>::Underlying>& _stored)
			: query(_query), stored(_stored), dimension(_query.size()) {}

public:
	virtual float score(size_t index) override {
		return (query - stored.at(index)).squaredNorm();
	}
	virtual void prefetch(size_t index) override {
#ifdef DIM
		for (size_t mult = 0; mult < DIM * sizeof(float) / 64; ++mult)
			_mm_prefetch(((char*)&stored[index]) + mult * 64, _MM_HINT_T0);
#else
		for (size_t mult = 0; mult < dimension * sizeof(float) / 64; ++mult)
			_mm_prefetch(((char*)&stored[index]) + mult * 64, _MM_HINT_T0);
#endif
	}
};

// TODO make this generic for other types
class quantizer_simple : public quantizer {
	std::vector<vec<float>::Underlying> stored;

public:
	quantizer_simple() = default;

	virtual void
	build(const std::vector<vec<float>::Underlying>& unquantized) override {
		stored = unquantized;
	}
	virtual std::unique_ptr<quantized_scorer>
	generate_scorer(const vec<float>::Underlying& query) override {
		return std::make_unique<quantized_scorer_simple>(query, stored);
	}
};

class quantized_scorer_ranged_q8 : public quantized_scorer {
private:
	vec<int8_t>::Underlying query;
	const std::vector<vec<int8_t>::Underlying>& stored;
	size_t dimension;
	float scale_factor;

public:
	quantized_scorer_ranged_q8(
			const vec<int8_t>::Underlying& _query,
			const std::vector<vec<int8_t>::Underlying>& _stored, float _scale_factor)
			: query(_query), stored(_stored), dimension(_query.size()),
				scale_factor(_scale_factor) {}

public:
	virtual float score(size_t index) override {
		// TODO somehow make this work properly. Use stuff from distance.h
		// need to use distance_compare_avx512f_i64
		return scale_factor *
					 float(distance_compare_avx512f_i64(
							 query.data(), stored.at(index).data(), dimension));
		// return (query - stored.at(index)).squaredNorm();
	}
	virtual void prefetch(size_t index) override {
#ifdef DIM
		for (size_t mult = 0; mult < DIM * sizeof(float) / 64; ++mult)
			_mm_prefetch(((char*)&stored[index]) + mult * 64, _MM_HINT_T0);
#else
		for (size_t mult = 0; mult < dimension * sizeof(float) / 64; ++mult)
			_mm_prefetch(((char*)&stored[index]) + mult * 64, _MM_HINT_T0);
#endif
	}
};

class quantizer_ranged_q8 : public quantizer {
	using qvec = vec<int8_t>::Underlying;
	using fvec = vec<float>::Underlying;
	std::vector<vec<int8_t>::Underlying> stored;
	size_t dimension;
	float scale_factor, offset;
	size_t q_min = 0; // std::numeric_limits<int8_t>::min() / 2;
	size_t q_max = std::numeric_limits<int8_t>::max();
	constexpr size_t q_range() { return q_max - q_min + 1; }

	int8_t convert_single(float x) {
		size_t rounded_x =
				static_cast<size_t>(std::round(scale_factor * x + offset));
		return static_cast<int8_t>(std::clamp(rounded_x, q_min, q_max));
	}
	qvec convert_vec(const fvec& v) {
		qvec res(dimension);
		for (size_t d = 0; d < dimension; ++d) {
			res[d] = convert_single(v[d]);
		}
		return res;
	}

public:
	quantizer_ranged_q8() = default;

	virtual void build(const std::vector<fvec>& unquantized) override {
		dimension = unquantized[0].size();

		float max_val = std::numeric_limits<float>::min(),
					min_val = std::numeric_limits<float>::max();
		for (const auto& vec : unquantized)
			for (size_t d = 0; d < dimension; ++d) {
				if (vec[d] > max_val)
					max_val = vec[d];
				if (vec[d] < min_val)
					min_val = vec[d];
			}
		scale_factor = q_range() / (max_val - min_val);
		offset = -scale_factor * min_val - q_min;

		stored = std::vector<qvec>(unquantized.size(), qvec(dimension));
		for (size_t i = 0; i < unquantized.size(); ++i)
			stored[i] = convert_vec(unquantized[i]);
	}
	virtual std::unique_ptr<quantized_scorer>
	generate_scorer(const fvec& query) override {
		return std::make_unique<quantized_scorer_ranged_q8>(
				convert_vec(query), stored, 1 / (scale_factor * scale_factor));
	}
};
