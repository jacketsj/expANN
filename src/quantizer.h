#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <vec.h>

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
