#include <iostream>

#include "arrangement_engine.h"
#include "basic_bench.h"
#include "brute_force_engine.h"
#include "randomgeometry.h"
#include "vec.h"

int main() {
	// arragement_generator<float> ag;

	// ANN engines
	brute_force_engine<float> engine_bf;
	arrangement_engine<float> engine_arrange;

	// benchmarkers
	basic_bench<float> basic_benchmarker;

	// perform benchmark on engines
	basic_benchmarker.perform_benchmark(engine_bf);
	basic_benchmarker.perform_benchmark(engine_arrange);
}
