#include <iostream>

#include "basic_bench.h"
#include "brute_force_engine.h"
#include "randomgeometry.h"
#include "vec.h"

int main() {
	// arragement_generator<float> ag;

	// ANN engines
	brute_force_engine<float> engine_bf;

	// benchmarkers
	basic_bench<float> basic_benchmarker;

	// perform benchmark on engines
	basic_benchmarker.perform_benchmark(engine_bf);
}
