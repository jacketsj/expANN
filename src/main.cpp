#include <iostream>

#include "arrangement_engine.h"
#include "basic_bench.h"
#include "brute_force_engine.h"
#include "hnsw_engine.h"
#include "randomgeometry.h"
#include "vec.h"

void plotter(const basic_bench<float>& basic_benchmarker) {
	for (size_t k = 4; k <= 100; k += 8) {
		for (int p2 = 1; p2 <= 8; ++p2) {
			std::cout << k << '\t' << p2 << '\t';
			hnsw_engine<float, false> engine_hnsw(50, k, 0.5 * p2);
			basic_benchmarker.perform_benchmark(engine_hnsw, true);
			std::cout << '\n';
		}
	}
}

int main() {
	// benchmarkers
	basic_bench<float> basic_benchmarker;

	plotter(basic_benchmarker);

	// ANN engines
	brute_force_engine<float> engine_bf;
	arrangement_engine<float> engine_arrange_1(2, 10), engine_arrange_2(4, 64),
			engine_arrange_3(1, 4), engine_arrange_4(2, 8), engine_arrange_5(2, 16);
	hnsw_engine<float, true> engine_hnsw_1(20, 40, 2.5);
	hnsw_engine<float, false> engine_hnsw_2(20, 40, 2.5);

	// perform benchmark on engines
	basic_benchmarker.perform_benchmark(engine_bf);
	basic_benchmarker.perform_benchmark(engine_arrange_1);
	basic_benchmarker.perform_benchmark(engine_arrange_2);
	basic_benchmarker.perform_benchmark(engine_arrange_3);
	basic_benchmarker.perform_benchmark(engine_arrange_4);
	basic_benchmarker.perform_benchmark(engine_arrange_5);
	basic_benchmarker.perform_benchmark(engine_hnsw_1);
	basic_benchmarker.perform_benchmark(engine_hnsw_2);
}
