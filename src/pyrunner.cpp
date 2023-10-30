#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ann_engine.h"
#include "ehnsw_engine_basic_fast_disk.h"

namespace py = pybind11;

PYBIND11_MODULE(expann_py, m) {
	py::class_<vec<float>>(m, "Vec")
			.def(py::init<>())
			.def(py::init<const std::vector<float>&>())
			//.def_readwrite("data", &vec<float>::data);
			.def("data",
					 [](vec<float>& v) {
						 return py::memoryview::from_buffer(v.data(), {v.size()},
																								{sizeof(float)});
					 })
			.def("size", &vec<float>::size);

	py::class_<ehnsw_engine_basic_fast_disk<float>>(m, "expANN")
			.def(py::init<>())
			.def("name", &ehnsw_engine_basic_fast_disk<float>::name)
			.def("param_list", &ehnsw_engine_basic_fast_disk<float>::param_list)
			.def("store_vector", &ehnsw_engine_basic_fast_disk<float>::store_vector)
			.def("build", &ehnsw_engine_basic_fast_disk<float>::build)
			.def("query_k", &ehnsw_engine_basic_fast_disk<float>::query_k)
			.def("query_k_batch",
					 &ehnsw_engine_basic_fast_disk<float>::query_k_batch);
}
