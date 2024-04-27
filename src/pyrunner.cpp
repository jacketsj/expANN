#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ann_engine.h"
#include "antitopo_engine.h"

namespace py = pybind11;

namespace MODULE_NAME {
PYBIND11_MODULE(MODULE_NAME, m) {
	py::class_<vec<float>>(m, "Vec", py::module_local())
			.def(py::init<>())
			.def(py::init<const std::vector<float>&>())
			.def("data",
					 [](vec<float>& v) {
						 return py::memoryview::from_buffer(v.data(), {v.size()},
																								{sizeof(float)});
					 })
			.def("size", &vec<float>::size)
			.def("normalize", &vec<float>::normalize);

	py::class_<antitopo_engine<float>>(m, "AntitopoEngine", py::module_local())
			.def(py::init<size_t, size_t, size_t, size_t, bool>())
			.def("name", &antitopo_engine<float>::name)
			.def("param_list", &antitopo_engine<float>::param_list)
			.def("store_vector", &antitopo_engine<float>::store_vector)
			.def("build", &antitopo_engine<float>::build)
			.def("query_k", &antitopo_engine<float>::query_k)
			.def("set_ef_search", &antitopo_engine<float>::set_ef_search);
}
} // namespace MODULE_NAME
