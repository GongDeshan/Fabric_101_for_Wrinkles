#include <pyuipc/constitution/elastic_moduli.h>
#include <uipc/constitution/elastic_moduli.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyElasticModuli::PyElasticModuli(py::module& m)
{
    py::class_<ElasticModuli>(m, "ElasticModuli")
        .def_static("lame", &ElasticModuli::lame, py::arg("lambda"), py::arg("mu"))
        .def_static("youngs_shear", &ElasticModuli::youngs_shear, py::arg("E"), py::arg("G"))
        .def_static("youngs_poisson", &ElasticModuli::youngs_poisson, py::arg("E"), py::arg("nu"))
        .def("lambda", &ElasticModuli::lambda)
        .def("mu", &ElasticModuli::mu);
}
}  // namespace pyuipc::constitution
