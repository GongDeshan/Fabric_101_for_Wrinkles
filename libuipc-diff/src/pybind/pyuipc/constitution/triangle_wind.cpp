#include <pyuipc/constitution/triangle_wind.h>
#include <uipc/constitution/triangle_wind.h>
#include <pyuipc/common/json.h>
#include <pybind11/eigen.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyTriangleWind::PyTriangleWind(py::module& m)
{
    py::class_<TriangleWind,FiniteElementExtraConstitution>(m,"TriangleWind")
        .def(py::init<const Json&>(),py::arg("config")=TriangleWind::default_config())
        .def_static("default_config",&TriangleWind::default_config)
        .def("apply_to",&TriangleWind::apply_to,py::arg("sc"),
             py::arg("velocity")=Vector3::Zero().eval(),py::arg("air_density")=1.225,
             py::arg("normal_drag")=1.0,py::arg("tangential_drag")=0.0,
             py::arg("enabled")=true);
}
}
