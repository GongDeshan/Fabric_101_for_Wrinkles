#include <pyuipc/diff_sim/adjoint_method_feature.h>
#include <uipc/diff_sim/adjoint_method_feature.h>
#include <uipc/core/feature.h>
namespace pyuipc::diff_sim
{
using namespace uipc::diff_sim;
PyAdjointMethodFeature::PyAdjointMethodFeature(py::module& m)
{
    auto class_AdjointMethodFeature =
        py::class_<AdjointMethodFeature, core::IFeature, S<AdjointMethodFeature>>(m, "AdjointMethodFeature");

    class_AdjointMethodFeature.def("select_dofs",
                                   &AdjointMethodFeature::select_dofs,
                                   py::arg("frame"),
                                   py::arg("in_SDI"));

    class_AdjointMethodFeature.def("receive_dofs",
                                   &AdjointMethodFeature::receive_dofs,
                                   py::arg("out_dofs"));

    class_AdjointMethodFeature.def(
        "compute_dLdP",
        [](AdjointMethodFeature& self, const backend::BufferView& out_dLdP, const backend::BufferView& in_dLdX)
        { self.compute_dLdP(out_dLdP, in_dLdX); },
        py::arg("out_dLdP"),
        py::arg("in_dLdX"));

    class_AdjointMethodFeature.attr("FeatureName") = AdjointMethodFeature::FeatureName;
}
}  // namespace pyuipc::diff_sim
