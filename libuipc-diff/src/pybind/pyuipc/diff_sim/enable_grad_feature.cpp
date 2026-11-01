#include <pyuipc/diff_sim/enable_grad_feature.h>
#include <uipc/diff_sim/enable_grad_feature.h>
#include <uipc/core/feature.h>
namespace pyuipc::diff_sim
{
using namespace uipc::diff_sim;
PyEnableGradFeature::PyEnableGradFeature(py::module& m)
{
    auto class_AdjointMethodFeature =
        py::class_<EnableGradFeature, core::IFeature, S<EnableGradFeature>>(m, "EnableGradFeature");
    class_AdjointMethodFeature.attr("FeatureName") = EnableGradFeature::FeatureName;

    class_AdjointMethodFeature.def("no_grad", &EnableGradFeature::no_grad);
    class_AdjointMethodFeature.def("with_grad", &EnableGradFeature::with_grad);
}
}  // namespace pyuipc::diff_sim
