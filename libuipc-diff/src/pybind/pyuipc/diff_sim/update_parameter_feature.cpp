#include <pyuipc/diff_sim/update_parameter_feature.h>
#include <uipc/diff_sim/update_parameter_feature.h>

namespace pyuipc::diff_sim
{
using namespace uipc::diff_sim;
PyUpdateParameterFeature::PyUpdateParameterFeature(py::module& m)
{
    auto class_UpdateParameterFeature =
        py::class_<UpdateParameterFeature, core::IFeature, S<UpdateParameterFeature>>(
            m, "UpdateParameterFeature");
    class_UpdateParameterFeature.attr("FeatureName") = UpdateParameterFeature::FeatureName;

    class_UpdateParameterFeature.def("update", &UpdateParameterFeature::update);
}
}  // namespace pyuipc::diff_sim
