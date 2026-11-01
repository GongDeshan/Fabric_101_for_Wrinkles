#include <pyuipc/diff_sim/learning_rate_feature.h>
#include <uipc/diff_sim/learning_rate_feature.h>

namespace pyuipc::diff_sim
{
using namespace uipc::diff_sim;
PyLearningRateFeature::PyLearningRateFeature(py::module& m)
{
    auto class_PyLearningRateFeature =
        py::class_<LearningRateFeature, core::IFeature, S<LearningRateFeature>>(m, "LearningRateFeature");

    class_PyLearningRateFeature.attr("FeatureName") = LearningRateFeature::FeatureName;

    class_PyLearningRateFeature.def("learning_rate", &LearningRateFeature::learning_rate);
}
}  // namespace pyuipc::diff_sim
