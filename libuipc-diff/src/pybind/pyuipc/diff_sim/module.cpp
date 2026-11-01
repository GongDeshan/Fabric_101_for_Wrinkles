#include <pyuipc/diff_sim/module.h>
#include <pyuipc/diff_sim/parameter_collection.h>
#include <pyuipc/diff_sim/adjoint_method_feature.h>
#include <pyuipc/diff_sim/enable_grad_feature.h>
#include <pyuipc/diff_sim/learning_rate_feature.h>
#include <pyuipc/diff_sim/update_parameter_feature.h>

namespace pyuipc::diff_sim
{
PyModule::PyModule(py::module& m)
{
    PyParameterCollection{m};
    PyAdjointMethodFeature{m};
    PyEnableGradFeature{m};
    PyLearningRateFeature{m};
    PyUpdateParameterFeature{m};
}
}  // namespace pyuipc::diff_sim
