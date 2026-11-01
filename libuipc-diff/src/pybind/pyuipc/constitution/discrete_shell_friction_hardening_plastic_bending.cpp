#include <pyuipc/constitution/discrete_shell_friction_hardening_plastic_bending.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/discrete_shell_friction_hardening_plastic_bending.h>
#include <pyuipc/common/json.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyDiscreteShellFrictionHardeningPlasticBending::PyDiscreteShellFrictionHardeningPlasticBending(py::module& m)
{
    auto class_DiscreteShellFrictionHardeningPlasticBending =
        py::class_<DiscreteShellFrictionHardeningPlasticBending, FiniteElementExtraConstitution>(
            m, "DiscreteShellFrictionHardeningPlasticBending");

    class_DiscreteShellFrictionHardeningPlasticBending.def(
        py::init<const Json&>(),
        py::arg("config") = DiscreteShellFrictionHardeningPlasticBending::default_config());

    class_DiscreteShellFrictionHardeningPlasticBending.def_static(
        "default_config", &DiscreteShellFrictionHardeningPlasticBending::default_config);

    class_DiscreteShellFrictionHardeningPlasticBending.def(
        "apply_to",
        &DiscreteShellFrictionHardeningPlasticBending::apply_to,
        py::arg("sc"),
        py::arg("bending_stiffness") = 100.0_kPa,
        py::arg("hardening_k")       = 100.0_kPa,
        py::arg("yield_theta")       = 1.57,
        py::arg("plastic_theta")     = 0.0,
        py::arg("friction_k")        = 1.0_kPa,
        py::arg("thres_theta")       = 1.57,
        py::arg("anchor_theta")      = 0.0);
}
}  // namespace pyuipc::constitution
