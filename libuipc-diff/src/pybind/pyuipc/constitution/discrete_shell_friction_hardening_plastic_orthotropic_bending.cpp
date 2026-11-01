#include <pyuipc/constitution/discrete_shell_friction_hardening_plastic_orthotropic_bending.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/discrete_shell_friction_hardening_plastic_orthotropic_bending.h>
#include <pyuipc/common/json.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyDiscreteShellFrictionHardeningPlasticOrthotropicBending::PyDiscreteShellFrictionHardeningPlasticOrthotropicBending(
    py::module& m)
{
    auto class_DiscreteShellFrictionHardeningPlasticOrthotroipcBending =
        py::class_<DiscreteShellFrictionHardeningPlasticOrthotropicBending, FiniteElementExtraConstitution>(
            m, "DiscreteShellFrictionHardeningPlasticOrthotropicBending");

    class_DiscreteShellFrictionHardeningPlasticOrthotroipcBending.def(
        py::init<const Json&>(),
        py::arg("config") =
            DiscreteShellFrictionHardeningPlasticOrthotropicBending::default_config());

    class_DiscreteShellFrictionHardeningPlasticOrthotroipcBending.def_static(
        "default_config",
        &DiscreteShellFrictionHardeningPlasticOrthotropicBending::default_config);

    class_DiscreteShellFrictionHardeningPlasticOrthotroipcBending.def(
        "apply_to",
        &DiscreteShellFrictionHardeningPlasticOrthotropicBending::apply_to,
        py::arg("sc"),
        py::arg("bending_stiffness_warp") = 100.0_kPa,
        py::arg("bending_stiffness_weft") = 100.0_kPa,
        py::arg("bending_torsion")        = 50.0_kPa,
        py::arg("bending_poisson")        = 0.0,
        py::arg("hardening_k_warp")       = 100.0_kPa,
        py::arg("hardening_k_weft")       = 100.0_kPa,
        py::arg("yield_theta_warp")       = 1.57,
        py::arg("yield_theta_weft")       = 1.57,
        py::arg("plastic_theta_warp")     = 0.0,
        py::arg("plastic_theta_weft")     = 0.0,
        py::arg("friction_k_warp")        = 1.0_kPa,
        py::arg("friction_k_weft")        = 1.0_kPa,
        py::arg("thres_theta_warp")       = 1.57,
        py::arg("thres_theta_weft")       = 1.57,
        py::arg("anchor_theta_warp")      = 0.0,
        py::arg("anchor_theta_weft")      = 0.0);
}
}  // namespace pyuipc::constitution
