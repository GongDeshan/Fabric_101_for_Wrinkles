#include <pyuipc/constitution/StVK_Nonlinear_Friction_Hardening_Plastic_Shear_shell.h>
#include <uipc/constitution/StVK_Nonlinear_Friction_Hardening_Plastic_Shear_shell.h>
#include <pyuipc/common/json.h>

namespace pyuipc::constitution
{

using namespace uipc::constitution;

PyStVKNonlinearFrictionHardeningPlasticShearShell::PyStVKNonlinearFrictionHardeningPlasticShearShell(
    py::module& m)
{
    auto class_StVKNonlinearFrictionHardeningPlasticShearShell =
        py::class_<StVKNonlinearFrictionHardeningPlasticShearShell, FiniteElementConstitution>(
            m, "StVKNonlinearFrictionHardeningPlasticShell");

    class_StVKNonlinearFrictionHardeningPlasticShearShell.def(
        py::init<const Json&>(),
        py::arg("config") =
            StVKNonlinearFrictionHardeningPlasticShearShell::default_config());

    class_StVKNonlinearFrictionHardeningPlasticShearShell.def_static(
        "default_config", &StVKNonlinearFrictionHardeningPlasticShearShell::default_config);

    class_StVKNonlinearFrictionHardeningPlasticShearShell.def(
        "apply_to",
        &StVKNonlinearFrictionHardeningPlasticShearShell::apply_to,
        py::arg("sc"),
        py::arg("k11")               = 10.0_kPa,
        py::arg("k12")               = 10.0_kPa,
        py::arg("k22")               = 10.0_kPa,
        py::arg("k33_0")             = 10.0_kPa,
        py::arg("k33_1")             = 10.0_kPa,
        py::arg("hardening_k33")     = 10.0_kPa,
        py::arg("friction_k33")      = 10.0_kPa,
        py::arg("thres_uv")          = 5.0,
        py::arg("anchor_strain_uv")  = 0.0,
        py::arg("yield_uv")          = 5.0,
        py::arg("plastic_strain_uv") = 0.0,
        py::arg("mass_density")      = 1.0e3,
        py::arg("thickness")         = 0.01_m);
}
}  // namespace pyuipc::constitution
