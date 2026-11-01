#include <pyuipc/constitution/StVK_Nonlinear_Yarn_Rotation_Friction_Hardening_Plastic_Shear_shell.h>
#include <uipc/constitution/StVK_Nonlinear_Yarn_Rotation_Friction_Hardening_Plastic_Shear_shell.h>
#include <pyuipc/common/json.h>

namespace pyuipc::constitution
{

using namespace uipc::constitution;

PyStVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::PyStVKNonlinearYarnRotationFrictionHardeningPlasticShearShell(
    py::module& m)
{
    auto class_StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell =
        py::class_<StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell, FiniteElementConstitution>(
            m, "StVKNonlinearYarnRotationFrictionHardeningPlasticShell");

    class_StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell.def(
        py::init<const Json&>(),
        py::arg("config") =
            StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::default_config());

    class_StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell.def_static(
        "default_config", &StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::default_config);

    class_StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell.def(
        "apply_to",
        &StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::apply_to,
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
        py::arg("thickness")         = 0.01_m,
        py::arg("enable_yarn_rotation") = true);
}
}  // namespace pyuipc::constitution
