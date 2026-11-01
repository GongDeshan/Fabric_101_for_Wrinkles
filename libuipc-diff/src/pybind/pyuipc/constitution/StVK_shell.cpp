#include <pyuipc/constitution/StVK_shell.h>
#include <uipc/constitution/StVK_shell.h>
#include <pyuipc/common/json.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyStVKShell::PyStVKShell(py::module& m)
{
    auto class_StVKShell = py::class_<StVKShell, FiniteElementConstitution>(m, "StVKShell");

    class_StVKShell.def(py::init<const Json&>(),
                        py::arg("config") = StVKShell::default_config());

    class_StVKShell.def_static("default_config", &StVKShell::default_config);

    class_StVKShell.def("apply_to",
                        &StVKShell::apply_to,
                        py::arg("sc"),
                        py::arg("k11")          = 10.0_MPa,
                        py::arg("k22")          = 10.0_MPa,
                        py::arg("k12")          = 10.0_MPa,
                        py::arg("k33")          = 10.0_MPa,
                        py::arg("mass_density") = 1.0e3,
                        py::arg("thickness")    = 0.01_m);
}
}  // namespace pyuipc::constitution
