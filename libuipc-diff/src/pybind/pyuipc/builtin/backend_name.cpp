#include <pyuipc/builtin/backend_name.h>

#define UIPC_BUILTIN_BACKEND(name) m.attr(#name) = #name;

namespace pyuipc::builtin
{
PyBackendName::PyBackendName(py::module& m)
{
#include <uipc/builtin/details/backend_name.h>
}
}  // namespace pyuipc::builtin

#undef UIPC_BUILTIN_BACKEND
