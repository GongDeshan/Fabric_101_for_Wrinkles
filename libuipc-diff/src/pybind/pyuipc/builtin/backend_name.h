#pragma once
#include <pyuipc/pyuipc.h>

namespace pyuipc::builtin
{
class PyBackendName
{
  public:
    PyBackendName(py::module& m);
};
}  // namespace pyuipc::builtin
