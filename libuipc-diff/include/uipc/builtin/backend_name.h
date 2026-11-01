#pragma once
#include <string_view>

#define UIPC_BUILTIN_BACKEND(name) constexpr std::string_view name = #name

namespace uipc::builtin
{
#include <uipc/builtin/details/backend_name.h>
}  // namespace uipc::builtin

#undef UIPC_BUILTIN_BACKEND