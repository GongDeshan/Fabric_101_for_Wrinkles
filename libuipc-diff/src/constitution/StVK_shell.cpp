#include <uipc/constitution/StVK_shell.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/constitution/conversion.h>
#include <uipc/common/log.h>

namespace uipc::constitution
{
REGISTER_CONSTITUTION_UIDS()
{
    using namespace uipc::builtin;
    list<UIDInfo> uids;
    uids.push_back(UIDInfo{.uid = 40, .name = "StVKShell", .type = string{builtin::FiniteElement}});
    return uids;
}

StVKShell::StVKShell(const Json& config) noexcept
    : m_config(config)
{
}

void StVKShell::apply_to(geometry::SimplicialComplex& sc,
                         Float                        k11,
                         Float                        k22,
                         Float                        k12,
                         Float                        k33,
                         Float                        mass_density,
                         Float                        thickness) const
{
    Base::apply_to(sc, mass_density, thickness);

    UIPC_ASSERT(sc.dim() == 2, "StVK only supports 2D simplicial complex");

    auto k11_attr = sc.triangles().find<Float>("k11");
    if(!k11_attr)
        k11_attr = sc.triangles().create<Float>("k11", k11);
    std::ranges::fill(geometry::view(*k11_attr), k11);

    auto k22_attr = sc.triangles().find<Float>("k22");
    if(!k22_attr)
        k22_attr = sc.triangles().create<Float>("k22", k22);
    std::ranges::fill(geometry::view(*k22_attr), k22);

    auto k12_attr = sc.triangles().find<Float>("k12");
    if(!k12_attr)
        k12_attr = sc.triangles().create<Float>("k12", k12);
    std::ranges::fill(geometry::view(*k12_attr), k12);

    auto k33_attr = sc.triangles().find<Float>("k33");
    if(!k33_attr)
        k33_attr = sc.triangles().create<Float>("k33", k33);
    std::ranges::fill(geometry::view(*k33_attr), k33);

    auto rest_x_attr = sc.triangles().find<Vector6>("rest_x");
    if(!rest_x_attr)
        rest_x_attr = sc.triangles().create<Vector6>("rest_x", Vector6::Zero());
    std::ranges::fill(geometry::view(*rest_x_attr), Vector6::Zero());
}

Json StVKShell::default_config() noexcept
{
    return Json::object();
}

U64 StVKShell::get_uid() const noexcept
{
    return 40;
}
}  // namespace uipc::constitution
