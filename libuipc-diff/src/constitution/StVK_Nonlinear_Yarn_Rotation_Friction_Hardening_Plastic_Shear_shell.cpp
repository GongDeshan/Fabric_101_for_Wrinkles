#include <uipc/constitution/StVK_Nonlinear_Yarn_Rotation_Friction_Hardening_Plastic_Shear_shell.h>
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
    uids.push_back(UIDInfo{.uid = 1068,
                           .name = "StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell",
                           .type = string{builtin::FiniteElement}});
    return uids;
}

StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell(
    const Json& config) noexcept
    : m_config(config)
{
}

void StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::apply_to(geometry::SimplicialComplex& sc,
                                                               Float k11,
                                                               Float k12,
                                                               Float k22,
                                                               Float k33_0,
                                                               Float k33_1,
                                                               Float hardening_k33,
                                                               Float friction_k33,
                                                               Float thres_uv,
                                                               Float anchor_strain_uv,
                                                               Float yield_uv,
                                                               Float plastic_strain_uv,
                                                               Float mass_density,
                                                               Float thickness,
                                                               bool enable_yarn_rotation) const
{
    Base::apply_to(sc, mass_density, thickness);
    auto rotation = sc.triangles().find<I32>("enable_yarn_rotation");
    if(!rotation) rotation=sc.triangles().create<I32>("enable_yarn_rotation",enable_yarn_rotation?1:0);
    std::ranges::fill(geometry::view(*rotation),enable_yarn_rotation?1:0);

    UIPC_ASSERT(sc.dim() == 2, "StVK only supports 2D simplicial complex");

    auto k11_attr = sc.triangles().find<Float>("k11");
    if(!k11_attr)
        k11_attr = sc.triangles().create<Float>("k11", k11);
    std::ranges::fill(geometry::view(*k11_attr), k11);

    auto k12_attr = sc.triangles().find<Float>("k12");
    if(!k12_attr)
        k12_attr = sc.triangles().create<Float>("k12", k12);
    std::ranges::fill(geometry::view(*k12_attr), k12);

    auto k22_attr = sc.triangles().find<Float>("k22");
    if(!k22_attr)
        k22_attr = sc.triangles().create<Float>("k22", k22);
    std::ranges::fill(geometry::view(*k22_attr), k22);
    // Nonlinear Shearing
    auto k33_0_attr = sc.triangles().find<Float>("k33_0");
    if(!k33_0_attr)
        k33_0_attr = sc.triangles().create<Float>("k33_0", k33_0);
    std::ranges::fill(geometry::view(*k33_0_attr), k33_0);

    auto k33_1_attr = sc.triangles().find<Float>("k33_1");
    if(!k33_1_attr)
        k33_1_attr = sc.triangles().create<Float>("k33_1", k33_1);
    std::ranges::fill(geometry::view(*k33_1_attr), k33_1);

    auto hardening_k33_attr = sc.triangles().find<Float>("hardening_k33");
    if(!hardening_k33_attr)
        hardening_k33_attr = sc.triangles().create<Float>("hardening_k33", hardening_k33);
    std::ranges::fill(geometry::view(*hardening_k33_attr), hardening_k33);

    auto rest_x_attr = sc.triangles().find<Vector6>("rest_x");
    if(!rest_x_attr)
        rest_x_attr = sc.triangles().create<Vector6>("rest_x", Vector6::Zero());
    std::ranges::fill(geometry::view(*rest_x_attr), Vector6::Zero());

    auto friction_k33_attr = sc.triangles().find<Float>("friction_k33");
    if(!friction_k33_attr)
        friction_k33_attr = sc.triangles().create<Float>("friction_k33", friction_k33);
    std::ranges::fill(geometry::view(*friction_k33_attr), friction_k33);

    auto thres_uv_attr = sc.triangles().find<Float>("thres_uv");
    if(!thres_uv_attr)
        thres_uv_attr = sc.triangles().create<Float>("thres_uv", thres_uv);
    std::ranges::fill(geometry::view(*thres_uv_attr), thres_uv);

    auto anchor_uv_attr = sc.triangles().find<Float>("anchor_strain_uv");
    if(!anchor_uv_attr)
        anchor_uv_attr = sc.triangles().create<Float>("anchor_strain_uv", anchor_strain_uv);
    std::ranges::fill(geometry::view(*anchor_uv_attr), anchor_strain_uv);

    auto yield_uv_attr = sc.triangles().find<Float>("yield_uv");
    if(!yield_uv_attr)
        yield_uv_attr = sc.triangles().create<Float>("yield_uv", yield_uv);
    std::ranges::fill(geometry::view(*yield_uv_attr), yield_uv);

    auto yield_ori_uv_attr = sc.triangles().find<Float>("yield_ori_uv");
    if(!yield_ori_uv_attr)
        yield_ori_uv_attr = sc.triangles().create<Float>("yield_ori_uv", yield_uv);
    std::ranges::fill(geometry::view(*yield_ori_uv_attr), yield_uv);

    auto plastic_uv_attr = sc.triangles().find<Float>("plastic_strain_uv");
    if(!plastic_uv_attr)
        plastic_uv_attr =
            sc.triangles().create<Float>("plastic_strain_uv", plastic_strain_uv);
    std::ranges::fill(geometry::view(*plastic_uv_attr), plastic_strain_uv);
}

Json StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::default_config() noexcept
{
    return Json::object();
}

U64 StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell::get_uid() const noexcept
{
    return 1068;
}
}  // namespace uipc::constitution
