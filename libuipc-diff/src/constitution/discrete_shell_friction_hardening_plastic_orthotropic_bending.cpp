#include <uipc/constitution/discrete_shell_friction_hardening_plastic_orthotropic_bending.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/constitution_uid_auto_register.h>

namespace uipc::constitution
{
constexpr U64 DiscreteShellBendingUID = 24;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = DiscreteShellBendingUID;
    info.name = "DiscreteShellFrictionHardeningPlasticOrthotropicBending";
    info.type = string{builtin::FiniteElement};
    uid_infos.push_back(info);
    return uid_infos;
}

DiscreteShellFrictionHardeningPlasticOrthotropicBending::DiscreteShellFrictionHardeningPlasticOrthotropicBending(
    const Json& json)
    : m_config{json}
{
}

void DiscreteShellFrictionHardeningPlasticOrthotropicBending::apply_to(
    geometry::SimplicialComplex& sc,
    Float                        E_warp,
    Float                        E_weft,
    Float                        E_torsion,
    Float                        E_poisson,
    Float                        Hardening_k_warp,
    Float                        Hardening_k_weft,
    Float                        yield_theta_warp,
    Float                        yield_theta_weft,
    Float                        plastic_theta_warp,
    Float                        plastic_theta_weft,
    Float                        friction_k_warp,
    Float                        friction_k_weft,
    Float                        thres_theta_warp,
    Float                        thres_theta_weft,
    Float                        anchor_theta_warp,
    Float                        anchor_theta_weft)
{
    Base::apply_to(sc);

    auto bs_warp = sc.edges().find<Float>("bending_stiffness_warp");
    if(!bs_warp)
    {
        bs_warp = sc.edges().create<Float>("bending_stiffness_warp");
    }
    auto bs_warp_view = geometry::view(*bs_warp);
    std::ranges::fill(bs_warp_view, E_warp);

    auto bs_weft = sc.edges().find<Float>("bending_stiffness_weft");
    if(!bs_weft)
    {
        bs_weft = sc.edges().create<Float>("bending_stiffness_weft");
    }
    auto bs_weft_view = geometry::view(*bs_weft);
    std::ranges::fill(bs_weft_view, E_weft);

    auto bs_torsion = sc.edges().find<Float>("bending_torsion");
    if(!bs_torsion)
    {
        bs_torsion = sc.edges().create<Float>("bending_torsion");
    }
    auto bs_torsion_view = geometry::view(*bs_torsion);
    std::ranges::fill(bs_torsion_view, E_torsion);

    auto bs_poisson = sc.edges().find<Float>("bending_poisson");
    if(!bs_poisson)
    {
        bs_poisson = sc.edges().create<Float>("bending_poisson");
    }
    auto bs_poisson_view = geometry::view(*bs_poisson);
    std::ranges::fill(bs_poisson_view, E_poisson);



    auto hks_warp = sc.edges().find<Float>("hardening_k_warp");
    if(!hks_warp)
    {
        hks_warp = sc.edges().create<Float>("hardening_k_warp");
    }
    auto hks_warp_view = geometry::view(*hks_warp);
    std::ranges::fill(hks_warp_view, Hardening_k_warp);


    auto hks_weft = sc.edges().find<Float>("hardening_k_weft");
    if(!hks_weft)
    {
        hks_weft = sc.edges().create<Float>("hardening_k_weft");
    }
    auto hks_weft_view = geometry::view(*hks_weft);
    std::ranges::fill(hks_weft_view, Hardening_k_weft);

    auto ds = sc.edges().find<Float>("dihedral_theta");
    if(!ds)
    {
        ds = sc.edges().create<Float>("dihedral_theta");
    }
    auto ds_view = geometry::view(*ds);
    std::ranges::fill(ds_view, 0.0);

    auto bias_angle_attr = sc.edges().find<Float>("bias_angle");
    if(!bias_angle_attr)
        bias_angle_attr = sc.edges().create<Float>("bias_angle", 0.0);

    auto yt_warp = sc.edges().find<Float>("yield_theta_warp");
    if(!yt_warp)
    {
        yt_warp = sc.edges().create<Float>("yield_theta_warp");
    }
    auto yt_warp_view = geometry::view(*yt_warp);
    std::ranges::fill(yt_warp_view, yield_theta_warp);

    auto yt_weft = sc.edges().find<Float>("yield_theta_weft");
    if(!yt_weft)
    {
        yt_weft = sc.edges().create<Float>("yield_theta_weft");
    }
    auto yt_weft_view = geometry::view(*yt_weft);
    std::ranges::fill(yt_weft_view, yield_theta_weft);

    auto y0t_warp = sc.edges().find<Float>("yield_theta_ori_warp");
    if(!y0t_warp)
    {
        y0t_warp = sc.edges().create<Float>("yield_theta_ori_warp");
    }
    auto y0t_warp_view = geometry::view(*y0t_warp);
    std::ranges::fill(y0t_warp_view, yield_theta_warp);

    auto y0t_weft = sc.edges().find<Float>("yield_theta_ori_weft");
    if(!y0t_weft)
    {
        y0t_weft = sc.edges().create<Float>("yield_theta_ori_weft");
    }
    auto y0t_weft_view = geometry::view(*y0t_weft);
    std::ranges::fill(y0t_weft_view, yield_theta_weft);

    auto pt_warp = sc.edges().find<Float>("plastic_theta_warp");
    if(!pt_warp)
    {
        pt_warp = sc.edges().create<Float>("plastic_theta_warp");
    }
    auto pt_warp_view = geometry::view(*pt_warp);
    std::ranges::fill(pt_warp_view, plastic_theta_warp);


    auto pt_weft = sc.edges().find<Float>("plastic_theta_weft");
    if(!pt_weft)
    {
        pt_weft = sc.edges().create<Float>("plastic_theta_weft");
    }
    auto pt_weft_view = geometry::view(*pt_weft);
    std::ranges::fill(pt_weft_view, plastic_theta_weft);

    auto ks_weft = sc.edges().find<Float>("friction_stiffness_weft");
    if(!ks_weft)
    {
        ks_weft = sc.edges().create<Float>("friction_stiffness_weft");
    }
    auto ks_weft_view = geometry::view(*ks_weft);
    std::ranges::fill(ks_weft_view, friction_k_weft);

    auto ks_warp = sc.edges().find<Float>("friction_stiffness_warp");
    if(!ks_warp)
    {
        ks_warp = sc.edges().create<Float>("friction_stiffness_warp");
    }
    auto ks_warp_view = geometry::view(*ks_warp);
    std::ranges::fill(ks_warp_view, friction_k_warp);

    auto tht_warp = sc.edges().find<Float>("thres_theta_warp");
    if(!tht_warp)
    {
        tht_warp = sc.edges().create<Float>("thres_theta_warp");
    }
    auto tht_warp_view = geometry::view(*tht_warp);
    std::ranges::fill(tht_warp_view, thres_theta_warp);

    auto tht_weft = sc.edges().find<Float>("thres_theta_weft");
    if(!tht_weft)
    {
        tht_weft = sc.edges().create<Float>("thres_theta_weft");
    }
    auto tht_weft_view = geometry::view(*tht_weft);
    std::ranges::fill(tht_weft_view, thres_theta_weft);

    auto acht_warp = sc.edges().find<Float>("anchor_theta_warp");
    if(!acht_warp)
    {
        acht_warp = sc.edges().create<Float>("anchor_theta_warp");
    }
    auto acht_warp_view = geometry::view(*acht_warp);
    std::ranges::fill(acht_warp_view, anchor_theta_warp);

    auto acht_weft = sc.edges().find<Float>("anchor_theta_weft");
    if(!acht_weft)
    {
        acht_weft = sc.edges().create<Float>("anchor_theta_weft");
    }
    auto acht_weft_view = geometry::view(*acht_weft);
    std::ranges::fill(acht_weft_view, anchor_theta_weft);

    printf("Friction Hardening Plastic Orthotropic Bending Apply To Done\r\n");
}

U64 DiscreteShellFrictionHardeningPlasticOrthotropicBending::get_uid() const noexcept
{
    return DiscreteShellBendingUID;
}

Json DiscreteShellFrictionHardeningPlasticOrthotropicBending::default_config()
{
    return Json::object();
}
}  // namespace uipc::constitution
