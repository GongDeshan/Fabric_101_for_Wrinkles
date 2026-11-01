#include <uipc/constitution/discrete_shell_friction_hardening_plastic_bending.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/constitution_uid_auto_register.h>

namespace uipc::constitution
{
constexpr U64 DiscreteShellBendingUID = 23;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = DiscreteShellBendingUID;
    info.name = "DiscreteShellFrictionHardeningPlasticBending";
    info.type = string{builtin::FiniteElement};
    uid_infos.push_back(info);
    return uid_infos;
}

DiscreteShellFrictionHardeningPlasticBending::DiscreteShellFrictionHardeningPlasticBending(const Json& json)
    : m_config{json}
{
}

void DiscreteShellFrictionHardeningPlasticBending::apply_to(geometry::SimplicialComplex& sc,
                                                            Float E,
                                                            Float Hardening_k,
                                                            Float yield_theta,
                                                            Float plastic_theta,
                                                            Float friction_k,
                                                            Float thres_theta,
                                                            Float anchor_theta)
{
    printf("Start Apply To\r\n");
    
    Base::apply_to(sc);

    auto bs = sc.edges().find<Float>("bending_stiffness");
    if(!bs)
    {
        bs = sc.edges().create<Float>("bending_stiffness");
    }
    auto bs_view = geometry::view(*bs);
    std::ranges::fill(bs_view, E);

    auto hks = sc.edges().find<Float>("hardening_k");
    if(!hks)
    {
        hks = sc.edges().create<Float>("hardening_k");
    }
    auto hks_view = geometry::view(*hks);
    std::ranges::fill(hks_view, Hardening_k);

    auto ds = sc.edges().find<Float>("dihedral_theta");
    if(!ds)
    {
        ds = sc.edges().create<Float>("dihedral_theta");
    }
    auto ds_view = geometry::view(*ds);
    std::ranges::fill(ds_view, 0.0);

    auto yt = sc.edges().find<Float>("yield_theta");
    if(!yt)
    {
        yt = sc.edges().create<Float>("yield_theta");
    }
    auto yt_view = geometry::view(*yt);
    std::ranges::fill(yt_view, yield_theta);

    auto y0t = sc.edges().find<Float>("yield_theta_ori");
    if(!y0t)
    {
        y0t = sc.edges().create<Float>("yield_theta_ori");
    }
    auto y0t_view = geometry::view(*y0t);
    std::ranges::fill(y0t_view, yield_theta);

    auto pt = sc.edges().find<Float>("plastic_theta");
    if(!pt)
    {
        pt = sc.edges().create<Float>("plastic_theta");
    }
    auto pt_view = geometry::view(*pt);
    std::ranges::fill(pt_view, plastic_theta);

    auto ks = sc.edges().find<Float>("friction_stiffness");
    if(!ks)
    {
        ks = sc.edges().create<Float>("friction_stiffness");
    }
    auto ks_view = geometry::view(*ks);
    std::ranges::fill(ks_view, friction_k);

    auto tht = sc.edges().find<Float>("thres_theta");
    if(!tht)
    {
        tht = sc.edges().create<Float>("thres_theta");
    }
    auto tht_view = geometry::view(*tht);
    std::ranges::fill(tht_view, thres_theta);

    auto acht = sc.edges().find<Float>("anchor_theta");
    if(!acht)
    {
        acht = sc.edges().create<Float>("anchor_theta");
    }
    auto acht_view = geometry::view(*acht);
    std::ranges::fill(acht_view, anchor_theta);

    printf("Friction Hardening Plastic Apply To Done\r\n");
}

U64 DiscreteShellFrictionHardeningPlasticBending::get_uid() const noexcept
{
    return DiscreteShellBendingUID;
}

Json DiscreteShellFrictionHardeningPlasticBending::default_config()
{
    return Json::object();
}
}  // namespace uipc::constitution
