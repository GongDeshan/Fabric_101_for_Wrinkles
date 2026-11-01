#pragma once
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/common/unit.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API DiscreteShellFrictionHardeningPlasticOrthotropicBending
    : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    DiscreteShellFrictionHardeningPlasticOrthotropicBending(const Json& json = default_config());

    void apply_to(geometry::SimplicialComplex& sc,
                  Float                        E_warp            = 100.0_kPa,
                  Float                        E_weft            = 100.0_kPa,
                  Float                        E_torsion         = 50.0_kPa,
                  Float                        E_poission        = 0.0,
                  Float                        Hardening_k_warp  = 100.0_kPa,
                  Float                        Hardening_k_weft  = 100.0_kPa,
                  Float                        yield_theta_warp  = 1.57,
                  Float                        yield_theta_weft  = 1.57,
                  Float                        plastic_warp      = 0.0,
                  Float                        plastic_weft      = 0.0,
                  Float                        friction_k_warp   = 1.0_kPa,
                  Float                        friction_k_weft   = 1.0_kPa,
                  Float                        thres_theta_warp  = 1.57,
                  Float                        thres_theta_weft  = 1.57,
                  Float                        anchor_theta_warp = 0.0,
                  Float                        anchor_theta_weft = 0.0);

    static Json default_config();

  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution
