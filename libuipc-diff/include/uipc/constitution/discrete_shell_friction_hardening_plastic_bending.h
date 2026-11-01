#pragma once
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/common/unit.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API DiscreteShellFrictionHardeningPlasticBending : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    DiscreteShellFrictionHardeningPlasticBending(const Json& json = default_config());

    void apply_to(geometry::SimplicialComplex& sc,
                  Float                        E             = 100.0_kPa,
                  Float                        Hardening_k   = 100.0_kPa,
                  Float                        yield_theta   = 1.57,
                  Float                        plastic_theta = 0.0,
                  Float                        friction_k    = 1.0_kPa,
                  Float                        thres_theta   = 1.57,
                  Float                        anchor_theta  = 0.0);

    static Json default_config();

  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution
