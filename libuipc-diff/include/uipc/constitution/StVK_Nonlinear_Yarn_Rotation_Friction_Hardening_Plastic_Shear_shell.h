#pragma once
#include <uipc/constitution/finite_element_constitution.h>
#include <uipc/common/unit.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell : public FiniteElementConstitution
{
    using Base = FiniteElementConstitution;

  public:
    StVKNonlinearYarnRotationFrictionHardeningPlasticShearShell(const Json& config = default_config()) noexcept;

    void apply_to(geometry::SimplicialComplex& sc,
                  Float                        k11               = 10.0_kPa,
                  Float                        k12               = 10.0_kPa,
                  Float                        k22               = 10.0_kPa,
                  Float                        k33_0             = 10.0_kPa,
                  Float                        k33_1             = 10.0_kPa,
                  Float                        hardening_k33     = 10.0_kPa,
                  Float                        friction_k33      = 10.0_kPa,
                  Float                        thres_uv          = 5.0,
                  Float                        anchor_strain_uv  = 0.0,
                  Float                        yield_uv          = 5.0,
                  Float                        plastic_strain_uv = 0.0,
                  Float                        mass_density      = 1e3,
                  Float                        thickness = 0.01_m,
                  bool                         enable_yarn_rotation = true) const;

    static Json default_config() noexcept;

  protected:
    virtual U64 get_uid() const noexcept override;

  private:
    Json m_config;
};
}  // namespace uipc::constitution
