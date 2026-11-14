#pragma once
#include <uipc/constitution/finite_element_extra_constitution.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API TriangleWind : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;
  public:
    TriangleWind(const Json& config = default_config());
    void apply_to(geometry::SimplicialComplex& sc,
                  const Vector3& velocity = Vector3::Zero(),
                  Float air_density = 1.225, Float normal_drag = 1.0,
                  Float tangential_drag = 0.0, bool enabled = true) const;
    static Json default_config();
  private:
    U64 get_uid() const noexcept override;
};
}
