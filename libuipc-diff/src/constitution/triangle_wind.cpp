#include <uipc/constitution/triangle_wind.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/constitution_type.h>
#include <cmath>

namespace uipc::constitution
{
constexpr U64 TriangleWindUID = 1069;
REGISTER_CONSTITUTION_UIDS()
{
    return list<builtin::UIDInfo>{builtin::UIDInfo{
        .uid=TriangleWindUID, .name="TriangleWind", .type=string{builtin::FiniteElement}}};
}
TriangleWind::TriangleWind(const Json&) {}
Json TriangleWind::default_config() { return Json::object(); }
U64 TriangleWind::get_uid() const noexcept { return TriangleWindUID; }

void TriangleWind::apply_to(geometry::SimplicialComplex& sc, const Vector3& velocity,
                           Float air_density, Float normal_drag,
                           Float tangential_drag, bool enabled) const
{
    UIPC_ASSERT(sc.dim()==2, "TriangleWind requires a triangle surface");
    UIPC_ASSERT(velocity.allFinite() && std::isfinite(air_density)
                && std::isfinite(normal_drag) && std::isfinite(tangential_drag)
                && air_density>=0 && normal_drag>=0 && tangential_drag>=0,
                "TriangleWind parameters must be finite and coefficients nonnegative");
    Base::apply_to(sc);
    auto set=[&]<typename T>(std::string_view name, const T& value)
    {
        auto a=sc.triangles().find<T>(name);
        if(!a) a=sc.triangles().create<T>(name,value);
        std::ranges::fill(geometry::view(*a),value);
    };
    set("wind/velocity",velocity);
    set("wind/air_density",air_density);
    set("wind/normal_drag",normal_drag);
    set("wind/tangential_drag",tangential_drag);
    set("wind/enabled",IndexT(enabled));
}
}
