#include <uipc/diff_sim/update_parameter_feature.h>

namespace uipc::diff_sim
{
UpdateParameterFeature::UpdateParameterFeature(S<UpdateParameterFeatureOverrider> overrider)
    : m_impl{std::move(overrider)}
{
}

void UpdateParameterFeature::update(backend::BufferView in_aim_P)
{
    m_impl->do_update(in_aim_P);
}

std::string_view UpdateParameterFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::diff_sim
