#include <uipc/diff_sim/learning_rate_feature.h>

namespace uipc::diff_sim
{
LearningRateFeature::LearningRateFeature(S<LearningRateFeatureOverrider> overrider)
    : m_impl(std::move(overrider))
{
    UIPC_ASSERT(m_impl, "LearningRateFeatureOverrider must not be null.");
}

Float LearningRateFeature::learning_rate(backend::BufferView aim_parms)
{
    return m_impl->get_learning_rate(aim_parms);
}

std::string_view LearningRateFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::diff_sim
