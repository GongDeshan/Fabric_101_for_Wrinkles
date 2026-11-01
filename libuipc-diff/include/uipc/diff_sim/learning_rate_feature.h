#pragma once
#include <uipc/core/feature.h>
#include <uipc/common/smart_pointer.h>
#include <uipc/backend/buffer_view.h>

namespace uipc::diff_sim
{
class UIPC_CORE_API LearningRateFeatureOverrider
{
  public:
    virtual ~LearningRateFeatureOverrider() = default;

    virtual Float get_learning_rate(backend::BufferView in_aim_parms) = 0;
};

class UIPC_CORE_API LearningRateFeature final : public core::Feature
{
  public:
    static constexpr std::string_view FeatureName = "diff_sim/learning_rate";

    LearningRateFeature(S<LearningRateFeatureOverrider> overrider);

    /**
     * @brief Get a base valid learning rate.
     * 
     * Get a conservative learning rate that guarantees validness:
     * 1. Penetration-Free
     * 2. ... (more to be added)
     * 
     * By giving the aim parameters, the backend can will calculate if we can move to the aim parameters,
     * if can, return 1.0, otherwise, return a smaller value that guarantees the validness.
     * 
     * @param[in] in_aim_parms:BufferView<Float> The parameters to aim at.
     * @return Float The learning rate.
     */
    Float learning_rate(backend::BufferView in_aim_parms);

  private:
    virtual std::string_view        get_name() const override final;
    S<LearningRateFeatureOverrider> m_impl;
};
}  // namespace uipc::diff_sim
