#pragma once
#include <uipc/core/feature.h>
#include <uipc/backend/buffer_view.h>

namespace uipc::diff_sim
{
class UIPC_CORE_API UpdateParameterFeatureOverrider
{
  public:
    virtual ~UpdateParameterFeatureOverrider()           = default;
    virtual void do_update(backend::BufferView in_aim_P) = 0;
};

class UIPC_CORE_API UpdateParameterFeature final : public core::Feature
{
  public:
    constexpr static std::string_view FeatureName = "diff_sim/update_parameter";

    UpdateParameterFeature(S<UpdateParameterFeatureOverrider> overrider);

    /**
     * @brief Updates the parameters.
     *
     * Note: No check will be performed in this function, the caller should ensure the input is valid.
     * By using `LearningRateFeature`, the caller can get a valid learning rate to update the parameters.
     * 
     * @param[in] in_aim_P:BufferView<Float> The aim parameters.
     */
    void update(backend::BufferView in_aim_P);

  private:
    virtual std::string_view           get_name() const final override;
    S<UpdateParameterFeatureOverrider> m_impl;
};
}  // namespace uipc::diff_sim
