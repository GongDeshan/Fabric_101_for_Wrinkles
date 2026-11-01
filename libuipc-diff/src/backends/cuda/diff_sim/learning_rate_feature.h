#pragma once
#include <type_define.h>
#include <uipc/diff_sim/learning_rate_feature.h>
#include <muda/buffer/device_buffer.h>

namespace uipc::backend::cuda
{
class GlobalLearningRateManager;
class LearningRateFeatureOverrider final : public diff_sim::LearningRateFeatureOverrider
{
  public:
    LearningRateFeatureOverrider(GlobalLearningRateManager& lrm);

    class Impl
    {
      public:
        muda::DeviceBuffer<Float> aim_parms;
    };

  private:
    virtual Float get_learning_rate(backend::BufferView in_aim_parms) override;

    GlobalLearningRateManager& m_lrm;
    Impl                       m_impl;
};
}  // namespace uipc::backend::cuda
