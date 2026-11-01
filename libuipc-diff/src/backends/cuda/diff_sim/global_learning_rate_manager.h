#pragma once
#include <sim_system.h>
#include <muda/buffer/device_buffer.h>

namespace uipc::backend::cuda
{
class GlobalDiffSimManager;
class LearningRateReporter;

class GlobalLearningRateManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class BaseParameterInfo
    {
      public:
        BaseParameterInfo(Impl* impl, muda::CBufferView<Float> aim_parms)
            : m_impl{impl}
            , m_aim_parms{aim_parms}
        {
        }

        muda::CBufferView<Float> aim_parameters() const;
        muda::CBufferView<Float> safe_parameters() const;

      private:
        friend class GlobalLearningRateManager;
        muda::CBufferView<Float> m_aim_parms;
        Impl*                    m_impl = nullptr;
    };

    class LearningRateInfo : public BaseParameterInfo
    {
      public:
        using BaseParameterInfo::BaseParameterInfo;

        void learning_rate(Float rate);

      private:
        friend class GlobalLearningRateManager;
        Float m_learning_rate = 0.0;
    };

    class RecordSafeStateInfo : public BaseParameterInfo
    {
      public:
        using BaseParameterInfo::BaseParameterInfo;
    };

    class Impl
    {
      public:
        GlobalDiffSimManager*     diff_sim_manager = nullptr;
        muda::DeviceBuffer<Float> safe_parameters;
        SimSystemSlotCollection<LearningRateReporter> learning_rate_reporters;
    };

  private:
    virtual void do_build() override;

    friend class LearningRateFeatureOverrider;
    Float learning_rate(muda::CBufferView<Float> aim_parms);  // only called by LearningRateFeatureOverrider

    friend class LearningRateReporter;
    void add_reporter(LearningRateReporter* reporter);  // only called by LearningRateReporter

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
