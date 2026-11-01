#pragma once
#include <type_define.h>
#include <uipc/diff_sim/update_parameter_feature.h>
#include <muda/buffer/device_buffer.h>

namespace uipc::backend::cuda
{
class GlobalDiffSimManager;

class UpdateParameterFeatureOverrider final : public diff_sim::UpdateParameterFeatureOverrider
{
  public:
    UpdateParameterFeatureOverrider(GlobalDiffSimManager& manager);

    class Impl
    {
      public:
        muda::DeviceBuffer<Float> aim_P;
    };

  private:
    virtual void do_update(backend::BufferView in_aim_P) override;

    GlobalDiffSimManager& m_manager;
    Impl                  m_impl;
};
}  // namespace uipc::backend::cuda
