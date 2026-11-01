#pragma once
#include <type_define.h>
#include <uipc/diff_sim/enable_grad_feature.h>

namespace uipc::backend::cuda
{
class EnableGradFeatureOverrider : public diff_sim::EnableGradFeatureOverrider
{
  public:
    EnableGradFeatureOverrider() = default;

  private:
    virtual void do_no_grad() override;
    virtual void do_with_grad() override;
    virtual bool get_is_grad_enabled() const override;
    bool         m_grad_enabled = false;
};
}  // namespace uipc::backend::cuda
