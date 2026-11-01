#include <diff_sim/enable_grad_feature.h>

namespace uipc::backend::cuda
{
void EnableGradFeatureOverrider::do_no_grad()
{
    m_grad_enabled = false;
}

void EnableGradFeatureOverrider::do_with_grad()
{
    m_grad_enabled = true;
}

bool EnableGradFeatureOverrider::get_is_grad_enabled() const
{
    return m_grad_enabled;
}
}  // namespace uipc::backend::cuda
