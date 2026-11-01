#include <diff_sim/update_parameter_feature.h>
#include <utils/auto_device_buffer_utils.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <uipc/common/log_pattern_guard.h>

namespace uipc::backend::cuda
{
UpdateParameterFeatureOverrider::UpdateParameterFeatureOverrider(GlobalDiffSimManager& manager)
    : m_manager{manager}
{
}

void UpdateParameterFeatureOverrider::do_update(backend::BufferView in_aim_P)
{
    LogPatternGuard guard{UIPC_BACKEND_NAME};

    UIPC_ASSERT(m_manager.is_valid(),
                "Invalid GlobalDiffSimManager SimSystem, why can it happen?");

    spdlog::info("Update Parameters({})", in_aim_P.size());
    InArgBufferView aim_p{m_impl.aim_P};
    m_manager.update(aim_p(in_aim_P));
}
}  // namespace uipc::backend::cuda
