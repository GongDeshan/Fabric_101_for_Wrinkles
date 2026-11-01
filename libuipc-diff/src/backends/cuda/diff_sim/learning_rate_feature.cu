#include <diff_sim/learning_rate_feature.h>
#include <diff_sim/global_learning_rate_manager.h>
#include <utils/auto_device_buffer_utils.h>
#include <uipc/common/log_pattern_guard.h>

namespace uipc::backend::cuda
{
LearningRateFeatureOverrider::LearningRateFeatureOverrider(GlobalLearningRateManager& lrm)
    : m_lrm{lrm}
{
}

Float LearningRateFeatureOverrider::get_learning_rate(backend::BufferView in_aim_parms)
{
    LogPatternGuard guard{UIPC_BACKEND_NAME};

    UIPC_ASSERT(m_lrm.is_valid(), "Invalid GlobalLearningRateManager SimSystem, why can it happen?");

    spdlog::info(">>> Begin Computing Learning Rate");
    InArgBufferView aim_parms{m_impl.aim_parms};
    Float           alpha = m_lrm.learning_rate(aim_parms(in_aim_parms));
    spdlog::info("<<< End Computing Learning Rate: {}", alpha);
    return alpha;
}
}  // namespace uipc::backend::cuda
