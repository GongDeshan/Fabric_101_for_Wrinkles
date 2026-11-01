#include <diff_sim/adjoint_method_feature.h>
#include <diff_sim/adjoint_method.h>
#include <uipc/builtin/backend_name.h>
#include <utils/auto_device_buffer_utils.h>
#include <uipc/common/log_pattern_guard.h>

namespace uipc::backend::cuda
{
AdjointMethodFeatureOverrider::AdjointMethodFeatureOverrider(AdjointMethod& am)
    : m_am{am}
{
}

void AdjointMethodFeatureOverrider::do_select_dofs(SizeT frame, backend::BufferView in_SDI)
{
    LogPatternGuard guard{UIPC_BACKEND_NAME};

    UIPC_ASSERT(m_am.is_valid(), "Invalid AdjointMethod SimSystem, why can it happen?");
    InArgBufferView SDI{m_impl.local_SDI};
    m_am.select_dofs(frame, SDI(in_SDI));
}

void AdjointMethodFeatureOverrider::do_receive_dofs(backend::BufferView out_dofs)
{
    LogPatternGuard guard{UIPC_BACKEND_NAME};

    UIPC_ASSERT(m_am.is_valid(), "Invalid AdjointMethod SimSystem, why can it happen?");
    OutArgBufferView dofs{m_impl.dofs};
    m_am.receive_dofs(dofs(out_dofs));

    // ~OutArgBufferView: will copy the data back to the host if necessary
}

void AdjointMethodFeatureOverrider::do_compute_dLdP(backend::BufferView out_dLdP,
                                                    backend::BufferView in_dLdX)
{
    LogPatternGuard guard{UIPC_BACKEND_NAME};

    UIPC_ASSERT(m_am.is_valid(), "Invalid AdjointMethod SimSystem, why can it happen?");
    OutArgBufferView dLdP{m_impl.dLdP};
    InArgBufferView  dLdX{m_impl.dLdX};
    m_am.compute_dLdP(dLdP(out_dLdP), dLdX(in_dLdX));

    // ~OutArgBufferView: will copy the data back to the host if necessary
}
}  // namespace uipc::backend::cuda
