#include <affine_body/abd_adjoint_method_replayer.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDAdjointMethodReplayer);

void ABDAdjointMethodReplayer::do_build(BuildInfo& info)
{
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();

    m_impl.body_offset_stack.push(0);  // dummy frame 1
}

static void append(muda::DeviceBuffer<Vector12>& buffer, muda::CBufferView<Vector12> data)
{
    auto offset = buffer.size();
    auto count  = data.size();
    auto total  = offset + count;
    buffer.resize(total);
    buffer.view(offset, count).copy_from(data);
}

void ABDAdjointMethodReplayer::do_record(RecordInfo& info)
{
    auto& stack = m_impl.body_offset_stack;

    auto F             = info.frame();
    auto RecordedFrame = stack.size();
    UIPC_ASSERT(F == RecordedFrame,
                "Frame mismatch, Input Frame={}, Last Frame={}, why can it happen?",
                F,
                RecordedFrame);

    auto b_count = m_impl.affine_body_dynamics->dof_count(F) / 12;

    stack.push(stack.top() + b_count);  // update the body offsets for the next frame


    append(m_impl.qs, m_impl.affine_body_dynamics->qs());
    append(m_impl.q_tildes, m_impl.affine_body_dynamics->q_tildes());
    append(m_impl.q_vs, m_impl.affine_body_dynamics->q_vs());
    append(m_impl.q_prevs, m_impl.affine_body_dynamics->q_prevs());
}

void ABDAdjointMethodReplayer::do_replay(ReplayInfo& info)
{
    auto& stack         = m_impl.body_offset_stack;
    auto  next_b_offset = stack.top();
    stack.pop();  // pop the last frame

    auto F             = info.frame();
    auto StackTopFrame = stack.size();

    UIPC_ASSERT(F == StackTopFrame,
                "Frame mismatch, Input Frame={}, Stack Top Frame={}, why can it happen?",
                F,
                StackTopFrame);

    auto b_offset = stack.top();  // get offset
    auto b_count  = m_impl.affine_body_dynamics->dof_count(F) / 12;

    auto& abd = m_impl.affine_body_dynamics->m_impl;
    abd.body_id_to_q.view().copy_from(m_impl.qs.view(b_offset, b_count));
    abd.body_id_to_q_tilde.view().copy_from(m_impl.q_tildes.view(b_offset, b_count));
    abd.body_id_to_q_v.view().copy_from(m_impl.q_vs.view(b_offset, b_count));
    abd.body_id_to_q_prev.view().copy_from(m_impl.q_prevs.view(b_offset, b_count));
}

void ABDAdjointMethodReplayer::do_clear()
{
    auto& stack = m_impl.body_offset_stack;
    m_impl.qs.clear();
    m_impl.q_tildes.clear();
    while(!stack.empty())
        stack.pop();
    stack.push(0);  // dummy frame 1
}
}  // namespace uipc::backend::cuda
