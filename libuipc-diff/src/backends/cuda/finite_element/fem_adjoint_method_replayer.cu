#include <finite_element/fem_adjoint_method_replayer.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMAdjointMethodReplayer);

void FEMAdjointMethodReplayer::do_build(BuildInfo& info)
{
    m_impl.finite_element_method = &require<FiniteElementMethod>();

    m_impl.vertex_offset_stack.push(0);  // dummy frame 1
}

static void append(muda::DeviceBuffer<Vector3>& buffer, muda::CBufferView<Vector3> data)
{
    auto offset = buffer.size();
    auto count  = data.size();
    auto total  = offset + count;
    buffer.resize(total);
    buffer.view(offset, count).copy_from(data);
}

void FEMAdjointMethodReplayer::do_record(RecordInfo& info)
{
    auto& stack = m_impl.vertex_offset_stack;

    auto F             = info.frame();  // 1,2,...,N
    auto RecordedFrame = stack.size();
    UIPC_ASSERT(F == RecordedFrame,
                "Frame mismatch, Input Frame={}, Last Frame={}, why can it happen?",
                F,
                RecordedFrame);

    auto v_count = m_impl.finite_element_method->dof_count(F) / 3;

    stack.push(stack.top() + v_count);  // update the vertex offsets for the next frame

    append(m_impl.xs, m_impl.finite_element_method->xs());
    append(m_impl.x_tildes, m_impl.finite_element_method->x_tildes());
}

void FEMAdjointMethodReplayer::do_replay(ReplayInfo& info)
{
    auto& stack         = m_impl.vertex_offset_stack;
    auto  next_v_offset = stack.top();
    stack.pop();  // pop the last frame

    auto F             = info.frame();
    auto StackTopFrame = stack.size();

    UIPC_ASSERT(F == StackTopFrame,
                "Frame mismatch, Input Frame={}, Stack Top Frame={}, why can it happen?",
                F,
                StackTopFrame);

    auto v_offset = stack.top();  // get offset
    auto v_count  = m_impl.finite_element_method->dof_count(F) / 3;

    UIPC_ASSERT(v_offset + v_count == next_v_offset,
                "Vertex offset mismatch, v_offset={}, v_count={}, next_v_offset={}, why can it happen?",
                v_offset,
                v_count,
                next_v_offset);

    auto& fem = m_impl.finite_element_method->m_impl;

    // copy the recorded data to the FEM
    fem.xs.view().copy_from(m_impl.xs.view(v_offset, v_count));
    fem.x_tildes.view().copy_from(m_impl.x_tildes.view(v_offset, v_count));
}

void FEMAdjointMethodReplayer::do_clear()
{
    auto& stack = m_impl.vertex_offset_stack;
    m_impl.xs.clear();
    m_impl.x_tildes.clear();
    while(!stack.empty())
        stack.pop();
    stack.push(0);  // dummy frame 1
}
}  // namespace uipc::backend::cuda
