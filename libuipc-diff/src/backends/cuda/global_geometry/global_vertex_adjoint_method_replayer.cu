#include <global_geometry/global_vertex_adjoint_method_replayer.h>
#include <global_geometry/global_vertex_manager.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalVertexAdjointMethodReplayer);

void GlobalVertexAdjointMethodReplayer::do_build(BuildInfo& info)
{
    m_impl.global_vertex_manager = &require<GlobalVertexManager>();

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

void GlobalVertexAdjointMethodReplayer::do_record(RecordInfo& info)
{
    auto& offset_stack = m_impl.vertex_offset_stack;
    auto& count_stack  = m_impl.vertex_count_stack;

    auto F             = info.frame();  // 1,2,...,N
    auto RecordedFrame = offset_stack.size();
    UIPC_ASSERT(F == RecordedFrame,
                "Frame mismatch, Input Frame={}, Last Frame={}, why can it happen?",
                F,
                RecordedFrame);

    auto v_count = m_impl.global_vertex_manager->positions().size();

    offset_stack.push(offset_stack.top() + v_count);  // update the vertex offsets for the next frame
    count_stack.push(v_count);  // record the vertex count for the current frame


    append(m_impl.positions, m_impl.global_vertex_manager->positions());
    append(m_impl.prev_positions, m_impl.global_vertex_manager->prev_positions());
}

void GlobalVertexAdjointMethodReplayer::do_replay(ReplayInfo& info)
{
    auto& offset_stack = m_impl.vertex_offset_stack;
    auto& count_stack  = m_impl.vertex_count_stack;

    auto next_v_offset = offset_stack.top();
    offset_stack.pop();  // pop the last frame

    auto F             = info.frame();
    auto StackTopFrame = offset_stack.size();

    UIPC_ASSERT(F == StackTopFrame,
                "Frame mismatch, Input Frame={}, Stack Top Frame={}, why can it happen?",
                F,
                StackTopFrame);

    auto v_offset = offset_stack.top();  // get offset
    auto v_count  = count_stack.top();   // get count

    UIPC_ASSERT(v_offset + v_count == next_v_offset,
                "Vertex offset mismatch, v_offset={}, v_count={}, next_v_offset={}, why can it happen?",
                v_offset,
                v_count,
                next_v_offset);

    auto& gvm = m_impl.global_vertex_manager->m_impl;

    // copy the recorded data to the global vertex manager
    gvm.positions.view().copy_from(m_impl.positions.view(v_offset, v_count));
    gvm.prev_positions.view().copy_from(m_impl.prev_positions.view(v_offset, v_count));
}

void GlobalVertexAdjointMethodReplayer::do_clear()
{
    auto& offset_stack = m_impl.vertex_offset_stack;
    while(!offset_stack.empty())
        offset_stack.pop();
    offset_stack.push(0);  // dummy frame 1

    auto& count_stack = m_impl.vertex_count_stack;
    while(!count_stack.empty())
        count_stack.pop();

    m_impl.positions.clear();
    m_impl.prev_positions.clear();
}
}  // namespace uipc::backend::cuda
