#include <coupling_system/fem_abd_contact_diff_dof_receiver.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMABDContactDiffDofReceiver);

void FEMABDContactDiffDofReceiver::do_build(BuildInfo& info)
{
    m_impl.fem_vertex_reporter = &require<FiniteElementVertexReporter>();
    m_impl.abd_vertex_reporter = &require<AffineBodyVertexReporter>();
}

void FEMABDContactDiffDofReceiver::do_init() {}

void FEMABDContactDiffDofReceiver::do_report(ClassifyInfo& info)
{
    auto fem_v_offset = m_impl.fem_vertex_reporter->vertex_offset();
    auto fem_v_count  = m_impl.fem_vertex_reporter->vertex_count();
    auto fem_v_begin  = fem_v_offset;
    auto fem_v_end    = fem_v_offset + fem_v_count;

    auto abd_v_offset = m_impl.abd_vertex_reporter->vertex_offset();
    auto abd_v_count  = m_impl.abd_vertex_reporter->vertex_count();
    auto abd_v_begin  = abd_v_offset;
    auto abd_v_end    = abd_v_offset + abd_v_count;

    info.range({fem_v_begin, fem_v_end}, {abd_v_begin, abd_v_end});
}

void FEMABDContactDiffDofReceiver::do_receive(ClassifiedDiffDofInfo& info)
{
    m_impl.H = info.H();
}
}  // namespace uipc::backend::cuda