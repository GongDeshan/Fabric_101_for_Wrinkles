#include <coupling_system/abd_fem_contact_diff_dof_receiver.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDFEMContactDiffDofReceiver);

void ABDFEMContactDiffDofReceiver::do_build(BuildInfo& info)
{
    m_impl.abd_vertex_reporter = &require<AffineBodyVertexReporter>();
    m_impl.fem_vertex_reporter = &require<FiniteElementVertexReporter>();
}

void ABDFEMContactDiffDofReceiver::do_init() {}

void ABDFEMContactDiffDofReceiver::do_report(ClassifyInfo& info)
{
    auto abd_v_offset = m_impl.abd_vertex_reporter->vertex_offset();
    auto abd_v_count  = m_impl.abd_vertex_reporter->vertex_count();
    auto abd_v_begin  = abd_v_offset;
    auto abd_v_end    = abd_v_offset + abd_v_count;

    auto fem_v_offset = m_impl.fem_vertex_reporter->vertex_offset();
    auto fem_v_count  = m_impl.fem_vertex_reporter->vertex_count();
    auto fem_v_begin  = fem_v_offset;
    auto fem_v_end    = fem_v_offset + fem_v_count;

    info.range({abd_v_begin, abd_v_end}, {fem_v_begin, fem_v_end});
}

void ABDFEMContactDiffDofReceiver::do_receive(ClassifiedDiffDofInfo& info)
{
    m_impl.H = info.H();
}
}  // namespace uipc::backend::cuda