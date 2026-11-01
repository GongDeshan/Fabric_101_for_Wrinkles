#include <affine_body/abd_contact_diff_dof_receiver.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDContactDiffDofReceiver);

void ABDContactDiffDofReceiver::do_build(BuildInfo& info)
{
    m_impl.vertex_reporter = &require<AffineBodyVertexReporter>();
}

void ABDContactDiffDofReceiver::do_init() {}

void ABDContactDiffDofReceiver::do_report(ClassifyInfo& info)
{
    auto v_offset = m_impl.vertex_reporter->vertex_offset();
    auto v_count  = m_impl.vertex_reporter->vertex_count();
    auto v_begin  = v_offset;
    auto v_end    = v_offset + v_count;
    info.range({v_begin, v_end});
}

void ABDContactDiffDofReceiver::do_receive(ClassifiedDiffDofInfo& info)
{
    m_impl.H = info.H();
}
}  // namespace uipc::backend::cuda