#include <affine_body/abd_contact_diff_parm_receiver.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDContactDiffParmReceiver);

void ABDContactDiffParmReceiver::do_build(BuildInfo& info)
{
    m_impl.vertex_reporter = &require<AffineBodyVertexReporter>();
}

void ABDContactDiffParmReceiver::do_init() {}

void ABDContactDiffParmReceiver::do_report(ClassifyInfo& info)
{
    auto v_offset = m_impl.vertex_reporter->vertex_offset();
    auto v_count  = m_impl.vertex_reporter->vertex_count();
    auto v_begin  = v_offset;
    auto v_end    = v_offset + v_count;
    info.range({v_begin, v_end});
}

void ABDContactDiffParmReceiver::do_receive(ClassifiedDiffParmInfo& info)
{
    m_impl.pGpP = info.pGpP();
}
}  // namespace uipc::backend::cuda