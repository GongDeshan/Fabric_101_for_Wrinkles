#include <finite_element/fem_contact_diff_parm_receiver.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMContactDiffParmReceiver);

void FEMContactDiffParmReceiver::do_build(BuildInfo& info)
{
    m_impl.vertex_reporter = &require<FiniteElementVertexReporter>();
}

void FEMContactDiffParmReceiver::do_init() {}

void FEMContactDiffParmReceiver::do_report(ClassifyInfo& info)
{
    auto v_offset = m_impl.vertex_reporter->vertex_offset();
    auto v_count  = m_impl.vertex_reporter->vertex_count();
    auto v_begin  = v_offset;
    auto v_end    = v_offset + v_count;
    info.range({v_begin, v_end});
}

void FEMContactDiffParmReceiver::do_receive(ClassifiedDiffParmInfo& info)
{
    m_impl.pGpP = info.pGpP();
}
}  // namespace uipc::backend::cuda