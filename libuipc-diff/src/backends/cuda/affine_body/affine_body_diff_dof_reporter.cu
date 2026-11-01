#include <affine_body/affine_body_diff_dof_reporter.h>

namespace uipc::backend::cuda
{
void AffineBodyDiffDofReporter::do_build(DiffDofReporter::BuildInfo& info)
{
    m_impl.abd = &require<AffineBodyDynamics>();
    m_impl.dt  = world().scene().info()["dt"].get<Float>();
    BuildInfo this_info;
    do_build(this_info);
}

void AffineBodyDiffDofReporter::do_assemble(GlobalDiffSimManager::DiffDofInfo& info)
{
    DiffDofInfo this_info{&m_impl, info, m_impl.dt};
    do_assemble(this_info);
}

SizeT AffineBodyDiffDofReporter::DiffDofInfo::frame() const
{
    return m_global_info.frame();
}

IndexT AffineBodyDiffDofReporter::DiffDofInfo::dof_offset(SizeT frame) const
{
    // Frame Dof Offset + ABD Dof Offset => Frame ABD Dof Offset
    return m_global_info.dof_offset(frame) + m_impl->abd->dof_offset(frame);
}

IndexT AffineBodyDiffDofReporter::DiffDofInfo::dof_count(SizeT frame) const
{
    // ABD Dof Count => Frame ABD Dof Count
    return m_impl->abd->dof_count(frame);
}

muda::TripletMatrixView<Float, 3, 3> AffineBodyDiffDofReporter::DiffDofInfo::H() const
{
    return m_global_info.H();
}

Float AffineBodyDiffDofReporter::DiffDofInfo::dt() const
{
    return m_dt;
}
}  // namespace uipc::backend::cuda