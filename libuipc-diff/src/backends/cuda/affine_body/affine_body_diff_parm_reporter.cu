#include <affine_body/affine_body_diff_parm_reporter.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::cuda
{
void AffineBodyDiffParmReporter::init()
{
    do_init();
}

void AffineBodyDiffParmReporter::do_build(DiffParmReporter::BuildInfo& info)
{
    m_impl.abd                  = &require<AffineBodyDynamics>();
    m_impl.dt                   = world().scene().info()["dt"].get<Float>();
    m_impl.abd_diff_sim_manager = &require<ABDDiffSimManager>();

    BuildInfo this_info;
    do_build(this_info);
}

void AffineBodyDiffParmReporter::do_assemble(GlobalDiffSimManager::DiffParmInfo& info)
{
    DiffParmInfo this_info{&m_impl, info, m_impl.dt};
    do_assemble(this_info);
}

void AffineBodyDiffParmReporter::do_report_extent(GlobalDiffSimManager::DiffParmExtentInfo& info)
{
    DiffParmExtentInfo this_info{&m_impl, info, m_impl.dt};
    do_report_extent(this_info);
}

SizeT AffineBodyDiffParmReporter::BaseInfo::frame() const
{
    return m_global_info.frame();
}

IndexT AffineBodyDiffParmReporter::BaseInfo::dof_offset(SizeT frame) const
{
    // Frame Dof Offset + ABD Dof Offset => Frame ABD Dof Offset
    return m_global_info.dof_offset(frame) + m_impl->abd->dof_offset(frame);
}

IndexT AffineBodyDiffParmReporter::BaseInfo::dof_count(SizeT frame) const
{
    // FEM ABD Count => Frame ABD Dof Count
    return m_impl->abd->dof_count(frame);
}

Float AffineBodyDiffParmReporter::BaseInfo::dt() const
{
    return m_impl->dt;
}

muda::CBufferView<ABDJacobi> AffineBodyDiffParmReporter::BaseInfo::Js() const
{
    return m_impl->abd->Js();
}

muda::CBufferView<IndexT> AffineBodyDiffParmReporter::BaseInfo::v2b() const
{
    return m_impl->abd->v2b();
}

muda::CBufferView<Vector12> AffineBodyDiffParmReporter::BaseInfo::qs() const
{
    return m_impl->abd->qs();
}

muda::CBufferView<Vector12> AffineBodyDiffParmReporter::BaseInfo::q_tildes() const
{
    return m_impl->abd->q_tildes();
}

muda::CBufferView<Vector12> AffineBodyDiffParmReporter::BaseInfo::q_prevs() const
{
    return m_impl->abd->q_prevs();
}

muda::CBufferView<Vector12> AffineBodyDiffParmReporter::BaseInfo::q_vs() const
{
    return m_impl->abd->q_vs();
}

muda::CBufferView<IndexT> AffineBodyDiffParmReporter::BaseInfo::body_is_fixed() const
{
    return m_impl->abd->body_is_fixed();
}

muda::CBufferView<IndexT> AffineBodyDiffParmReporter::BaseInfo::body_is_dynamic() const
{
    return m_impl->abd->body_is_dynamic();
}

muda::CBufferView<ABDJacobiDyadicMass> AffineBodyDiffParmReporter::BaseInfo::body_masses() const
{
    return m_impl->abd->body_masses();
}

muda::CBufferView<Matrix12x12> AffineBodyDiffParmReporter::BaseInfo::body_mass_invs() const
{
    return m_impl->abd->body_mass_invs();
}

muda::CBufferView<Vector12> AffineBodyDiffParmReporter::BaseInfo::body_gravities() const
{
    return m_impl->abd->body_gravities();
}

muda::CBufferView<Vector3i> AffineBodyDiffParmReporter::BaseInfo::diff_triangles() const
{
    return m_impl->abd_diff_sim_manager->diff_triangles();
}

muda::CBufferView<Vector4i> AffineBodyDiffParmReporter::BaseInfo::diff_tetrahedra() const
{
    return m_impl->abd_diff_sim_manager->diff_tetrahedra();
}

muda::CBufferView<IndexT> AffineBodyDiffParmReporter::BaseInfo::diff_vertices() const
{
    return m_impl->abd_diff_sim_manager->diff_vertices();
}

muda::CBufferView<IndexT> AffineBodyDiffParmReporter::BaseInfo::diff_bodies() const
{
    return m_impl->abd_diff_sim_manager->diff_bodies();
}

muda::CBufferView<Float> AffineBodyDiffParmReporter::BaseInfo::body_mass_density() const
{
    return m_impl->abd_diff_sim_manager->body_mass_density();
}

muda::CBufferView<Float> AffineBodyDiffParmReporter::BaseInfo::body_volumes() const
{
    return m_impl->abd->body_volumes();
}

muda::CBufferView<Vector3i> AffineBodyDiffParmReporter::BaseInfo::diff_rest_positions() const
{
    return m_impl->abd_diff_sim_manager->diff_rest_positions();
}

muda::TripletMatrixView<Float, 3, 1> AffineBodyDiffParmReporter::DiffParmInfo::pGpP() const
{
    IndexT IF = IndexT{frame()};

    auto pGpP = m_info.pGpP();

    auto row_offset = dof_offset(IF) / 3;
    auto col_offset = 0;

    auto row_count = dof_count(IF) / 3;
    auto col_count = pGpP.extent().y;

    return pGpP.submatrix({row_offset, col_offset}, {row_count, col_count});
}

void AffineBodyDiffParmReporter::DiffParmExtentInfo::triplet_count(SizeT N)
{
    m_info.triplet_count(N);
}
}  // namespace uipc::backend::cuda
