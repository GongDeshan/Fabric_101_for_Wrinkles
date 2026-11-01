#include <contact_system/vertex_half_plane_frictional_contact_diff_parm.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <muda/ext/eigen/evd.h>
#include <utils/matrix_assembly_utils.h>

namespace uipc::backend::cuda
{
void VertexHalfPlaneFrictionalContactDiffParm::do_build(ContactDiffParmReporter::BuildInfo&)
{
    bool enable = world().scene().info()["contact"]["friction"]["enable"].get<bool>();

    if(!enable)
    {
        throw SimSystemException("Frictional contact is disabled");
    }

    m_impl.global_trajectory_filter    = &require<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager      = &require<GlobalContactManager>();
    m_impl.global_vertex_manager       = &require<GlobalVertexManager>();
    m_impl.global_diff_contact_manager = &require<GlobalDiffContactManager>();

    m_impl.dt = world().scene().info()["dt"].get<Float>();

    BuildInfo info;
    do_build(info);
}

void VertexHalfPlaneFrictionalContactDiffParm::do_report_extent(DiffParmExtentInfo& info)
{
    auto filter = m_impl.veretx_half_plane_trajectory_filter;
    SizeT count = filter->friction_PHs().size();  // pGpmu,pGpkappa -> 2 * Matrix3x1
    info.triplet_count(2 * count);
}

void VertexHalfPlaneFrictionalContactDiffParm::do_assemble(ContactDiffParmReporter::DiffParmInfo& info)
{
    DiffParmInfo this_info{&m_impl, info.pGpP()};
    // let subclass to fill in the data
    do_assemble(this_info);
}

void VertexHalfPlaneFrictionalContactDiffParm::do_init()
{
    // Ensure that VertexHalfPlaneTrajectoryFilter is already registered in GlobalTrajectoryFilter.
    m_impl.veretx_half_plane_trajectory_filter =
        m_impl.global_trajectory_filter->find<VertexHalfPlaneTrajectoryFilter>();

    InitInfo info;
    do_init(info);
}

muda::CBuffer2DView<ContactCoeff> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::contact_tabular() const
{
    return m_impl->global_contact_manager->contact_tabular();
}

muda::CBuffer2DView<DiffContactCoeff> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::diff_contact_tabular() const
{
    return m_impl->global_diff_contact_manager->diff_contact_tabular();
}

muda::CBufferView<Vector2i> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::friction_PHs() const
{
    return m_impl->veretx_half_plane_trajectory_filter->friction_PHs();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::positions() const
{
    return m_impl->global_vertex_manager->positions();
}

muda::CBufferView<Float> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::thicknesses() const
{
    return m_impl->global_vertex_manager->thicknesses();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::prev_positions() const
{
    return m_impl->global_vertex_manager->prev_positions();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::rest_positions() const
{
    return m_impl->global_vertex_manager->rest_positions();
}

muda::CBufferView<IndexT> VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::contact_element_ids() const
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

Float VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::d_hat() const
{
    return m_impl->global_contact_manager->d_hat();
}

Float VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::dt() const
{
    return m_impl->dt;
}

Float VertexHalfPlaneFrictionalContactDiffParm::BaseInfo::eps_velocity() const
{
    return m_impl->global_contact_manager->eps_velocity();
}
}  // namespace uipc::backend::cuda
