#include <contact_system/vertex_half_plane_frictional_contact_diff_dof.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <muda/ext/eigen/evd.h>
#include <utils/matrix_assembly_utils.h>

namespace uipc::backend::cuda
{
void VertexHalfPlaneFrictionalContactDiffDof::do_build(ContactDiffDofReporter::BuildInfo&)
{
    bool enable = world().scene().info()["contact"]["friction"]["enable"].get<bool>();

    if(!enable)
    {
        throw SimSystemException("Frictional contact is disabled");
    }

    m_impl.global_trajectory_filter = &require<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager   = &require<GlobalContactManager>();
    m_impl.global_vertex_manager    = &require<GlobalVertexManager>();

    m_impl.dt = world().scene().info()["dt"].get<Float>();

    BuildInfo info;
    do_build(info);
}

void VertexHalfPlaneFrictionalContactDiffDof::do_report_extent(DiffDofExtentInfo& info)
{
    auto  filter = m_impl.veretx_half_plane_trajectory_filter;
    SizeT count  = filter->friction_PHs().size();
    info.triplet_count(count);
}

void VertexHalfPlaneFrictionalContactDiffDof::do_assemble(ContactDiffDofReporter::DiffDofInfo& info)
{
    DiffDofInfo this_info{&m_impl, info.H()};
    // let subclass to fill in the data
    do_assemble(this_info);
}

void VertexHalfPlaneFrictionalContactDiffDof::do_init()
{
    // Ensure that VertexHalfPlaneTrajectoryFilter is already registered in GlobalTrajectoryFilter.
    m_impl.veretx_half_plane_trajectory_filter =
        m_impl.global_trajectory_filter->find<VertexHalfPlaneTrajectoryFilter>();

    InitInfo info;
    do_init(info);
}

muda::CBuffer2DView<ContactCoeff> VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::contact_tabular() const
{
    return m_impl->global_contact_manager->contact_tabular();
}

muda::CBufferView<Vector2i> VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::friction_PHs() const
{
    return m_impl->veretx_half_plane_trajectory_filter->friction_PHs();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::positions() const
{
    return m_impl->global_vertex_manager->positions();
}

muda::CBufferView<Float> VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::thicknesses() const
{
    return m_impl->global_vertex_manager->thicknesses();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::prev_positions() const
{
    return m_impl->global_vertex_manager->prev_positions();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::rest_positions() const
{
    return m_impl->global_vertex_manager->rest_positions();
}

muda::CBufferView<IndexT> VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::contact_element_ids() const
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

Float VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::d_hat() const
{
    return m_impl->global_contact_manager->d_hat();
}

Float VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::dt() const
{
    return m_impl->dt;
}

Float VertexHalfPlaneFrictionalContactDiffDof::BaseInfo::eps_velocity() const
{
    return m_impl->global_contact_manager->eps_velocity();
}
}  // namespace uipc::backend::cuda
