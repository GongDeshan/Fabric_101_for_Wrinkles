#include <contact_system/simplex_frictional_contact_diff_dof.h>
#include <muda/ext/eigen/evd.h>

namespace uipc::backend::cuda
{
void SimplexFrictionalContactDiffDof::do_build(ContactDiffDofReporter::BuildInfo&)
{
    bool enable = world().scene().info()["contact"]["friction"]["enable"].get<bool>();

    if(!enable)
    {
        throw SimSystemException("Frictional contact is disabled");
    }

    m_impl.global_trajectory_filter = &require<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager   = &require<GlobalContactManager>();
    m_impl.global_vertex_manager    = &require<GlobalVertexManager>();

    BuildInfo info;
    do_build(info);

    m_impl.dt = world().scene().info()["dt"].get<Float>();
}

void SimplexFrictionalContactDiffDof::do_init()
{
    // Ensure that SimplexTrajectoryFilter is already registered in GlobalTrajectoryFilter.
    m_impl.simplex_trajectory_filter =
        m_impl.global_trajectory_filter->find<SimplexTrajectoryFilter>();

    InitInfo info;
    do_init(info);
}

void SimplexFrictionalContactDiffDof::do_report_extent(DiffDofExtentInfo& info)
{
    BaseInfo base_info{&m_impl};

    auto offset = 0;
    {
        auto pt_count = base_info.friction_PTs().size();
        m_impl.PT_H_count = pt_count * 4 * 4;  // 16 Matrix3x3 Blocks for each PT
        m_impl.PT_H_offset = offset;
        offset += m_impl.PT_H_count;
    }

    {
        auto ee_count = base_info.friction_EEs().size();
        m_impl.EE_H_count = ee_count * 4 * 4;  // 16 Matrix3x3 Blocks for each EE
        m_impl.EE_H_offset = offset;
        offset += m_impl.EE_H_count;
    }

    {
        auto pe_count     = base_info.friction_PEs().size();
        m_impl.PE_H_count = pe_count * 3 * 3;  // 9 Matrix3x3 Blocks for each PE
        m_impl.PE_H_offset = offset;
        offset += m_impl.PE_H_count;
    }

    {
        auto pp_count     = base_info.friction_PPs().size();
        m_impl.PP_H_count = pp_count * 2 * 2;  // 4 Matrix3x3 Blocks for each PP
        m_impl.PP_H_offset = offset;
        offset += m_impl.PP_H_count;
    }

    UIPC_ASSERT(offset
                    == m_impl.PT_H_count + m_impl.EE_H_count + m_impl.PE_H_count
                           + m_impl.PP_H_count,
                "offset is not correct");

    auto count = offset;

    info.triplet_count(count);
}

void SimplexFrictionalContactDiffDof::do_assemble(ContactDiffDofReporter::DiffDofInfo& info)
{
    DiffDofInfo contact_diff_dof_info{&m_impl, info.H()};
    // let the subclass do the assembly
    do_assemble(contact_diff_dof_info);
}

muda::CBuffer2DView<ContactCoeff> SimplexFrictionalContactDiffDof::BaseInfo::contact_tabular() const
{
    return m_impl->global_contact_manager->contact_tabular();
}

muda::CBufferView<Vector4i> SimplexFrictionalContactDiffDof::BaseInfo::friction_PTs() const
{
    return m_impl->simplex_trajectory_filter->friction_PTs();
}

muda::CBufferView<Vector4i> SimplexFrictionalContactDiffDof::BaseInfo::friction_EEs() const
{
    return m_impl->simplex_trajectory_filter->friction_EEs();
}

muda::CBufferView<Vector3i> SimplexFrictionalContactDiffDof::BaseInfo::friction_PEs() const
{
    return m_impl->simplex_trajectory_filter->friction_PEs();
}

muda::CBufferView<Vector2i> SimplexFrictionalContactDiffDof::BaseInfo::friction_PPs() const
{
    return m_impl->simplex_trajectory_filter->friction_PPs();
}

muda::CBufferView<Vector3> SimplexFrictionalContactDiffDof::BaseInfo::positions() const
{
    return m_impl->global_vertex_manager->positions();
}

muda::CBufferView<Vector3> SimplexFrictionalContactDiffDof::BaseInfo::prev_positions() const
{
    return m_impl->global_vertex_manager->prev_positions();
}

muda::CBufferView<Vector3> SimplexFrictionalContactDiffDof::BaseInfo::rest_positions() const
{
    return m_impl->global_vertex_manager->rest_positions();
}

muda::CBufferView<Float> SimplexFrictionalContactDiffDof::BaseInfo::thicknesses() const
{
    return m_impl->global_vertex_manager->thicknesses();
}

muda::CBufferView<IndexT> SimplexFrictionalContactDiffDof::BaseInfo::contact_element_ids() const
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

Float SimplexFrictionalContactDiffDof::BaseInfo::d_hat() const
{
    return m_impl->global_contact_manager->d_hat();
}

Float SimplexFrictionalContactDiffDof::BaseInfo::dt() const
{
    return m_impl->dt;
}

Float SimplexFrictionalContactDiffDof::BaseInfo::eps_velocity() const
{
    return m_impl->global_contact_manager->eps_velocity();
}
muda::TripletMatrixView<Float, 3> SimplexFrictionalContactDiffDof::DiffDofInfo::PT_H() const
{
    return m_H.subview(m_impl->PT_H_offset, m_impl->PT_H_count);
}
muda::TripletMatrixView<Float, 3> SimplexFrictionalContactDiffDof::DiffDofInfo::EE_H() const
{
    return m_H.subview(m_impl->EE_H_offset, m_impl->EE_H_count);
}
muda::TripletMatrixView<Float, 3> SimplexFrictionalContactDiffDof::DiffDofInfo::PE_H() const
{
    return m_H.subview(m_impl->PE_H_offset, m_impl->PE_H_count);
}
muda::TripletMatrixView<Float, 3> SimplexFrictionalContactDiffDof::DiffDofInfo::PP_H() const
{
    return m_H.subview(m_impl->PP_H_offset, m_impl->PP_H_count);
}
}  // namespace uipc::backend::cuda
