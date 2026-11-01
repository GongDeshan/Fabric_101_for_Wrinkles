#include <contact_system/simplex_frictional_contact_diff_parm.h>
#include <muda/ext/eigen/evd.h>

namespace uipc::backend::cuda
{
void SimplexFrictionalContactDiffParm::do_build(ContactDiffParmReporter::BuildInfo&)
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

    BuildInfo info;
    do_build(info);

    m_impl.dt = world().scene().info()["dt"].get<Float>();
}

void SimplexFrictionalContactDiffParm::do_init()
{
    // Ensure that SimplexTrajectoryFilter is already registered in GlobalTrajectoryFilter.
    m_impl.simplex_trajectory_filter =
        m_impl.global_trajectory_filter->find<SimplexTrajectoryFilter>();

    InitInfo info;
    do_init(info);
}

void SimplexFrictionalContactDiffParm::do_report_extent(DiffParmExtentInfo& info)
{
    BaseInfo base_info{&m_impl};

    auto offset = 0;
    {
        auto pt_count = base_info.friction_PTs().size();
        m_impl.PT_pGpP_count = pt_count * 4 * 2;  // 4 * Matrix3x1 * 2 (kappa, mu)
        m_impl.PT_pGpP_offset = offset;
        offset += m_impl.PT_pGpP_count;
    }

    {
        auto ee_count = base_info.friction_EEs().size();
        m_impl.EE_pGpP_count = ee_count * 4 * 2;  // 4 * Matrix3x1 * 2 (kappa, mu)
        m_impl.EE_pGpP_offset = offset;
        offset += m_impl.EE_pGpP_count;
    }

    {
        auto pe_count = base_info.friction_PEs().size();
        m_impl.PE_pGpP_count = pe_count * 3 * 2;  // 3 * Matrix3x1 * 2 (kappa, mu)
        m_impl.PE_pGpP_offset = offset;
        offset += m_impl.PE_pGpP_count;
    }

    {
        auto pp_count = base_info.friction_PPs().size();
        m_impl.PP_pGpP_count = pp_count * 2 * 2;  // 2 * Matrix3x1 * 2 (kappa, mu)
        m_impl.PP_pGpP_offset = offset;
        offset += m_impl.PP_pGpP_count;
    }

    UIPC_ASSERT(offset
                    == m_impl.PT_pGpP_count + m_impl.EE_pGpP_count
                           + m_impl.PE_pGpP_count + m_impl.PP_pGpP_count,
                "offset is not correct");

    auto count = offset;

    info.triplet_count(count);
}

void SimplexFrictionalContactDiffParm::do_assemble(ContactDiffParmReporter::DiffParmInfo& info)
{
    DiffParmInfo contact_diff_dof_info{&m_impl, info.pGpP()};
    // let the subclass do the assembly
    do_assemble(contact_diff_dof_info);
}

muda::CBuffer2DView<ContactCoeff> SimplexFrictionalContactDiffParm::BaseInfo::contact_tabular() const
{
    return m_impl->global_contact_manager->contact_tabular();
}

muda::CBufferView<Vector4i> SimplexFrictionalContactDiffParm::BaseInfo::friction_PTs() const
{
    return m_impl->simplex_trajectory_filter->friction_PTs();
}

muda::CBufferView<Vector4i> SimplexFrictionalContactDiffParm::BaseInfo::friction_EEs() const
{
    return m_impl->simplex_trajectory_filter->friction_EEs();
}

muda::CBufferView<Vector3i> SimplexFrictionalContactDiffParm::BaseInfo::friction_PEs() const
{
    return m_impl->simplex_trajectory_filter->friction_PEs();
}

muda::CBufferView<Vector2i> SimplexFrictionalContactDiffParm::BaseInfo::friction_PPs() const
{
    return m_impl->simplex_trajectory_filter->friction_PPs();
}

muda::CBufferView<Vector3> SimplexFrictionalContactDiffParm::BaseInfo::positions() const
{
    return m_impl->global_vertex_manager->positions();
}

muda::CBufferView<Vector3> SimplexFrictionalContactDiffParm::BaseInfo::prev_positions() const
{
    return m_impl->global_vertex_manager->prev_positions();
}

muda::CBufferView<Vector3> SimplexFrictionalContactDiffParm::BaseInfo::rest_positions() const
{
    return m_impl->global_vertex_manager->rest_positions();
}

muda::CBufferView<Float> SimplexFrictionalContactDiffParm::BaseInfo::thicknesses() const
{
    return m_impl->global_vertex_manager->thicknesses();
}

muda::CBufferView<IndexT> SimplexFrictionalContactDiffParm::BaseInfo::contact_element_ids() const
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

Float SimplexFrictionalContactDiffParm::BaseInfo::d_hat() const
{
    return m_impl->global_contact_manager->d_hat();
}

Float SimplexFrictionalContactDiffParm::BaseInfo::dt() const
{
    return m_impl->dt;
}

Float SimplexFrictionalContactDiffParm::BaseInfo::eps_velocity() const
{
    return m_impl->global_contact_manager->eps_velocity();
}

muda::CBuffer2DView<DiffContactCoeff> SimplexFrictionalContactDiffParm::BaseInfo::diff_contact_tabular() const
{
    return m_impl->global_diff_contact_manager->diff_contact_tabular();
}

muda::TripletMatrixView<Float, 3, 1> SimplexFrictionalContactDiffParm::DiffParmInfo::PT_pGpP() const
{
    return m_pGpP.subview(m_impl->PT_pGpP_offset, m_impl->PT_pGpP_count);
}
muda::TripletMatrixView<Float, 3, 1> SimplexFrictionalContactDiffParm::DiffParmInfo::EE_pGpP() const
{
    return m_pGpP.subview(m_impl->EE_pGpP_offset, m_impl->EE_pGpP_count);
}
muda::TripletMatrixView<Float, 3, 1> SimplexFrictionalContactDiffParm::DiffParmInfo::PE_pGpP() const
{
    return m_pGpP.subview(m_impl->PE_pGpP_offset, m_impl->PE_pGpP_count);
}
muda::TripletMatrixView<Float, 3, 1> SimplexFrictionalContactDiffParm::DiffParmInfo::PP_pGpP() const
{
    return m_pGpP.subview(m_impl->PP_pGpP_offset, m_impl->PP_pGpP_count);
}
}  // namespace uipc::backend::cuda
