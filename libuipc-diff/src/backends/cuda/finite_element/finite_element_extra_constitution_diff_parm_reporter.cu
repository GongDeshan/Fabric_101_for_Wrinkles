#include <finite_element/finite_element_extra_constitution_diff_parm_reporter.h>

namespace uipc::backend::cuda
{
U64 FiniteElementExtraConstitutionDiffParmReporter::uid() const noexcept
{
    return get_uid();
}
FiniteElementExtraConstitution& FiniteElementExtraConstitutionDiffParmReporter::constitution() noexcept
{
    UIPC_ASSERT(m_constitution != nullptr,
                "Extra Constitution ({}) is not connected, why?",
                uid());
    return *m_constitution;
}

void FiniteElementExtraConstitutionDiffParmReporter::do_build(FiniteElementDiffParmReporter::BuildInfo& info)
{
    m_fem = &require<FiniteElementMethod>();

    auto uids = world().scene().constitution_tabular().uids();

    auto this_uid = uid();

    auto find = std::ranges::binary_search(uids, this_uid);
    if(!find)
    {
        throw SimSystemException(fmt::format("UID {} not found in the constitution tabular, shutdown diff parm.",
                                             this_uid));
    }

    m_fem->add_reporter(this);

    BuildInfo this_info;
    do_build(this_info);
}

void FiniteElementExtraConstitutionDiffParmReporter::connect()
{
    auto it = fem().extra_constitution_uid_to_index.find(uid());
    UIPC_ASSERT(it != fem().extra_constitution_uid_to_index.end(),
                "Extra Constitution: {} ({}) is not connected, why?",
                name(),
                uid());

    auto c         = fem().extra_constitutions.view()[it->second];
    m_constitution = c;  // connect
    UIPC_ASSERT(m_constitution != nullptr, "Extra Constitution ({}) is nullptr, why?", uid());
}

void FiniteElementExtraConstitutionDiffParmReporter::do_init()
{
    FilteredInfo info{&m_constitution->m_impl};
    do_init(info);
}

void FiniteElementExtraConstitutionDiffParmReporter::do_assemble(Base::DiffParmInfo& info)
{
    DiffParmInfo this_info{this, info};
    do_assemble(this_info);
}

muda::CBufferView<Vector3> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::xs() const noexcept
{
    return m_impl->m_fem->xs();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Vector3> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::x_bars() const noexcept
{
    return m_impl->m_fem->x_bars();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::thicknesses() const noexcept
{
    return m_impl->m_fem->thicknesses();
}

muda::CBufferView<IndexT> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::is_fixed() const noexcept
{
    return m_impl->m_fem->is_fixed();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::plastic_thetas() const noexcept
{
    return m_impl->m_fem->plastic_thetas;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::anchor_thetas() const noexcept
{
    return m_impl->m_fem->anchor_thetas;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_anchor_thetas_d_thres_theta() const noexcept
{
    return m_impl->m_fem->d_anchor_theta_d_thres_theta;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_yield_ori() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_yield_ori;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_k() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_k;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_hardening_k() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_hardening_k;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::plastic_thetas_warp() const noexcept
{
    return m_impl->m_fem->plastic_thetas_warp;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::plastic_thetas_weft() const noexcept
{
    return m_impl->m_fem->plastic_thetas_weft;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::anchor_thetas_warp() const noexcept
{
    return m_impl->m_fem->anchor_thetas_warp;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::anchor_thetas_weft() const noexcept
{
    return m_impl->m_fem->anchor_thetas_weft;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_anchor_thetas_d_thres_theta_warp() const noexcept
{
    return m_impl->m_fem->d_anchor_theta_d_thres_theta_warp;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_anchor_thetas_d_thres_theta_weft() const noexcept
{
    return m_impl->m_fem->d_anchor_theta_d_thres_theta_weft;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_yield_ori_warp() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_yield_ori_warp;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_yield_ori_weft() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_yield_ori_weft;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_k_warp() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_k_warp;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_k_weft() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_k_weft;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_hardening_k_warp() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_hardening_k_warp;
}

muda::CBufferView<Float> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::d_plastic_theta_d_hardening_k_weft() const noexcept
{
    return m_impl->m_fem->d_plastic_theta_d_hardening_k_weft;
}

SizeT FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::frame() const
{
    return m_diff_parm_info.frame();
}
IndexT FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::dof_offset(SizeT frame) const
{
    return m_diff_parm_info.dof_offset(frame);
}
IndexT FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::dof_count(SizeT frame) const
{
    return m_diff_parm_info.dof_count(frame);
}

muda::TripletMatrixView<Float, 3, 1> FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::pGpP() const
{
    return m_diff_parm_info.pGpP();
}

Float FiniteElementExtraConstitutionDiffParmReporter::DiffParmInfo::dt() const
{
    return m_diff_parm_info.dt();
}
}  // namespace uipc::backend::cuda
