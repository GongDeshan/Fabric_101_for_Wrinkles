#include <finite_element/codim_2d_constitution_diff_parm_reporter.h>

namespace uipc::backend::cuda
{
void Codim2DConstitutionDiffParmReporter::do_assemble(Base::DiffParmInfo& info)
{
    DiffParmInfo this_info(this, info);
    do_assemble(this_info);
}

void Codim2DConstitutionDiffParmReporter::do_build(Base::BuildInfo& info)
{
    fem = &require<FiniteElementMethod>();
    BuildInfo this_info;
    do_build(this_info);
}

IndexT Codim2DConstitutionDiffParmReporter::get_dim() const noexcept
{
    return 2;
}

muda::CBufferView<Vector3> Codim2DConstitutionDiffParmReporter::DiffParmInfo::xs() const noexcept
{
    return m_impl->fem->xs();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Vector3> Codim2DConstitutionDiffParmReporter::DiffParmInfo::x_bars() const noexcept
{
    return m_impl->fem->x_bars();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::rest_areas() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem->rest_areas().subview(info.primitive_offset, info.primitive_count);
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::thicknesses() const noexcept
{
    return m_impl->fem->thicknesses();
}

muda::CBufferView<IndexT> Codim2DConstitutionDiffParmReporter::DiffParmInfo::is_fixed() const noexcept
{
    return m_impl->fem->is_fixed();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Vector3i> Codim2DConstitutionDiffParmReporter::DiffParmInfo::indices() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem->codim_2ds().subview(info.primitive_offset, info.primitive_count);
}

const FiniteElementMethod::ConstitutionInfo& Codim2DConstitutionDiffParmReporter::DiffParmInfo::constitution_info() const noexcept
{
    return m_impl->constitution_info();
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::strain_uvs() const noexcept
{
    return m_impl->fem->strain_uvs;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::plastic_uus() const noexcept
{
    return m_impl->fem->plastic_strain_uus;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::plastic_vvs() const noexcept
{
    return m_impl->fem->plastic_strain_vvs;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::plastic_uvs() const noexcept
{
    return m_impl->fem->plastic_strain_uvs;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::anchor_uus() const noexcept
{
    return m_impl->fem->anchor_strain_uus;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::anchor_vvs() const noexcept
{
    return m_impl->fem->anchor_strain_vvs;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::anchor_uvs() const noexcept
{
    return m_impl->fem->anchor_strain_uvs;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uu_d_yield_uu() const noexcept
{
    return m_impl->fem->d_plastic_uu_d_yield_uu;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_vv_d_yield_vv() const noexcept
{
    return m_impl->fem->d_plastic_vv_d_yield_vv;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uv_d_yield_uv() const noexcept
{
    return m_impl->fem->d_plastic_uv_d_yield_uv;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_anchor_uu_d_thres_uu() const noexcept
{
    return m_impl->fem->d_anchor_uu_d_thres_uu;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_anchor_vv_d_thres_vv() const noexcept
{
    return m_impl->fem->d_anchor_vv_d_thres_vv;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_anchor_uv_d_thres_uv() const noexcept
{
    return m_impl->fem->d_anchor_uv_d_thres_uv;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uu_d_k11() const noexcept
{
    return m_impl->fem->d_plastic_uu_d_k11;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_vv_d_k22() const noexcept
{
    return m_impl->fem->d_plastic_vv_d_k22;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uu_d_hardening_k11() const noexcept
{
    return m_impl->fem->d_plastic_uu_d_hardening_k11;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_vv_d_hardening_k22() const noexcept
{
    return m_impl->fem->d_plastic_vv_d_hardening_k22;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uu_d_yield_uu_ori() const noexcept
{
    return m_impl->fem->d_plastic_uu_d_yield_uu_ori;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_vv_d_yield_vv_ori() const noexcept
{
    return m_impl->fem->d_plastic_vv_d_yield_vv_ori;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uv_d_k33() const noexcept
{
    return m_impl->fem->d_plastic_uv_d_k33;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uv_d_hardening_k33() const noexcept
{
    return m_impl->fem->d_plastic_uv_d_hardening_k33;
}

muda::CBufferView<Float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::d_plastic_uv_d_yield_ori() const noexcept
{
    return m_impl->fem->d_plastic_uv_d_yield_ori;
}

SizeT Codim2DConstitutionDiffParmReporter::DiffParmInfo::frame() const
{
    return m_diff_parm_info.frame();
}
IndexT Codim2DConstitutionDiffParmReporter::DiffParmInfo::dof_offset(SizeT frame) const
{
    return m_diff_parm_info.dof_offset(frame);
}
IndexT Codim2DConstitutionDiffParmReporter::DiffParmInfo::dof_count(SizeT frame) const
{
    return m_diff_parm_info.dof_count(frame);
}

muda::TripletMatrixView<Float, 3, 1> Codim2DConstitutionDiffParmReporter::DiffParmInfo::pGpP() const
{
    return m_diff_parm_info.pGpP();
}

Float Codim2DConstitutionDiffParmReporter::DiffParmInfo::dt() const
{
    return m_diff_parm_info.dt();
}
}  // namespace uipc::backend::cuda
