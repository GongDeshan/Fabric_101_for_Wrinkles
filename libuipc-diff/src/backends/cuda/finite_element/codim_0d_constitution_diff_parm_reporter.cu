#include <finite_element/codim_0d_constitution_diff_parm_reporter.h>

namespace uipc::backend::cuda
{
void Codim0DConstitutionDiffParmReporter::do_assemble(Base::DiffParmInfo& info)
{
    DiffParmInfo this_info(this, info);
    do_assemble(this_info);
}

void Codim0DConstitutionDiffParmReporter::do_build(Base::BuildInfo& info)
{
    fem = &require<FiniteElementMethod>();
    BuildInfo this_info;
    do_build(this_info);
}

IndexT Codim0DConstitutionDiffParmReporter::get_dim() const noexcept
{
    return 0;
}

muda::CBufferView<Vector3> Codim0DConstitutionDiffParmReporter::DiffParmInfo::xs() const noexcept
{
    return m_impl->fem->xs();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Vector3> Codim0DConstitutionDiffParmReporter::DiffParmInfo::x_bars() const noexcept
{
    return m_impl->fem->x_bars();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Float> Codim0DConstitutionDiffParmReporter::DiffParmInfo::thicknesses() const noexcept
{
    return m_impl->fem->thicknesses();
}

muda::CBufferView<IndexT> Codim0DConstitutionDiffParmReporter::DiffParmInfo::is_fixed() const noexcept
{
    return m_impl->fem->is_fixed();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<IndexT> Codim0DConstitutionDiffParmReporter::DiffParmInfo::indices() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem->codim_0ds().subview(info.primitive_offset, info.primitive_count);
}

const FiniteElementMethod::ConstitutionInfo& Codim0DConstitutionDiffParmReporter::DiffParmInfo::constitution_info() const noexcept
{
    return m_impl->constitution_info();
}

SizeT Codim0DConstitutionDiffParmReporter::DiffParmInfo::frame() const
{
    return m_diff_parm_info.frame();
}
IndexT Codim0DConstitutionDiffParmReporter::DiffParmInfo::dof_offset(SizeT frame) const
{
    return m_diff_parm_info.dof_offset(frame);
}
IndexT Codim0DConstitutionDiffParmReporter::DiffParmInfo::dof_count(SizeT frame) const
{
    return m_diff_parm_info.dof_count(frame);
}

muda::TripletMatrixView<Float, 3, 1> Codim0DConstitutionDiffParmReporter::DiffParmInfo::pGpP() const
{
    return m_diff_parm_info.pGpP();
}

Float Codim0DConstitutionDiffParmReporter::DiffParmInfo::dt() const
{
    return m_diff_parm_info.dt();
}
}  // namespace uipc::backend::cuda
