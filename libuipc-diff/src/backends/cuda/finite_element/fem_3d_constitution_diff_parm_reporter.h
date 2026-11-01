#pragma once
#include <finite_element/finite_element_constitution_diff_parm_reporter.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::cuda
{
class FEM3DConstitutionDiffParmReporter : public FiniteElementConstitutionDiffParmReporter
{
  public:
    using Base = FiniteElementConstitutionDiffParmReporter;
    using Base::Base;

    class BuildInfo
    {
      public:
    };

    class DiffParmInfo
    {
      public:
        DiffParmInfo(FEM3DConstitutionDiffParmReporter* impl, Base::DiffParmInfo& diff_parm_info)
            : m_impl(impl)
            , m_diff_parm_info(diff_parm_info)
        {
        }

        muda::CBufferView<Vector3>   xs() const noexcept;
        muda::CBufferView<Vector3>   x_bars() const noexcept;
        muda::CBufferView<Float>     rest_volumes() const noexcept;
        muda::CBufferView<Matrix3x3> Dm_invs() const noexcept;
        muda::CBufferView<Vector4i>  indices() const noexcept;
        muda::CBufferView<Float>     thicknesses() const noexcept;
        muda::CBufferView<IndexT>    is_fixed() const noexcept;
        const FiniteElementMethod::ConstitutionInfo& constitution_info() const noexcept;

        SizeT  frame() const;
        IndexT dof_offset(SizeT frame) const;
        IndexT dof_count(SizeT frame) const;

        muda::TripletMatrixView<Float, 3, 1> pGpP() const;
        Float                                dt() const;

      protected:
        FEM3DConstitutionDiffParmReporter* m_impl = nullptr;
        Base::DiffParmInfo&                m_diff_parm_info;
    };

  protected:
    virtual void do_assemble(DiffParmInfo& info) = 0;
    virtual void do_build(BuildInfo& info)       = 0;

  private:
    virtual void do_assemble(Base::DiffParmInfo& info) final override;
    virtual void do_build(Base::BuildInfo& info) final override;
    IndexT       get_dim() const noexcept final override;

    FiniteElementMethod* fem = nullptr;
};
}  // namespace uipc::backend::cuda
