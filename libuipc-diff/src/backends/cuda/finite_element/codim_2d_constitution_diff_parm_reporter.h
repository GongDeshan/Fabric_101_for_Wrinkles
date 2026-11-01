#pragma once
#include <finite_element/finite_element_constitution_diff_parm_reporter.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::cuda
{
class Codim2DConstitutionDiffParmReporter : public FiniteElementConstitutionDiffParmReporter
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
        DiffParmInfo(Codim2DConstitutionDiffParmReporter* impl, Base::DiffParmInfo& diff_parm_info)
            : m_impl(impl)
            , m_diff_parm_info(diff_parm_info)
        {
        }

        muda::CBufferView<Vector3>  xs() const noexcept;
        muda::CBufferView<Vector3>  x_bars() const noexcept;
        muda::CBufferView<Float>    rest_areas() const noexcept;
        muda::CBufferView<Vector3i> indices() const noexcept;
        muda::CBufferView<Float>    thicknesses() const noexcept;
        muda::CBufferView<IndexT>   is_fixed() const noexcept;
        const FiniteElementMethod::ConstitutionInfo& constitution_info() const noexcept;

        // Report Plastic Parameters
        muda::CBufferView<Float> strain_uvs() const noexcept;

        muda::CBufferView<Float> plastic_uus() const noexcept;
        muda::CBufferView<Float> plastic_vvs() const noexcept;
        muda::CBufferView<Float> plastic_uvs() const noexcept;

        muda::CBufferView<Float> anchor_uus() const noexcept;
        muda::CBufferView<Float> anchor_vvs() const noexcept;
        muda::CBufferView<Float> anchor_uvs() const noexcept;

        muda::CBufferView<Float> d_plastic_uu_d_yield_uu() const noexcept;
        muda::CBufferView<Float> d_plastic_vv_d_yield_vv() const noexcept;
        muda::CBufferView<Float> d_plastic_uv_d_yield_uv() const noexcept;

        muda::CBufferView<Float> d_anchor_uu_d_thres_uu() const noexcept;
        muda::CBufferView<Float> d_anchor_vv_d_thres_vv() const noexcept;
        muda::CBufferView<Float> d_anchor_uv_d_thres_uv() const noexcept;

        muda::CBufferView<Float> d_plastic_uu_d_k11() const noexcept;
        muda::CBufferView<Float> d_plastic_vv_d_k22() const noexcept;
        muda::CBufferView<Float> d_plastic_uu_d_hardening_k11() const noexcept;
        muda::CBufferView<Float> d_plastic_vv_d_hardening_k22() const noexcept;
        muda::CBufferView<Float> d_plastic_uu_d_yield_uu_ori() const noexcept;
        muda::CBufferView<Float> d_plastic_vv_d_yield_vv_ori() const noexcept;


        muda::CBufferView<Float> d_plastic_uv_d_k33() const noexcept;
        muda::CBufferView<Float> d_plastic_uv_d_hardening_k33() const noexcept;
        muda::CBufferView<Float> d_plastic_uv_d_yield_ori() const noexcept;

        SizeT  frame() const;
        IndexT dof_offset(SizeT frame) const;
        IndexT dof_count(SizeT frame) const;

        muda::TripletMatrixView<Float, 3, 1> pGpP() const;
        Float                                dt() const;

      protected:
        Codim2DConstitutionDiffParmReporter* m_impl = nullptr;
        Base::DiffParmInfo&                  m_diff_parm_info;
    };

  protected:
    virtual void do_assemble(DiffParmInfo& info) = 0;
    virtual void do_build(BuildInfo& info)       = 0;

  private:
    virtual void do_assemble(Base::DiffParmInfo& info) final override;
    virtual void do_build(Base::BuildInfo& info) final override;

    IndexT get_dim() const noexcept final override;

    FiniteElementMethod* fem = nullptr;
};
}  // namespace uipc::backend::cuda
