#pragma once
#include <finite_element/finite_element_diff_parm_reporter.h>
#include <finite_element/finite_element_extra_constitution.h>

namespace uipc::backend::cuda
{
class FiniteElementExtraConstitutionDiffParmReporter : public FiniteElementDiffParmReporter
{
  public:
    using Base = FiniteElementDiffParmReporter;
    using Base::Base;

    U64 uid() const noexcept;

    class BuildInfo
    {
      public:
    };

    class DiffParmInfo
    {
      public:
        DiffParmInfo(FiniteElementExtraConstitutionDiffParmReporter* impl,
                     Base::DiffParmInfo& diff_parm_info)
            : m_impl(impl)
            , m_diff_parm_info(diff_parm_info)
        {
        }

        muda::CBufferView<Vector3>  xs() const noexcept;
        muda::CBufferView<Vector3>  x_bars() const noexcept;
        muda::CBufferView<Vector3i> indices() const noexcept;
        muda::CBufferView<Float>    thicknesses() const noexcept;
        muda::CBufferView<IndexT>   is_fixed() const noexcept;

        muda::CBufferView<Float> plastic_thetas() const noexcept;
        muda::CBufferView<Float> anchor_thetas() const noexcept;

        muda::CBufferView<Float> d_anchor_thetas_d_thres_theta() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_yield_ori() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_k() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_hardening_k() const noexcept;

        muda::CBufferView<Float> plastic_thetas_warp() const noexcept;
        muda::CBufferView<Float> plastic_thetas_weft() const noexcept;
        muda::CBufferView<Float> anchor_thetas_warp() const noexcept;
        muda::CBufferView<Float> anchor_thetas_weft() const noexcept;

        muda::CBufferView<Float> d_anchor_thetas_d_thres_theta_warp() const noexcept;
        muda::CBufferView<Float> d_anchor_thetas_d_thres_theta_weft() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_yield_ori_warp() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_yield_ori_weft() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_k_warp() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_k_weft() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_hardening_k_warp() const noexcept;
        muda::CBufferView<Float> d_plastic_theta_d_hardening_k_weft() const noexcept;


        SizeT  frame() const;
        IndexT dof_offset(SizeT frame) const;
        IndexT dof_count(SizeT frame) const;

        muda::TripletMatrixView<Float, 3, 1> pGpP() const;
        Float                                dt() const;

      protected:
        FiniteElementExtraConstitutionDiffParmReporter* m_impl = nullptr;
        Base::DiffParmInfo&                             m_diff_parm_info;
    };

    using DiffParmUpdateInfo = GlobalDiffSimManager::DiffParmUpdateInfo;
    using FilteredInfo       = FiniteElementExtraConstitution::FilteredInfo;

  protected:
    virtual U64 get_uid() const noexcept = 0;

    virtual void do_build(BuildInfo& info)   = 0;
    virtual void do_init(FilteredInfo& info) = 0;

    virtual void do_assemble(DiffParmInfo& info) = 0;

    FiniteElementExtraConstitution& constitution() noexcept;

  private:
    friend class FiniteElementMethod;

    virtual void do_build(FiniteElementDiffParmReporter::BuildInfo& info) override final;

    void connect();  // only be called by FiniteElementMethod

    virtual void do_init() final;

    virtual void do_assemble(Base::DiffParmInfo& info) final;

    FiniteElementExtraConstitution* m_constitution = nullptr;
    FiniteElementMethod*            m_fem          = nullptr;
};
}  // namespace uipc::backend::cuda
