#pragma once
#include <contact_system/contact_diff_parm_reporter.h>
#include <contact_system/contact_coeff.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/global_diff_contact_manager.h>

namespace uipc::backend::cuda
{
class SimplexFrictionalContactDiffParm : public ContactDiffParmReporter
{
  public:
    using ContactDiffParmReporter::ContactDiffParmReporter;

    class Impl;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        muda::CBuffer2DView<ContactCoeff>     contact_tabular() const;
        muda::CBuffer2DView<DiffContactCoeff> diff_contact_tabular() const;
        muda::CBufferView<Vector4i>           friction_PTs() const;
        muda::CBufferView<Vector4i>           friction_EEs() const;
        muda::CBufferView<Vector3i>           friction_PEs() const;
        muda::CBufferView<Vector2i>           friction_PPs() const;
        muda::CBufferView<Vector3>            positions() const;
        muda::CBufferView<Vector3>            prev_positions() const;
        muda::CBufferView<Vector3>            rest_positions() const;
        muda::CBufferView<Float>              thicknesses() const;
        muda::CBufferView<IndexT>             contact_element_ids() const;
        Float                                 d_hat() const;
        Float                                 dt() const;
        Float                                 eps_velocity() const;

      private:
        friend class SimplexFrictionalContactDiffParm;
        Impl* m_impl;
    };

    class DiffParmInfo : public BaseInfo
    {
      public:
        DiffParmInfo(Impl* impl, muda::TripletMatrixView<Float, 3, 1> H) noexcept
            : BaseInfo{impl}
            , m_pGpP{H}
        {
        }

        // PT DiffParm
        muda::TripletMatrixView<Float, 3, 1> PT_pGpP() const;
        // EE DiffParm
        muda::TripletMatrixView<Float, 3, 1> EE_pGpP() const;
        // PE DiffParm
        muda::TripletMatrixView<Float, 3, 1> PE_pGpP() const;
        // PP DiffParm
        muda::TripletMatrixView<Float, 3, 1> PP_pGpP() const;


      private:
        friend class SimplexFrictionalContactDiffParm;
        muda::TripletMatrixView<Float, 3, 1> m_pGpP;
    };

    class Impl
    {
      public:
        GlobalTrajectoryFilter*   global_trajectory_filter    = nullptr;
        GlobalContactManager*     global_contact_manager      = nullptr;
        GlobalVertexManager*      global_vertex_manager       = nullptr;
        GlobalDiffContactManager* global_diff_contact_manager = nullptr;
        SimSystemSlot<SimplexTrajectoryFilter> simplex_trajectory_filter;

        SizeT PT_pGpP_offset = 0;
        SizeT PT_pGpP_count  = 0;

        SizeT EE_pGpP_offset = 0;
        SizeT EE_pGpP_count  = 0;

        SizeT PE_pGpP_offset = 0;
        SizeT PE_pGpP_count  = 0;

        SizeT PP_pGpP_offset = 0;
        SizeT PP_pGpP_count  = 0;

        Float dt = 0;
    };

  protected:
    virtual void do_build(BuildInfo& info)       = 0;
    virtual void do_init(InitInfo&)              = 0;
    virtual void do_assemble(DiffParmInfo& info) = 0;

  private:
    virtual void do_build(ContactDiffParmReporter::BuildInfo&) override final;
    virtual void do_report_extent(DiffParmExtentInfo& info) override final;
    virtual void do_assemble(ContactDiffParmReporter::DiffParmInfo& info) override final;
    virtual void do_init() override final;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
