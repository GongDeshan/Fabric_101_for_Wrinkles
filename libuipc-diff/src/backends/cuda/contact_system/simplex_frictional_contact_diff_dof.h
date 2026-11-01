#pragma once
#include <contact_system/contact_diff_dof_reporter.h>
#include <contact_system/contact_coeff.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::cuda
{
class SimplexFrictionalContactDiffDof : public ContactDiffDofReporter
{
  public:
    using ContactDiffDofReporter::ContactDiffDofReporter;

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

        muda::CBuffer2DView<ContactCoeff> contact_tabular() const;
        muda::CBufferView<Vector4i>       friction_PTs() const;
        muda::CBufferView<Vector4i>       friction_EEs() const;
        muda::CBufferView<Vector3i>       friction_PEs() const;
        muda::CBufferView<Vector2i>       friction_PPs() const;
        muda::CBufferView<Vector3>        positions() const;
        muda::CBufferView<Vector3>        prev_positions() const;
        muda::CBufferView<Vector3>        rest_positions() const;
        muda::CBufferView<Float>          thicknesses() const;
        muda::CBufferView<IndexT>         contact_element_ids() const;
        Float                             d_hat() const;
        Float                             dt() const;
        Float                             eps_velocity() const;

      private:
        friend class SimplexFrictionalContactDiffDof;
        Impl* m_impl;
    };

    class DiffDofInfo : public BaseInfo
    {
      public:
        DiffDofInfo(Impl* impl, muda::TripletMatrixView<Float, 3> H) noexcept
            : BaseInfo{impl}
            , m_H{H}
        {
        }

        // PT DiffDof
        muda::TripletMatrixView<Float, 3> PT_H() const;
        // EE DiffDof
        muda::TripletMatrixView<Float, 3> EE_H() const;
        // PE DiffDof
        muda::TripletMatrixView<Float, 3> PE_H() const;
        // PP DiffDof
        muda::TripletMatrixView<Float, 3> PP_H() const;


      private:
        friend class SimplexFrictionalContactDiffDof;
        muda::TripletMatrixView<Float, 3> m_H;
    };

    class Impl
    {
      public:
        GlobalTrajectoryFilter* global_trajectory_filter = nullptr;
        GlobalContactManager*   global_contact_manager   = nullptr;
        GlobalVertexManager*    global_vertex_manager    = nullptr;
        SimSystemSlot<SimplexTrajectoryFilter> simplex_trajectory_filter;

        SizeT PT_H_offset = 0;
        SizeT PT_H_count  = 0;

        SizeT EE_H_offset = 0;
        SizeT EE_H_count  = 0;

        SizeT PE_H_offset = 0;
        SizeT PE_H_count  = 0;

        SizeT PP_H_offset = 0;
        SizeT PP_H_count  = 0;

        Float dt = 0;
    };

  protected:
    virtual void do_build(BuildInfo& info)      = 0;
    virtual void do_init(InitInfo&)             = 0;
    virtual void do_assemble(DiffDofInfo& info) = 0;

  private:
    virtual void do_build(ContactDiffDofReporter::BuildInfo&) override final;
    virtual void do_report_extent(DiffDofExtentInfo& info) override final;
    virtual void do_assemble(ContactDiffDofReporter::DiffDofInfo& info) override final;
    virtual void do_init() override final;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
