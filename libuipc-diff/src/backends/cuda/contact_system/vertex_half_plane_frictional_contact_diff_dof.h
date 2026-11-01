#pragma once
#include <contact_system/contact_diff_dof_reporter.h>
#include <contact_system/contact_coeff.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>

namespace uipc::backend::cuda
{
class GlobalTrajectoryFilter;
class VertexHalfPlaneTrajectoryFilter;

class VertexHalfPlaneFrictionalContactDiffDof : public ContactDiffDofReporter
{
  public:
    using ContactDiffDofReporter::ContactDiffDofReporter;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    class Impl;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        muda::CBuffer2DView<ContactCoeff> contact_tabular() const;
        muda::CBufferView<Vector2i>       friction_PHs() const;
        muda::CBufferView<Vector3>        positions() const;
        muda::CBufferView<Float>          thicknesses() const;
        muda::CBufferView<Vector3>        prev_positions() const;
        muda::CBufferView<Vector3>        rest_positions() const;
        muda::CBufferView<IndexT>         contact_element_ids() const;

        Float d_hat() const;
        Float dt() const;
        Float eps_velocity() const;

        // NOTE: No need to provide half plane related information
        // User can access half plane by require<HalfPlane>() themselves

      private:
        friend class VertexHalfPlaneFrictionalContact;
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

        auto H() const noexcept { return m_H; }

      private:
        friend class VertexHalfPlaneFrictionalContact;
        muda::TripletMatrixView<Float, 3> m_H;
    };

    class Impl
    {
      public:
        GlobalTrajectoryFilter* global_trajectory_filter = nullptr;
        GlobalContactManager*   global_contact_manager   = nullptr;
        GlobalVertexManager*    global_vertex_manager    = nullptr;

        SimSystemSlot<VertexHalfPlaneTrajectoryFilter> veretx_half_plane_trajectory_filter;


        SizeT PH_count = 0;
        Float dt;

        muda::DeviceBuffer<Float>     energies;
        muda::DeviceBuffer<Vector3>   gradients;
        muda::DeviceBuffer<Matrix3x3> hessians;

        Float reserve_ratio = 1.1;

        template <typename T>
        void loose_resize(muda::DeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }
    };

  protected:
    virtual void do_build(BuildInfo& info)      = 0;
    virtual void do_init(InitInfo& info)        = 0;
    virtual void do_assemble(DiffDofInfo& info) = 0;


  private:
    virtual void do_build(ContactDiffDofReporter::BuildInfo&) override final;
    virtual void do_report_extent(DiffDofExtentInfo& info) override final;
    virtual void do_assemble(ContactDiffDofReporter::DiffDofInfo& info) override final;
    virtual void do_init() override final;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
