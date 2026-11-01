#pragma once
#include <diff_sim/diff_parm_reporter.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_diff_sim_manager.h>
#include <diff_sim/global_diff_sim_manager.h>

namespace uipc::backend::cuda
{
class AffineBodyDiffParmReporter : public DiffParmReporter
{
  public:
    using DiffParmReporter::DiffParmReporter;

    class Impl;

    class BuildInfo
    {
      public:
    };

    using DiffParmUpdateInfo = GlobalDiffSimManager::DiffParmUpdateInfo;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, GlobalDiffSimManager::BaseInfo& global_info, Float dt)
            : m_impl(impl)
            , m_global_info(global_info)
            , m_dt(dt)
        {
        }

        SizeT  frame() const;
        IndexT dof_offset(SizeT frame) const;
        IndexT dof_count(SizeT frame) const;
        Float  dt() const;

        muda::CBufferView<ABDJacobi>           Js() const;
        muda::CBufferView<IndexT>              v2b() const;
        muda::CBufferView<Vector12>            qs() const;
        muda::CBufferView<Vector12>            q_tildes() const;
        muda::CBufferView<Vector12>            q_prevs() const;
        muda::CBufferView<Vector12>            q_vs() const;
        muda::CBufferView<IndexT>              body_is_fixed() const;
        muda::CBufferView<IndexT>              body_is_dynamic() const;
        muda::CBufferView<ABDJacobiDyadicMass> body_masses() const;
        muda::CBufferView<Matrix12x12>         body_mass_invs() const;
        muda::CBufferView<Vector12>            body_gravities() const;
        muda::CBufferView<Vector3i>            diff_triangles() const;
        muda::CBufferView<Vector4i>            diff_tetrahedra() const;
        muda::CBufferView<IndexT>              diff_vertices() const;
        muda::CBufferView<IndexT>              diff_bodies() const;
        muda::CBufferView<Float>               body_mass_density() const;
        muda::CBufferView<Float>               body_volumes() const;
        muda::CBufferView<Vector3i>            diff_rest_positions() const;


      private:
        friend class Impl;
        Impl*                           m_impl;
        GlobalDiffSimManager::BaseInfo& m_global_info;
        Float                           m_dt = 0.0;
    };

    class DiffParmExtentInfo : public BaseInfo
    {
      public:
        DiffParmExtentInfo(Impl* impl, GlobalDiffSimManager::DiffParmExtentInfo& global_info, Float dt)
            : BaseInfo(impl, global_info, dt)
            , m_info(global_info)
        {
        }

        void triplet_count(SizeT N);

      private:
        friend class Impl;
        GlobalDiffSimManager::DiffParmExtentInfo& m_info;
    };

    class DiffParmInfo : public BaseInfo
    {
      public:
        DiffParmInfo(Impl* impl, GlobalDiffSimManager::DiffParmInfo& global_info, Float dt)
            : BaseInfo(impl, global_info, dt)
            , m_info(global_info)
        {
        }

        muda::TripletMatrixView<Float, 3, 1> pGpP() const;

      private:
        friend class Impl;
        GlobalDiffSimManager::DiffParmInfo& m_info;
    };

    class Impl
    {
      public:
        AffineBodyDynamics* abd                  = nullptr;
        ABDDiffSimManager*  abd_diff_sim_manager = nullptr;
        Float               dt                   = 0.0;
    };

  protected:
    virtual void do_build(BuildInfo& info)                  = 0;
    virtual void do_init()                                  = 0;
    virtual void do_assemble(DiffParmInfo& info)            = 0;
    virtual void do_report_extent(DiffParmExtentInfo& info) = 0;

    AffineBodyDynamics::Impl& abd() const noexcept
    {
        return m_impl.abd->m_impl;
    }

  private:
    friend class AffineBodyDynamics;
    void         init();  // only called by AffineBodyDynamics
    virtual void do_build(DiffParmReporter::BuildInfo& info) override final;
    virtual void do_assemble(GlobalDiffSimManager::DiffParmInfo& info) override final;
    virtual void do_report_extent(GlobalDiffSimManager::DiffParmExtentInfo& info) override final;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
