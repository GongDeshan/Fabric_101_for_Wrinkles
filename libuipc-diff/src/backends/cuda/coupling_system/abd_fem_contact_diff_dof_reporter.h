#pragma once
#include <coupling_system/abd_fem_contact_diff_dof_receiver.h>
#include <coupling_system/fem_abd_contact_diff_dof_receiver.h>
#include <diff_sim/diff_dof_reporter.h>

namespace uipc::backend::cuda
{
class ABDFEMDiffDofReporter final : public DiffDofReporter
{
  public:
    using DiffDofReporter::DiffDofReporter;

    using DiffDofExtentInfo = GlobalDiffSimManager::DiffDofExtentInfo;
    using DiffDofInfo       = GlobalDiffSimManager::DiffDofInfo;

    class Impl
    {
      public:
        void assemble(GlobalDiffSimManager::DiffDofInfo& info);

        AffineBodyDynamics*  abd = nullptr;
        FiniteElementMethod* fem = nullptr;

        AffineBodyVertexReporter*    abd_vertex_reporter = nullptr;
        FiniteElementVertexReporter* fem_vertex_reporter = nullptr;

        ABDFEMContactDiffDofReceiver* abd_fem_receiver = nullptr;
        FEMABDContactDiffDofReceiver* fem_abd_receiver = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info) override;
    virtual void do_assemble(GlobalDiffSimManager::DiffDofInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda