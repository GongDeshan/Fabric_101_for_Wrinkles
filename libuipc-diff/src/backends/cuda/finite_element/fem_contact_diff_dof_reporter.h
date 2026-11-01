#pragma once
#include <finite_element/finite_element_diff_dof_reporter.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <finite_element/fem_contact_diff_dof_receiver.h>
namespace uipc::backend::cuda
{
class FEMContactDiffDofReporter final : public FiniteElementDiffDofReporter
{
  public:
    using FiniteElementDiffDofReporter::FiniteElementDiffDofReporter;

    class Impl
    {
      public:
        void assemble(DiffDofInfo& info);

        FiniteElementVertexReporter* vertex_reporter = nullptr;
        FEMContactDiffDofReceiver*   receiver        = nullptr;
        FiniteElementMethod*         fem             = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info) override;
    virtual void do_assemble(DiffDofInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda