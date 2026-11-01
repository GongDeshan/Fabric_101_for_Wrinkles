#pragma once
#include <finite_element/finite_element_diff_parm_reporter.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <finite_element/fem_contact_diff_parm_receiver.h>
namespace uipc::backend::cuda
{
class FEMContactDiffParmReporter final : public FiniteElementDiffParmReporter
{
  public:
    using FiniteElementDiffParmReporter::FiniteElementDiffParmReporter;

    class Impl
    {
      public:
        void assemble(DiffParmInfo& info);

        FiniteElementVertexReporter* vertex_reporter = nullptr;
        FEMContactDiffParmReceiver*  receiver        = nullptr;
        FiniteElementMethod*         fem             = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_init() override;
    virtual void do_update(GlobalDiffSimManager::DiffParmUpdateInfo& info) override;
    virtual void do_report_extent(GlobalDiffSimManager::DiffParmExtentInfo& info) override;
    virtual void do_assemble(DiffParmInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda