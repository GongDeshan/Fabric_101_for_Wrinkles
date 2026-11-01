#pragma once
#include <contact_system/contact_diff_parm_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>
namespace uipc::backend::cuda
{
class FEMContactDiffParmReceiver final : public ContactDiffParmReceiver
{
  public:
    using ContactDiffParmReceiver::ContactDiffParmReceiver;

    class Impl
    {
      public:
        FiniteElementVertexReporter*       vertex_reporter = nullptr;
        muda::CBCOOMatrixView<Float, 3, 1> pGpP;
    };

    auto pGpP() const { return m_impl.pGpP; }

  private:
    virtual void do_report(ClassifyInfo& info) override;
    virtual void do_receive(ClassifiedDiffParmInfo& info) override;
    virtual void do_build(BuildInfo& info) override;
    virtual void do_init() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda