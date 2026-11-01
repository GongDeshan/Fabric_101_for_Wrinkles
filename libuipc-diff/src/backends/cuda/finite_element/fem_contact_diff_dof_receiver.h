#pragma once
#include <contact_system/contact_diff_dof_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>
namespace uipc::backend::cuda
{
class FEMContactDiffDofReceiver final : public ContactDiffDofReceiver
{
  public:
    using ContactDiffDofReceiver::ContactDiffDofReceiver;

    class Impl
    {
      public:
        FiniteElementVertexReporter*    vertex_reporter = nullptr;
        muda::CBCOOMatrixView<Float, 3> H;
    };

    auto H() const { return m_impl.H; }

  private:
    virtual void do_report(ClassifyInfo& info) override;
    virtual void do_receive(ClassifiedDiffDofInfo& info) override;
    virtual void do_build(BuildInfo& info) override;
    virtual void do_init() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda