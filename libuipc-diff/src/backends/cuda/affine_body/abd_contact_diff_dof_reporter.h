#pragma once
#include <affine_body/affine_body_diff_dof_reporter.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <affine_body/abd_contact_diff_dof_receiver.h>

namespace uipc::backend::cuda
{
class ABDContactDiffDofReporter final : public AffineBodyDiffDofReporter
{
  public:
    using AffineBodyDiffDofReporter::AffineBodyDiffDofReporter;

    class Impl
    {
      public:
        void assemble(DiffDofInfo& info);

        AffineBodyDynamics*        abd             = nullptr;
        AffineBodyVertexReporter*  vertex_reporter = nullptr;
        ABDContactDiffDofReceiver* receiver        = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info) override;
    virtual void do_assemble(DiffDofInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda