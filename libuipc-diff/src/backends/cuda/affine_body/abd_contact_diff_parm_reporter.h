#pragma once
#include <affine_body/affine_body_diff_parm_reporter.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <affine_body/abd_contact_diff_parm_receiver.h>

namespace uipc::backend::cuda
{
class ABDContactDiffParmReporter final : public AffineBodyDiffParmReporter
{
  public:
    using AffineBodyDiffParmReporter::AffineBodyDiffParmReporter;

    class Impl
    {
      public:
        void assemble(DiffParmInfo& info);

        AffineBodyDynamics*         abd             = nullptr;
        AffineBodyVertexReporter*   vertex_reporter = nullptr;
        ABDContactDiffParmReceiver* receiver        = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_init() override;
    virtual void do_update(DiffParmUpdateInfo& info) override;
    virtual void do_report_extent(DiffParmExtentInfo& info) override;
    virtual void do_assemble(DiffParmInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda