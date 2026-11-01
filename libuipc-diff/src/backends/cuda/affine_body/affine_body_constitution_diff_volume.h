#pragma once
#include <affine_body/affine_body_diff_parm_reporter.h>

namespace uipc::backend::cuda
{
class ABDGradientHessianComputer;
class ABDDiffSimManager;

class AffineBodyConstitutionDiffVolume : public AffineBodyDiffParmReporter
{
  public:
    using AffineBodyDiffParmReporter::AffineBodyDiffParmReporter;

    class Impl
    {
      public:
        ABDGradientHessianComputer* abd_gradient_hessian_computer;
        ABDDiffSimManager*          abd_diff_sim_manager;

        void assemble(DiffParmInfo& info);
    };

  private:
    virtual void do_build(BuildInfo& info) override;

    virtual void do_report_extent(DiffParmExtentInfo& info) override;
    virtual void do_update(DiffParmUpdateInfo& info) override;

    virtual void do_init() override;
    virtual void do_assemble(DiffParmInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
