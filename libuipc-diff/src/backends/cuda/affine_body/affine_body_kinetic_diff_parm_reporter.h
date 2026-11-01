#pragma once
#include <affine_body/affine_body_diff_parm_reporter.h>

namespace uipc::backend::cuda
{
class AffineBodyKineticDiffParmReporter : public AffineBodyDiffParmReporter
{
  public:
    using AffineBodyDiffParmReporter::AffineBodyDiffParmReporter;

    class BuildInfo
    {
      public:
    };

    using DiffParmExtentInfo = AffineBodyDiffParmReporter::DiffParmExtentInfo;

    using DiffParmInfo = AffineBodyDiffParmReporter::DiffParmInfo;

  protected:
    virtual void do_build(BuildInfo& info) = 0;

  private:
    virtual void do_build(AffineBodyDiffParmReporter::BuildInfo& info) override final;
};
}  // namespace uipc::backend::cuda
