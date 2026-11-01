#include <affine_body/affine_body_kinetic_diff_parm_reporter.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::cuda
{
void AffineBodyKineticDiffParmReporter::do_build(AffineBodyDiffParmReporter::BuildInfo& info)
{
    auto& affine_body_dynamics = require<AffineBodyDynamics>();
    affine_body_dynamics.add_reporter(this);

    BuildInfo this_info;
    do_build(this_info);  // let subclass build the reporter
}
}  // namespace uipc::backend::cuda
