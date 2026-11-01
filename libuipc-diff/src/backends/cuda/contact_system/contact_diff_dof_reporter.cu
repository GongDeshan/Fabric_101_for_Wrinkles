#include <contact_system/contact_diff_dof_reporter.h>

namespace uipc::backend::cuda
{
void ContactDiffDofReporter::do_build()
{
    auto& manager = require<GlobalDiffContactManager>();

    BuildInfo info;
    do_build(info);

    manager.add_reporter(this);
}
}  // namespace uipc::backend::cuda