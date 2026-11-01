#include <contact_system/contact_diff_dof_receiver.h>

namespace uipc::backend::cuda
{
void ContactDiffDofReceiver::do_build()
{
    auto& manager = require<GlobalDiffContactManager>();

    BuildInfo info;
    do_build(info);

    manager.add_receiver(this);
}
}  // namespace uipc::backend::cuda