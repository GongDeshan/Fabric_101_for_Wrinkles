#include <contact_system/contact_diff_parm_receiver.h>

namespace uipc::backend::cuda
{
void ContactDiffParmReceiver::do_build()
{
    auto& manager = require<GlobalDiffContactManager>();

    BuildInfo info;
    do_build(info);

    manager.add_receiver(this);
}

void ContactDiffParmReceiver::report(ClassifyInfo& info)
{
    this->do_report(info);
    UIPC_ASSERT(info.is_empty() || info.is_diag(),
                "<{}>: off-diagonal ClassifyInfo is meaningless in ContactDiffParmReceiver",
                name());
}

void ContactDiffParmReceiver::receive(ClassifiedDiffParmInfo& info)
{
    this->do_receive(info);
}
}  // namespace uipc::backend::cuda