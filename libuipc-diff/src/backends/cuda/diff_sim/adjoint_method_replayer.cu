#include <diff_sim/adjoint_method_replayer.h>

namespace uipc::backend::cuda
{
void AdjointMethodReplayer::do_build()
{
    auto& global_replayer = require<GlobalAdjointMethodReplayer>();
    global_replayer.add_replayer(this);

    BuildInfo info;
    do_build(info);
}

void AdjointMethodReplayer::record(RecordInfo& info)
{
    do_record(info);
}

void AdjointMethodReplayer::replay(GlobalAdjointMethodReplayer::ReplayInfo& info)
{
    do_replay(info);
}

void AdjointMethodReplayer::clear()
{
    do_clear();
}
}  // namespace uipc::backend::cuda
