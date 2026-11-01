#include <diff_sim/learning_rate_reporter.h>
#include <diff_sim/global_learning_rate_manager.h>

namespace uipc::backend::cuda
{
void LearningRateReporter::do_build()
{
    auto& lr_manager = require<GlobalLearningRateManager>();
    lr_manager.add_reporter(this);

    // let subclasses do their thing
    BuildInfo info;
    do_build(info);
}
}  // namespace uipc::backend::cuda
