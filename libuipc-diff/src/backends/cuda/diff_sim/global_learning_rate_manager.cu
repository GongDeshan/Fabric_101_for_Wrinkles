#include <diff_sim/global_learning_rate_manager.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <diff_sim/learning_rate_feature.h>
#include <diff_sim/learning_rate_reporter.h>
#include <sim_engine.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalLearningRateManager);

void GlobalLearningRateManager::do_build()
{
    m_impl.diff_sim_manager = &require<GlobalDiffSimManager>();

    // create the feature
    auto overrider = uipc::make_shared<LearningRateFeatureOverrider>(*this);
    auto feature = uipc::make_shared<diff_sim::LearningRateFeature>(overrider);
    features().insert(feature);
}

Float GlobalLearningRateManager::learning_rate(muda::CBufferView<Float> aim_parms)
{
    Timer timer{"Compute Learning Rate"};

    auto F = engine().frame();
    UIPC_ASSERT(F == 0, "`learning_rate()` should be called at frame 0, but now {}", F);

    // 1. try update the parameters
    spdlog::info("Try Update Parameters({})", aim_parms.size());
    {
        // 1.1 record the safe state
        auto parms = m_impl.diff_sim_manager->parameters();

        m_impl.safe_parameters.resize(parms.size());
        m_impl.safe_parameters.view().copy_from(parms);

        UIPC_ASSERT(aim_parms.size() == parms.size(),
                    "The size of aim_parms({}) is not equal to the size of parameters({})",
                    aim_parms.size(),
                    parms.size());

        for(auto&& reporter : m_impl.learning_rate_reporters.view())
        {
            RecordSafeStateInfo info{&m_impl, aim_parms};
            reporter->record_safe_state(info);
        }

        // 1.2 update the parameters
        m_impl.diff_sim_manager->update(aim_parms);
    }


    // 2. try to find out a proper learning rate
    Float alpha = 1.0;
    {
        LearningRateReporter* min_reporter = nullptr;

        for(auto&& reporter : m_impl.learning_rate_reporters.view())
        {
            LearningRateInfo info{&m_impl, aim_parms};
            reporter->report_learning_rate(info);

            UIPC_ASSERT(info.m_learning_rate >= 0.0,
                        "Negative learning rate ({}) from {}, why can it happen?",
                        info.m_learning_rate,
                        reporter->name());

            if(info.m_learning_rate <= alpha)
            {
                alpha        = info.m_learning_rate;
                min_reporter = reporter;
            }
        }

        alpha = std::min(alpha, 1.0);  // no more than 1.0
    }
    spdlog::info("Global Learning Rate: {}", alpha);

    // 3. recover the prev parameters
    {
        m_impl.diff_sim_manager->update(m_impl.safe_parameters.view());
    }
    spdlog::info("Recover Previous Parameters({})", m_impl.safe_parameters.size());

    return alpha;
}

void GlobalLearningRateManager::add_reporter(LearningRateReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "nullptr reporter");
    m_impl.learning_rate_reporters.register_subsystem(*reporter);
}

muda::CBufferView<Float> GlobalLearningRateManager::BaseParameterInfo::aim_parameters() const
{
    return m_aim_parms;
}

muda::CBufferView<Float> GlobalLearningRateManager::BaseParameterInfo::safe_parameters() const
{
    return m_impl->diff_sim_manager->parameters();
}

void GlobalLearningRateManager::LearningRateInfo::learning_rate(Float rate)
{
    m_learning_rate = rate;
}
}  // namespace uipc::backend::cuda
