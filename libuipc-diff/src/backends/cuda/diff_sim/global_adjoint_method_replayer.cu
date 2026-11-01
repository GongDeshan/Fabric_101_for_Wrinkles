#include <diff_sim/global_adjoint_method_replayer.h>
#include <diff_sim/adjoint_method_replayer.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<cuda::GlobalAdjointMethodReplayer>
{
  public:
    static U<cuda::GlobalAdjointMethodReplayer> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<cuda::SimEngine&>(engine).world().scene();
        if(!scene.info()["diff_sim"]["enable"].get<bool>())
            return nullptr;
        return uipc::make_unique<cuda::GlobalAdjointMethodReplayer>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalAdjointMethodReplayer);

IndexT GlobalAdjointMethodReplayer::dof_offset(SizeT frame) const
{
    UIPC_ASSERT(frame >= 1 && frame < m_impl.frame_dof_offsets.size(),
                "Frame out of range, frame={}, frame_dof_offsets.size={}, why can it happen?",
                frame,
                m_impl.frame_dof_offsets.size());

    return m_impl.frame_dof_offsets[frame];
}

IndexT GlobalAdjointMethodReplayer::dof_count(SizeT frame) const
{
    UIPC_ASSERT(frame >= 1 && frame < m_impl.frame_dof_counts.size(),
                "Frame out of range, frame={}, frame_dof_counts.size={}, why can it happen?",
                frame,
                m_impl.frame_dof_counts.size());

    return m_impl.frame_dof_counts[frame];
}

void GlobalAdjointMethodReplayer::do_build()
{
    m_impl.global_linear_system = &require<GlobalLinearSystem>();

    m_impl.frame_dof_counts.reserve(1024);
    m_impl.frame_dof_offsets.reserve(1024);
}

void GlobalAdjointMethodReplayer::add_replayer(AdjointMethodReplayer* replayer)
{
    check_state(SimEngineState::BuildSystems, "add_replayer()");
    m_impl.replayers.register_subsystem(*replayer);
}

void GlobalAdjointMethodReplayer::record(bool enable_grad, SizeT frame)
{
    // 1. Update frame dof counts and offsets
    {
        if(frame == 1)  // clear the frame records when frame == 1 (reset)
        {
            m_impl.frame_dof_counts.clear();
            m_impl.frame_dof_offsets.clear();

            m_impl.frame_dof_counts.push_back(0);   // invalid frame 0
            m_impl.frame_dof_offsets.push_back(0);  // invalid frame 0

            for(auto& replayer : m_impl.replayers.view())
            {
                replayer->clear();
            }
        }


        UIPC_ASSERT(m_impl.frame_dof_counts.size() == frame,
                    "Frame mismatch, frame_dof_counts.size={}, frame={}, why can it happen?",
                    m_impl.frame_dof_counts.size(),
                    frame);


        auto dof_count = m_impl.global_linear_system->dof_count();

        m_impl.frame_dof_offsets.push_back(m_impl.frame_dof_offsets.back()
                                           + m_impl.frame_dof_counts.back());
        m_impl.frame_dof_counts.push_back(dof_count);
    }

    // 2. Let AdjointMethodReplayers record themselves
    if(enable_grad)
    {
        Timer timer{"Adjoint Method Record State"};
        auto  view = m_impl.replayers.view();
        for(auto& replayer : view)
        {
            RecordInfo info;
            info.m_frame = frame;
            replayer->record(info);
        }
    }
}

void GlobalAdjointMethodReplayer::replay(SizeT frame)
{
    // replay to the frame
    engine().m_current_frame = frame;
    auto view                = m_impl.replayers.view();
    for(auto& replayer : view)
    {
        ReplayInfo info;
        info.m_frame = frame;
        replayer->replay(info);
    }
}
}  // namespace uipc::backend::cuda
