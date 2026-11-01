#include <diff_sim/global_diff_sim_manager.h>
#include <diff_sim/diff_dof_reporter.h>
#include <diff_sim/diff_parm_reporter.h>
#include <linear_system/global_linear_system.h>
#include <sim_engine.h>
#include <utils/offset_count_collection.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <diff_sim/enable_grad_feature.h>
#include <diff_sim/global_adjoint_method_replayer.h>
#include <global_geometry/global_vertex_manager.h>
#include <diff_sim/update_parameter_feature.h>
#include <uipc/core/scene.h>
#include <contact_system/global_diff_contact_manager.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<cuda::GlobalDiffSimManager>
{
  public:
    static U<cuda::GlobalDiffSimManager> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<SimEngine&>(engine).world().scene();
        if(!scene.info()["diff_sim"]["enable"].get<bool>())
        {
            return nullptr;
        }
        return uipc::make_unique<cuda::GlobalDiffSimManager>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalDiffSimManager);

void GlobalDiffSimManager::do_build()
{
    m_impl.global_linear_system = &require<GlobalLinearSystem>();
    m_impl.global_adjoint_method_replayer = &require<GlobalAdjointMethodReplayer>();
    m_impl.global_vertex_manager       = &require<GlobalVertexManager>();
    m_impl.sim_engine                  = &engine();
    m_impl.global_diff_contact_manager = find<GlobalDiffContactManager>();

    {
        auto overrider = uipc::make_shared<EnableGradFeatureOverrider>();
        m_impl.enable_grad_feature =
            uipc::make_shared<diff_sim::EnableGradFeature>(overrider);
        features().insert(m_impl.enable_grad_feature);
    }

    {
        auto overrider = uipc::make_shared<UpdateParameterFeatureOverrider>(*this);
        auto feature = uipc::make_shared<diff_sim::UpdateParameterFeature>(overrider);
        features().insert(feature);
    }

    on_write_scene([&] { m_impl.write_scene(world()); });
}

void GlobalDiffSimManager::Impl::init(WorldVisitor& world)
{
    auto& diff_sim   = world.scene().diff_sim();
    auto  parm_view  = diff_sim.parameters().view();
    total_parm_count = parm_view.size();

    // 1) Copy the parameters to the device
    parameters.resize(total_parm_count);
    parameters.view().copy_from(parm_view.data());

    // 2) Init the diff_parm_reporters
    {
        auto diff_parm_reporter_view = diff_parm_reporters.view();
        for(auto&& [i, R] : enumerate(diff_parm_reporter_view))
        {
            R->m_index = i;
        }

        diff_parm_triplet_offset_count.resize(diff_parm_reporter_view.size());
    }

    // 3) Init the diff_dof_reporters
    {
        auto diff_dof_reporter_view = diff_dof_reporters.view();
        for(auto&& [i, R] : enumerate(diff_dof_reporter_view))
        {
            R->m_index = i;
        }

        diff_dof_triplet_offset_count.resize(diff_dof_reporter_view.size());
    }
}

void GlobalDiffSimManager::Impl::update(muda::CBufferView<Float> parms)
{
    Timer timer{"Update Parameter"};

    auto& diff_sim = sim_engine->world().scene().diff_sim();

    // 1) Update the parameters
    parameters.view().copy_from(parms);

    // 2) Update the diff_parm_reporters

    if(global_diff_contact_manager)
    {
        global_diff_contact_manager->update();
    }

    auto diff_parm_reporter_view = diff_parm_reporters.view();
    for(auto&& R : diff_parm_reporter_view)
    {
        DiffParmUpdateInfo info{this};
        R->update_diff_parm(info);
    }

    auto action_view = on_parm_update.view();
    for(auto&& f : action_view)
    {
        DiffParmUpdateInfo info{this};
        f(info);
    }

    global_vertex_manager->update();
}

void GlobalDiffSimManager::Impl::assemble(SizeT frame)
{
    auto F = frame;

    auto dof_offset = global_adjoint_method_replayer->dof_offset(F);
    auto dof_count  = global_adjoint_method_replayer->dof_count(F);

    auto& diff_sim = sim_engine->world().scene().diff_sim();

    // 1) Assemble the pGpP Triplets
    {
        auto diff_parm_reporter_view  = diff_parm_reporters.view();
        auto diff_parm_triplet_counts = diff_parm_triplet_offset_count.counts();

        for(auto&& [i, R] : enumerate(diff_parm_reporter_view))
        {
            DiffParmExtentInfo info{this, R->m_index};
            R->report_extent(info);

            diff_parm_triplet_counts[i] = info.m_triplet_count;
        }

        diff_parm_triplet_offset_count.scan();

        auto M             = dof_offset + dof_count;  // total dof count
        auto N             = total_parm_count;
        auto triplet_count = diff_parm_triplet_offset_count.total_count();

        local_triplet_pGpP.reshape(M, N);
        local_triplet_pGpP.resize_triplets(triplet_count);

        for(auto&& R : diff_parm_reporter_view)
        {
            DiffParmInfo info{this, R->m_index};
            R->assemble_diff_parm(info);
        }

        // NOW: Local Triplet is ready
    }

    // 2) Assemble the H Triplets
    {
        auto diff_dof_reporter_view  = diff_dof_reporters.view();
        auto diff_dof_triplet_counts = diff_dof_triplet_offset_count.counts();

        for(auto&& [i, R] : enumerate(diff_dof_reporter_view))
        {
            DiffDofExtentInfo info{this, R->m_index};
            R->report_extent(info);

            diff_dof_triplet_counts[i] = info.m_triplet_count;
        }

        diff_dof_triplet_offset_count.scan();

        auto N             = dof_offset + dof_count;  // total dof count
        auto triplet_count = diff_dof_triplet_offset_count.total_count();

        local_triplet_H.reshape(N, N);
        local_triplet_H.resize_triplets(triplet_count);

        for(auto&& R : diff_dof_reporter_view)
        {
            DiffDofInfo info{this, R->m_index};
            R->assemble_diff_dof(info);
        }

        // NOW: Local Triplet is ready
    }
}

void GlobalDiffSimManager::Impl::write_scene(WorldVisitor& world)
{
    auto& scene    = world.scene().ref();
    auto& diff_sim = scene.diff_sim();
    auto  parm     = view(diff_sim.parameters());
    parameters.view().copy_to(parm.data());
    scene.diff_sim().parameters().broadcast();
}

IndexT GlobalDiffSimManager::dof_offset(SizeT frame) const
{
    return m_impl.global_adjoint_method_replayer->dof_offset(frame);
}

IndexT GlobalDiffSimManager::dof_count(SizeT frame) const
{
    return m_impl.global_adjoint_method_replayer->dof_count(frame);
}

muda::CBufferView<Float> GlobalDiffSimManager::parameters() const
{
    return m_impl.parameters.view();
}

muda::CTripletMatrixView<Float, 3, 1> GlobalDiffSimManager::pGipP() const
{
    return m_impl.local_triplet_pGpP;
}

muda::CTripletMatrixView<Float, 3, 3> GlobalDiffSimManager::Hij() const
{
    return m_impl.local_triplet_H;
}

bool GlobalDiffSimManager::is_grad_enabled() const
{
    return m_impl.enable_grad_feature->is_grad_enabled();
}

void GlobalDiffSimManager::init()
{
    m_impl.init(world());
}

void GlobalDiffSimManager::assemble(SizeT frame)
{
    m_impl.assemble(frame);
}

void GlobalDiffSimManager::update(muda::CBufferView<Float> parms)
{
    m_impl.update(parms);
}

void GlobalDiffSimManager::add_reporter(DiffDofReporter* subsystem)
{
    UIPC_ASSERT(subsystem != nullptr, "subsystem is nullptr");
    m_impl.diff_dof_reporters.register_subsystem(*subsystem);
}

void GlobalDiffSimManager::add_reporter(DiffParmReporter* subsystem)
{
    UIPC_ASSERT(subsystem != nullptr, "subsystem is nullptr");
    m_impl.diff_parm_reporters.register_subsystem(*subsystem);
}

void GlobalDiffSimManager::on_parm_update(SimSystem* system,
                                          std::function<void(DiffParmUpdateInfo&)>&& f)
{
    UIPC_ASSERT(system != nullptr, "system is nullptr");
    m_impl.on_parm_update.register_action(*system, std::move(f));
}

muda::TripletMatrixView<Float, 3, 1> GlobalDiffSimManager::DiffParmInfo::pGpP() const
{
    auto offset = m_impl->diff_parm_triplet_offset_count.offsets()[m_index];
    auto count  = m_impl->diff_parm_triplet_offset_count.counts()[m_index];
    return m_impl->local_triplet_pGpP.view().subview(offset, count);
}

muda::TripletMatrixView<Float, 3, 3> GlobalDiffSimManager::DiffDofInfo::H() const
{
    auto offset = m_impl->diff_dof_triplet_offset_count.offsets()[m_index];
    auto count  = m_impl->diff_dof_triplet_offset_count.counts()[m_index];
    return m_impl->local_triplet_H.view().subview(offset, count);
}

SizeT GlobalDiffSimManager::BaseInfo::frame() const
{
    return m_impl->sim_engine->frame();
}

IndexT GlobalDiffSimManager::BaseInfo::dof_offset(SizeT frame) const
{
    return m_impl->global_adjoint_method_replayer->dof_offset(frame);
}

IndexT GlobalDiffSimManager::BaseInfo::dof_count(SizeT frame) const
{
    return m_impl->global_adjoint_method_replayer->dof_count(frame);
}

muda::CBufferView<Float> GlobalDiffSimManager::DiffParmUpdateInfo::parameters() const noexcept
{
    return m_impl->parameters.view();
}
}  // namespace uipc::backend::cuda