#include <sim_engine.h>
#include <contact_system/global_diff_contact_manager.h>
#include <contact_system/contact_diff_dof_reporter.h>
#include <contact_system/contact_diff_dof_receiver.h>
#include <contact_system/contact_diff_parm_receiver.h>
#include <contact_system/contact_diff_parm_reporter.h>
#include <contact_system/global_contact_manager.h>
#include <uipc/common/zip.h>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::GlobalDiffContactManager>
{
  public:
    static U<cuda::GlobalDiffContactManager> create(cuda::SimEngine& engine)
    {
        auto& info = engine.world().scene().info();

        if(info["contact"]["enable"].get<bool>()       // need enable contact
           && info["diff_sim"]["enable"].get<bool>())  // need enable diff sim
            return make_unique<cuda::GlobalDiffContactManager>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalDiffContactManager);

void GlobalDiffContactManager::do_build()
{
    m_impl.global_vertex_manager   = &require<GlobalVertexManager>();
    m_impl.global_contact_manager  = &require<GlobalContactManager>();
    m_impl.global_diff_sim_manager = &require<GlobalDiffSimManager>();
}

void GlobalDiffContactManager::add_reporter(ContactDiffDofReporter* reporter) noexcept
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    m_impl.diff_dof_reporters.register_subsystem(*reporter);
}

void GlobalDiffContactManager::add_receiver(ContactDiffDofReceiver* receiver) noexcept
{
    check_state(SimEngineState::BuildSystems, "add_receiver()");
    UIPC_ASSERT(receiver != nullptr, "receiver is nullptr");
    m_impl.diff_dof_receivers.register_subsystem(*receiver);
}

void GlobalDiffContactManager::add_reporter(ContactDiffParmReporter* reporter) noexcept
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    m_impl.diff_parm_reporters.register_subsystem(*reporter);
}

void GlobalDiffContactManager::add_receiver(ContactDiffParmReceiver* receiver) noexcept
{
    check_state(SimEngineState::BuildSystems, "add_receiver()");
    UIPC_ASSERT(receiver != nullptr, "receiver is nullptr");
    m_impl.diff_parm_receivers.register_subsystem(*receiver);
}

void GlobalDiffContactManager::init()
{
    m_impl.init(world());
}

void GlobalDiffContactManager::clear_contact_for_adjoint()
{
    m_impl.clear_contact_for_adjoint();
}

// EXPERIMENT (2026-09-20): refresh every receiver, including empty ranges.
// Do not use the regular zero-entry classification/scan path here and do
// not leave stale H[k][k-1] or pGpP views from an earlier backward call.
void GlobalDiffContactManager::Impl::clear_contact_for_adjoint()
{
    auto vertex_count = global_vertex_manager->positions().size();
    auto parm_count = global_diff_sim_manager->parameters().size();
    for(auto&& [i, receiver] : enumerate(diff_dof_receivers.view()))
    {
        auto& H = classified_H[i];
        H.reshape(vertex_count, vertex_count);
        H.resize_triplets(0);
        ClassifiedContactDiffDofInfo info;
        info.m_H = H.view();
        receiver->do_receive(info);
    }
    for(auto&& [i, receiver] : enumerate(diff_parm_receivers.view()))
    {
        auto& pGpP = classified_pGpP[i];
        pGpP.reshape(vertex_count, parm_count);
        pGpP.resize_triplets(0);
        ClassifiedContactDiffParmInfo info;
        info.m_pGpP = pGpP.cview();
        receiver->receive(info);
    }
}

void GlobalDiffContactManager::compute_diff_contact()
{
    m_impl.assemble_diff_dof();
    m_impl.assemble_diff_parm();
    m_impl.convert_matrix();
    m_impl.distribute_diff_dof();
    m_impl.distribute_diff_parm();
}

void GlobalDiffContactManager::update()
{
    auto& diff_sim = world().scene().diff_sim();

    // because the contact tabular is most of the time small
    // we rebuild it every time
    m_impl.global_contact_manager->m_impl._build_device_tabular(world());
}

void GlobalDiffContactManager::Impl::init(WorldVisitor& world)
{
    // 0) Build Diff Contact Coeff
    _build_diff_contact_coeff(world);

    // 1) Diff Dof
    diff_dof_reporter_offsets_counts.resize(diff_dof_reporters.view().size());
    {
        auto reporter_view = diff_dof_reporters.view();
        for(auto&& [i, reporter] : enumerate(reporter_view))
            reporter->do_init();

        auto receiver_view = diff_dof_receivers.view();
        classified_H.resize(receiver_view.size());
        for(auto&& [i, receiver] : enumerate(receiver_view))
            receiver->do_init();
    }

    // 2) Diff Parm
    diff_parm_reporter_offsets_counts.resize(diff_parm_reporters.view().size());
    {
        auto reporter_view = diff_parm_reporters.view();
        for(auto&& [i, reporter] : enumerate(reporter_view))
            reporter->do_init();

        auto receiver_view = diff_parm_receivers.view();
        classified_pGpP.resize(receiver_view.size());
        for(auto&& [i, receiver] : enumerate(receiver_view))
            receiver->do_init();
    }
}

void GlobalDiffContactManager::Impl::_build_diff_contact_coeff(WorldVisitor& world)
{
    auto contact_models = world.scene().contact_tabular().contact_models();

    auto attr_topo            = contact_models.find<Vector2i>("topo");
    auto attr_diff_resistance = contact_models.find<IndexT>("diff/resistance");
    auto attr_diff_friction_rate = contact_models.find<IndexT>("diff/friction_rate");

    UIPC_ASSERT(attr_topo != nullptr, "topo is not found in contact tabular");

    auto topo_view = attr_topo->view();

    IndexT default_diff_resistance    = -1;
    IndexT default_diff_friction_rate = -1;

    if(attr_diff_resistance)
    {
        auto diff_resistance_view = attr_diff_resistance->view();
        default_diff_resistance   = diff_resistance_view[0];
    }

    if(attr_diff_friction_rate)
    {
        auto diff_friction_rate_view = attr_diff_friction_rate->view();
        default_diff_friction_rate   = diff_friction_rate_view[0];
    }

    // if(attr_diff_resistance || attr_diff_friction_rate)
    {
        auto N = world.scene().contact_tabular().element_count();

        h_diff_contact_coeff.resize(N * N,
                                    DiffContactCoeff{.diff_kappa = default_diff_resistance,
                                                     .diff_mu = default_diff_friction_rate});

        if(attr_diff_friction_rate)
        {
            auto diff_friction_rate_view = attr_diff_friction_rate->view();
            for(auto&& [ids, mu] : zip(topo_view, diff_friction_rate_view))
            {

                auto upper                          = ids.x() * N + ids.y();
                h_diff_contact_coeff[upper].diff_mu = mu;
                auto lower                          = ids.y() * N + ids.x();
                h_diff_contact_coeff[lower].diff_mu = mu;
            }
        }

        if(attr_diff_resistance)
        {
            auto diff_resistance_view = attr_diff_resistance->view();
            for(auto&& [ids, kappa] : zip(topo_view, diff_resistance_view))
            {
                auto upper                             = ids.x() * N + ids.y();
                h_diff_contact_coeff[upper].diff_kappa = kappa;
                auto lower                             = ids.y() * N + ids.x();
                h_diff_contact_coeff[lower].diff_kappa = kappa;
            }
        }

        diff_contact_coeff.resize(muda::Extent2D{N, N});
        diff_contact_coeff.copy_from(h_diff_contact_coeff);
    }
}

void GlobalDiffContactManager::Impl::assemble_diff_dof()
{
    auto reporter_view = diff_dof_reporters.view();

    // 1) Collect Extent
    auto triplet_counts = diff_dof_reporter_offsets_counts.counts();
    for(auto&& [i, reporter] : enumerate(reporter_view))
    {
        ContactDiffDofExtentInfo extent_info;
        reporter->do_report_extent(extent_info);
        triplet_counts[i] = extent_info.m_triplet_count;
    }

    // 2) Allocate Memory
    diff_dof_reporter_offsets_counts.scan();
    auto total_count = diff_dof_reporter_offsets_counts.total_count();
    auto N           = global_vertex_manager->positions().size();
    triplet_H.resize(N, N, total_count);

    // 3) Assemble
    auto triplet_view    = triplet_H.view();
    auto triplet_offsets = diff_dof_reporter_offsets_counts.offsets();
    for(auto&& [i, reporter] : enumerate(reporter_view))
    {
        ContactDiffDofInfo info;
        auto               offset = triplet_offsets[i];
        auto               count  = triplet_counts[i];

        info.m_H = triplet_view.subview(offset, count);
        reporter->do_assemble(info);
    }
}

void GlobalDiffContactManager::Impl::assemble_diff_parm()
{
    // If there is no diff contact coeff, we don't need to assemble contact pGpP
    //if(!diff_contact_coeff.total_size())
    //    return;

    auto reporter_view = diff_parm_reporters.view();

    // 1) Collect Extent
    auto triplet_counts = diff_parm_reporter_offsets_counts.counts();
    for(auto&& [i, reporter] : enumerate(reporter_view))
    {
        ContactDiffParmExtentInfo extent_info;
        reporter->do_report_extent(extent_info);
        triplet_counts[i] = extent_info.m_triplet_count;
    }

    // 2) Allocate Memory
    diff_parm_reporter_offsets_counts.scan();
    auto total_count = diff_parm_reporter_offsets_counts.total_count();
    auto N           = global_vertex_manager->positions().size();
    triplet_pGpP.resize(N, N, total_count);

    // 3) Assemble
    auto triplet_view    = triplet_pGpP.view();
    auto triplet_offsets = diff_parm_reporter_offsets_counts.offsets();
    for(auto&& [i, reporter] : enumerate(reporter_view))
    {
        auto                offset = triplet_offsets[i];
        auto                count  = triplet_counts[i];
        ContactDiffParmInfo info{this, triplet_view.subview(offset, count)};

        reporter->do_assemble(info);
    }
}

void GlobalDiffContactManager::Impl::convert_matrix()
{
    matrix_converter_H.convert(triplet_H, bcoo_H);
    matrix_converter_pGpP.convert(triplet_pGpP, bcoo_pGpP);
}

void GlobalDiffContactManager::Impl::distribute_diff_dof()
{
    using namespace muda;
    auto vertex_count = global_vertex_manager->positions().size();
    for(auto&& [i, receiver] : enumerate(diff_dof_receivers.view()))
    {
        ClassifyInfo info;
        receiver->do_report(info);

        const auto N = bcoo_H.triplet_count();

        auto& dstH = classified_H[i];
        dstH.reshape(vertex_count, vertex_count);

        if(!info.is_empty())
        {
            // +1 for calculate the total count
            loose_resize(selected_hessian, N + 1);
            loose_resize(selected_hessian_offsets, N + 1);

            // select
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [selected_hessian = selected_hessian.view(0, N).viewer().name("selected_hessian"),
                     last =
                         VarView<IndexT>{selected_hessian.data() + N}.viewer().name("last"),

                     bcoo_H  = bcoo_H.cviewer().name("bcoo_H"),
                     i_range = info.hessian_i_range(),
                     j_range = info.hessian_j_range()] __device__(int I) mutable
                    {
                        auto&& [i, j, H] = bcoo_H(I);

                        auto in_range = [](int i, const Vector2i& range)
                        { return i >= range.x() && i < range.y(); };

                        selected_hessian(I) =
                            in_range(i, i_range) && in_range(j, j_range) ? 1 : 0;

                        // fill the last one as 0, so that we can calculate the total count
                        // during the exclusive scan
                        if(I == 0)
                            last = 0;
                    });

            // scan
            DeviceScan().ExclusiveSum(selected_hessian.data(),
                                      selected_hessian_offsets.data(),
                                      selected_hessian.size());

            IndexT h_total_count = 0;
            VarView<IndexT>{selected_hessian_offsets.data() + N}.copy_to(&h_total_count);

            loose_resize_entries(dstH, h_total_count);

            // fill
            if(h_total_count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [selected_hessian = selected_hessian.cviewer().name("selected_hessian"),
                            selected_hessian_offsets =
                                selected_hessian_offsets.cviewer().name("selected_hessian_offsets"),
                            bcoo_H  = bcoo_H.cviewer().name("bcoo_H"),
                            dstH    = dstH.viewer().name("classified_hessian"),
                            i_range = info.hessian_i_range(),
                            j_range = info.hessian_j_range()] __device__(int I) mutable
                           {
                               if(selected_hessian(I))
                               {
                                   auto&& [i, j, H] = bcoo_H(I);
                                   auto offset = selected_hessian_offsets(I);

                                   dstH(offset).write(i, j, H);
                               }
                           });
            }
        }

        ClassifiedContactDiffDofInfo classified_info;
        classified_info.m_H = dstH.view();
        receiver->do_receive(classified_info);
    }
}

void GlobalDiffContactManager::Impl::distribute_diff_parm()
{
    // If there is no diff contact coeff, we don't need to assemble contact pGpP
    if(!diff_contact_coeff.total_size())
        return;

    using namespace muda;
    auto  vertex_count = global_vertex_manager->positions().size();
    SizeT parm_count   = global_diff_sim_manager->parameters().size();

    for(auto&& [i, receiver] : enumerate(diff_parm_receivers.view()))
    {
        ClassifyInfo info;
        receiver->report(info);

        auto& dst_pGpP = classified_pGpP[i];
        dst_pGpP.clear();
        dst_pGpP.reshape(vertex_count, parm_count);  // extent = (3 * V, P)

        if(!info.is_empty())
        {
            const auto N = bcoo_pGpP.triplet_count();

            // +1 for calculate the total count
            loose_resize(selected_pGpP, N + 1);
            loose_resize(selected_pGpP_offsets, N + 1);

            // select
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(N,
                       [selected_pGpP = selected_pGpP.view(0, N).viewer().name("selected_pGpP"),
                        last =
                            VarView<IndexT>{selected_pGpP.data() + N}.viewer().name("last"),

                        bcoo_pGpP = bcoo_pGpP.cviewer().name("bcoo_pGpP"),
                        i_range = info.hessian_i_range()] __device__(int I) mutable
                       {
                           auto&& [i, j, pGpP] = bcoo_pGpP(I);

                           auto in_range = [](int i, const Vector2i& range)
                           { return i >= range.x() && i < range.y(); };

                           selected_pGpP(I) = in_range(i, i_range) ? 1 : 0;

                           // fill the last one as 0, so that we can calculate the total count
                           // during the exclusive scan
                           if(I == 0)
                               last = 0;
                       });

            // scan
            DeviceScan().ExclusiveSum(selected_pGpP.data(),
                                      selected_pGpP_offsets.data(),
                                      selected_pGpP.size());

            IndexT pGpP_total_count = 0;
            VarView<IndexT>{selected_pGpP_offsets.data() + N}.copy_to(&pGpP_total_count);

            loose_resize_entries(dst_pGpP, pGpP_total_count);

            // fill
            if(pGpP_total_count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [selected_pGpP = selected_pGpP.cviewer().name("selected_pGpP"),
                            selected_pGpP_offsets =
                                selected_pGpP_offsets.cviewer().name("selected_pGpP_offsets"),
                            bcoo_pGpP = bcoo_pGpP.cviewer().name("bcoo_pGpP"),
                            dst_pGpP = dst_pGpP.viewer().name("classified_pGpP"),
                            i_range = info.gradient_i_range()] __device__(int I) mutable
                           {
                               if(selected_pGpP(I))
                               {
                                   auto&& [i, j, pGpP] = bcoo_pGpP(I);
                                   auto offset = selected_pGpP_offsets(I);

                                   dst_pGpP(offset).write(i, j, pGpP);
                               }
                           });
            }
        }


        ClassifiedContactDiffParmInfo classified_info;
        classified_info.m_pGpP = dst_pGpP.cview();
        receiver->receive(classified_info);
    }
}
}  // namespace uipc::backend::cuda