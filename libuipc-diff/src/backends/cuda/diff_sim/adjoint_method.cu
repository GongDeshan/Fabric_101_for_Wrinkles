#include <diff_sim/adjoint_method.h>
#include <uipc/common/log.h>
#include <uipc/diff_sim/adjoint_method_feature.h>  // base feature
#include <diff_sim/adjoint_method_feature.h>       // overridden feature
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <diff_sim/global_adjoint_method_replayer.h>
#include <collision_detection/global_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <contact_system/global_diff_contact_manager.h>
#include <gradient_hessian_computer.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(AdjointMethod);

void AdjointMethod::do_build()
{
    m_impl.global_linear_system      = &require<GlobalLinearSystem>();
    m_impl.global_diff_sim_manager   = &require<GlobalDiffSimManager>();
    m_impl.gradient_hessian_computer = &require<GradientHessianComputer>();

    m_impl.global_adjoint_method_replayer = &require<GlobalAdjointMethodReplayer>();
    m_impl.global_trajectory_filter    = find<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager      = find<GlobalContactManager>();
    m_impl.global_diff_contact_manager = find<GlobalDiffContactManager>();
    // EXPERIMENT (2026-09-20): omit contact derivatives ONLY in adjoint.
    // Default true preserves the original complete assembly path.
    m_impl.contact_gradient =
        world().scene().info()["diff_sim"].value("contact_gradient", true);
    if(!m_impl.contact_gradient)
        spdlog::warn("[adjoint-contact-stop-v1] Contact derivatives OFF in adjoint; "
                     "forward collision ON if configured. Material/inertia derivatives retained. "
                     "This is an approximate adjoint, NOT a full simulator derivative.");


    using uipc::diff_sim::AdjointMethodFeature;
    auto overrider = uipc::make_shared<AdjointMethodFeatureOverrider>(*this);
    auto feature   = uipc::make_shared<AdjointMethodFeature>(overrider);    
    features().insert(feature);  // insert the overridden feature
}

// y = alpha * A^T * x + y
template <int M, int N>
void spmTv(muda::DenseVectorView<Float>          y,
           Float                                 alpha,
           muda::CTripletMatrixView<Float, M, N> A,
           muda::CDenseVectorView<Float>         x,
           IndexT                                x_offset);

void fill_dLdX(muda::DenseVectorView<Float> f,
               muda::CBufferView<IndexT>    SDI,
               muda::CBufferView<Float>     dLdx);

void append_with_offset(muda::DeviceBuffer<IndexT>& dst, muda::CBufferView<IndexT> src, IndexT offset);

static muda::DenseVectorView<Float> from_buffer_view(muda::BufferView<Float> view);

void AdjointMethod::select_dofs(SizeT frame, muda::CBufferView<IndexT> in_SDI)
{
    auto current_frame = engine().frame();
    UIPC_ASSERT(frame == current_frame, "Frame mismatch, expected {}, yours {}.", current_frame, frame);

    if(frame == 1)  // clear the dofs and SDI at the first frame
    {
        m_impl.dofs.clear();
        m_impl.SDI.clear();
    }

    if(in_SDI.size() == 0)
        return;

    Timer timer{"Select Dofs"};

    {  // Append Selected Dofs to m_impl.dofs
        auto offset = m_impl.dofs.size();
        m_impl.dofs.resize(m_impl.dofs.size() + in_SDI.size());
        auto dst_dofs = m_impl.dofs.view().subview(offset, in_SDI.size());
        m_impl.global_linear_system->collect_dofs(dst_dofs, in_SDI);
    }

    {  // Append Selected Dof Indices to m_impl.SDI (with frame-based offset)
        auto dof_offset = m_impl.global_diff_sim_manager->dof_offset(frame);
        append_with_offset(m_impl.SDI, in_SDI, dof_offset);
    }
}

void AdjointMethod::receive_dofs(muda::BufferView<Float> out_dofs)
{
    UIPC_ASSERT(out_dofs.size() == m_impl.dofs.size(),
                "Size mismatch, expected dofs {}, yours {}",
                m_impl.dofs.size(),
                out_dofs.size());

    Timer timer{"Receive Dofs"};

    UIPC_ASSERT(m_impl.dofs.size() == m_impl.SDI.size(),
                "Selected Dofs and Selected Dof Indices size mismatch, dofs {}, SDI {}, why can it happen?",
                m_impl.dofs.size(),
                m_impl.SDI.size());

    out_dofs.copy_from(m_impl.dofs);
}

void AdjointMethod::compute_dLdP(muda::BufferView<Float>  out_dLdP,
                                 muda::CBufferView<Float> in_dLdx)
{
    /***************************************************************************************
    *                                  Function Shortcuts
    ***************************************************************************************/

    auto dof_count = [&](SizeT N)
    { return m_impl.global_adjoint_method_replayer->dof_count(N); };

    auto dof_offset = [&](SizeT N)
    { return m_impl.global_adjoint_method_replayer->dof_offset(N); };

    auto replay = [&](SizeT frame)
    { m_impl.global_adjoint_method_replayer->replay(frame); };

    auto solve_adjoint = [&](muda::DenseVectorView<Float>  lambda_i,
                             muda::CDenseVectorView<Float> f)
    { m_impl.global_linear_system->solve_adjoint_variable(lambda_i, f); };

    auto Hij   = [&] { return m_impl.global_diff_sim_manager->Hij(); };
    auto pGipP = [&] { return m_impl.global_diff_sim_manager->pGipP(); };

    /***************************************************************************************
    *                                  Core Pipeline
    ***************************************************************************************/

    auto& f        = m_impl.f;
    auto& lambda_i = m_impl.lambda_i;
    auto& SDI      = m_impl.SDI;
    auto  dLdP     = from_buffer_view(out_dLdP);

    Timer timer{"Adjoint Method"};

    auto N = engine().frame();
    // Diagnostic only; production defaults preserve the complete backward loop.
    // Caller must also freeze prefix X/v/history in its FD experiment. For
    // history-dependent parameters, cached history sensitivities need separate
    // treatment: stopping the temporal adjoint alone is NOT enough.
    const auto& diff_info = world().scene().info()["diff_sim"];
    const bool audit_single_step = diff_info.value("audit_single_step", false);
    const std::string audit_directory = diff_info.value("audit_hessian_directory", std::string{});
    if(audit_single_step)
        spdlog::warn("[buckling-single-step-v1] Single-frame projected adjoint at {}; "
                     "prefix sensitivities are intentionally excluded", N);

    UIPC_ASSERT(N >= 1, "requires frame >= 1, yours {}.", N);
    auto dof_offset_N = dof_offset(N);
    auto dof_count_N  = dof_count(N);
    auto total_dof    = dof_count_N + dof_offset_N;

    // 1. f = -dLdx
    {
        Timer timer{"Fill dLdX"};

        f.resize(total_dof);
        f.fill(0.0);
        fill_dLdX(f, SDI, in_dLdx);
    }
    // 2. dLdP = 0
    out_dLdP.fill(0.0);

    // 3. Replay from frame N to 1
    for(SizeT i = N; i >= 1; --i)
    {
        spdlog::info(R"(>>> Begin Backward: {})", i);

        auto dof_count_i  = dof_count(i);
        auto dof_offset_i = dof_offset(i);
        // 1) lambda_i = 0
        {
            lambda_i.resize(dof_count_i);
            lambda_i.fill(0.0);
        }

        // 2) replay to frame i
        replay(i);

        if(i == N && !audit_directory.empty())
        {
            // Contact state and animation constraints are assembled by the
            // ordinary path first. Export reuses them at precisely this frame.
            assemble(i);
            const auto prefix = fmt::format("{}/frame_{}", audit_directory, i);
            m_impl.global_linear_system->dump_diagnostic_hessian(prefix + "_projected", false);
            m_impl.global_linear_system->dump_diagnostic_hessian(prefix + "_material_raw", true);
            // Both exports restore flags; the ordinary assemble/solve below
            // rebuilds projected material Hessians and its preconditioner.
        }

        // 3) assemble H[ii], H[ij] and pG[i]pP
        {
            Timer timer{"Assemble System"};
            assemble(i);
        }

        // 4) solve lambda_i: // H[ii] * lambda[i] = f[i]
        {
            Timer timer{"Solve Adjoint"};
            auto  f_i = f.view().subview(dof_offset_i, dof_count_i);

            // local PCG
            solve_adjoint(lambda_i, f_i);
        }


        // 5) update f: (spmv)
        // f = f - H[ij]^T * lambda_i
        {
            Timer timer{"Update Adjoint System Rhs"};
            auto  HijV = Hij();
            spmTv<3, 3>(f.view(), -1.0, HijV, lambda_i, dof_offset_i);
        }

        // 6) update dLdP: (spmv)
        // dLdP = dLdP + pG[i]pP^T * lambda_i
        {
            Timer timer{"Uptate dLdP"};
            auto  pGipPV = pGipP();
            spmTv<3, 1>(dLdP, 1.0, pGipPV, lambda_i, dof_offset_i);
        }

        //std::vector<double> h_dLdP;
        //h_dLdP.resize(dLdP.size());

        //dLdP.buffer_view().copy_to(h_dLdP.data());

        //std::cout << "dLdP" << std::endl;

        //for(auto d_v : h_dLdP)
        //    std::cout << d_v << std::endl;
 
        spdlog::info(R"(<<< End Backward: {})", i);
        if(audit_single_step)
        {
            spdlog::info("[buckling-single-step-v1] Stopped after frame {}", i);
            break;
        }
    }
}

void AdjointMethod::assemble(SizeT frame)
{
    /***************************************************************************************
    *                                  Function Shortcuts
    ***************************************************************************************/

    auto detect_dcd_candidates = [this]
    {
        if(m_impl.global_trajectory_filter)
        {
            Timer timer{"Detect DCD Candidates"};
            m_impl.global_trajectory_filter->detect(0.0);
            m_impl.global_trajectory_filter->filter_active();
        }
    };

    auto compute_contact = [this]
    {
        if(m_impl.global_contact_manager)
        {
            Timer timer{"Compute Contact"};
            m_impl.global_contact_manager->compute_contact();
        }
    };

    auto compute_diff_contact = [this]
    {
        if(m_impl.global_diff_contact_manager)
        {
            Timer timer{"Compute Contact Diff Dof"};
            m_impl.global_diff_contact_manager->compute_diff_contact();
        }
    };

    /***************************************************************************************
    *                                  Core Pipeline
    ***************************************************************************************/

    try
    {
        // 1) detect the DCD candidates and compute the contact
        // OLD/default path: detect_dcd_candidates(); compute_contact();
        if(m_impl.contact_gradient)
        {
            detect_dcd_candidates();
            compute_contact();
        }
        else if(m_impl.global_contact_manager)
        {
            // Publish fresh empty views: merely skipping compute leaves stale H[ii].
            m_impl.global_contact_manager->clear_contact_for_adjoint();
        }

        // 2) compute gradient and hessian for current frame
        m_impl.gradient_hessian_computer->compute_gradient_hessian();

        // 3) compute the diff contact: H and pGpP
        // OLD/default path: compute_diff_contact();
        if(m_impl.contact_gradient)
            compute_diff_contact();
        else if(m_impl.global_diff_contact_manager)
            m_impl.global_diff_contact_manager->clear_contact_for_adjoint();

        // 5) assemble H and pGpP: H[ij] and pG[i]pP
        {
            Timer timer{"Assemble Diff Dof&Parm"};
            m_impl.global_diff_sim_manager->assemble(frame);
        }
    }
    catch(const SimEngineException& e)
    {
        spdlog::error("SimEngine Backward Error: {}", e.what());
        engine().status().push_back(core::EngineStatus::error(e.what()));
    }
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
// y = alpha * A^T * x + y
template <int M, int N>
void spmTv(muda::DenseVectorView<Float>          y,
           Float                                 alpha,
           muda::CTripletMatrixView<Float, M, N> A,
           muda::CDenseVectorView<Float>         x,
           IndexT                                x_offset)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(A.triplet_count(),
               [y        = y.viewer().name("y"),
                A        = A.viewer().name("A"),
                alpha    = alpha,
                x        = x.viewer().name("x"),
                x_offset = x_offset] __device__(int I)
               {
                   //
                   auto&& [i, j, Aij] = A(I);

                   // y[j] += alpha * {Aij}^T * x[i]
                   Vector<Float, N> V = alpha * Aij.transpose()
                                        * x.segment<M>(M * i - x_offset).as_eigen();

                   y.segment<N>(N * j).atomic_add(V);
               });
}

// f = -dLdx
void fill_dLdX(muda::DenseVectorView<Float> f,
               muda::CBufferView<IndexT>    SDI,
               muda::CBufferView<Float>     dLdx)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(SDI.size(),
               [f    = f.viewer().name("f"),
                SDI  = SDI.viewer().name("SDI"),
                dLdx = dLdx.viewer().name("dLdx")] __device__(int I)
               {
                   auto i = SDI(I);    // dof index
                   f(i)   = -dLdx(I);  // value
               });
}


void append_with_offset(muda::DeviceBuffer<IndexT>& dst, muda::CBufferView<IndexT> src, IndexT offset)
{
    auto dst_size = dst.size();
    dst.resize(dst_size + src.size());
    auto dst_view = dst.view().subview(dst_size, src.size());

    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(src.size(),
               [dst_view = dst_view.viewer().name("dst_view"),
                src      = src.viewer().name("src"),
                offset] __device__(int I)
               {
                   //
                   dst_view(I) = src(I) + offset;
               });
}

static muda::DenseVectorView<Float> from_buffer_view(muda::BufferView<Float> view)
{
    return muda::DenseVectorView<Float>{view.origin_data(),
                                        nullptr,
                                        (int)view.offset(),
                                        1,
                                        (int)view.size(),
                                        (int)view.size()};
}


}  // namespace uipc::backend::cuda
