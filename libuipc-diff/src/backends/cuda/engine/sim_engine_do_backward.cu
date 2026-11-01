#include <sim_engine.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <linear_system/global_linear_system.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <contact_system/global_diff_contact_manager.h>

namespace uipc::backend::cuda
{
// The DiffSim Pipeline
// ref: https://github.com/spiriMirror/libuipc/issues/37

void SimEngine::do_backward()
{
    UIPC_ASSERT(false, "[Deprecated] SimEngine::do_backward() is deprecated. Use the Adjoint Method Feature instead.");

    ///***************************************************************************************
    //*                                  Function Shortcuts
    //***************************************************************************************/

    //auto detect_dcd_candidates = [this]
    //{
    //    if(m_global_trajectory_filter)
    //    {
    //        Timer timer{"Detect DCD Candidates"};
    //        m_global_trajectory_filter->detect(0.0);
    //        m_global_trajectory_filter->filter_active();
    //    }
    //};

    //auto compute_contact = [this]
    //{
    //    if(m_global_contact_manager)
    //    {
    //        Timer timer{"Compute Contact"};
    //        m_global_contact_manager->compute_contact();
    //    }
    //};

    //auto compute_diff_contact = [this]
    //{
    //    if(m_global_diff_contact_manager)
    //    {
    //        Timer timer{"Compute Contact Diff Dof"};
    //        m_global_diff_contact_manager->compute_diff_contact();
    //    }
    //};


    ///***************************************************************************************
    //*                                  Core Pipeline
    //***************************************************************************************/

    //try
    //{
    //    if(m_global_diff_sim_manager)
    //    {
    //        spdlog::info(R"(>>> Begin Backward: {})", m_current_frame);

    //        if(m_last_solved_frame != m_current_frame)
    //        {
    //            throw SimEngineException(  //
    //                fmt::format("The current frame({}) is not solved yet. The last solved frame is {}. "
    //                            "DiffSim depends on the simlulation result of the current frame. "
    //                            "Make sure the current frame is solved (rather than recovered) before calling `backward()`.",
    //                            m_current_frame,
    //                            m_last_solved_frame));
    //        }

    //        Timer timer{"Backward Pipeline"};

    //        // 1) detect the DCD candidates and compute the contact
    //        detect_dcd_candidates();
    //        compute_contact();

    //        // 2) prepare the global hessian matrix for current frame: H^{[k][k]}
    //        {
    //            Timer timer{"Build Current Frame Hessian"};
    //            m_global_linear_system->prepare_hessian();
    //        }

    //        // 3) compute the diff contact: H and pGpP
    //        compute_diff_contact();

    //        // 4) assemble H and pGpP: H^{[k][k-i]}, i=1,2,3,...
    //        {
    //            Timer timer{"Assemble Diff Dof&Parm"};
    //            m_global_diff_sim_manager->assemble();
    //        }

    //        spdlog::info(R"(<<< End Backward: {})", m_current_frame);
    //    }
    //    else
    //    {
    //        spdlog::warn("DiffSim is disabled. Backward does nothing.");
    //    }
    //}
    //catch(const SimEngineException& e)
    //{
    //    spdlog::error("SimEngine Backward Error: {}", e.what());
    //    status().push_back(core::EngineStatus::error(e.what()));
    //}
}

}  // namespace uipc::backend::cuda
