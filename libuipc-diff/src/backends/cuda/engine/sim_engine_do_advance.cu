#include <sim_engine.h>
#include <uipc/common/range.h>
#include <dof_predictor.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <line_search/line_searcher.h>
#include <line_search/line_search_reporter.h>
#include <finite_element/fem_line_search_reporter.h>
#include <gradient_hessian_computer.h>
#include <linear_system/global_linear_system.h>
#include <animator/global_animator.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <contact_system/global_diff_contact_manager.h>
#include <diff_sim/global_adjoint_method_replayer.h>
#include <cstdlib>
#include <limits>

namespace uipc::backend::cuda
{
void SimEngine::do_advance()
{
    Float alpha     = 1.0;
    Float ccd_alpha = 1.0;
    Float cfl_alpha = 1.0;
    const char* audit_env = std::getenv("UIPC_NEWTON_AUDIT");
    const bool audit_newton = audit_env && audit_env[0] == '1';
    // Opt-in until the before/after regression checks pass. Unset/0 keeps
    // legacy behavior for rollback. This is NOT a negative-curvature solver.
    const char* safe_env = std::getenv("UIPC_NEWTON_SAFE_LINE_SEARCH");
    const bool safe_newton = safe_env && safe_env[0] == '1';
    const char* curvature_env = std::getenv("UIPC_NEWTON_CURVATURE_AUDIT");
    const bool audit_curvature = curvature_env && curvature_env[0] == '1';
    // Signed finite-difference probes bypass CCD. Limit this diagnostic to
    // contact-disabled early frames, never production collision rollouts.
    if(audit_curvature && world().scene().info()["contact"]["enable"].get<bool>())
        throw SimEngineException("Curvature audit requires contact disabled");

    bool dump_surface =
        world().scene().info()["extras"]["debug"]["dump_surface"].get<bool>();

    /***************************************************************************************
    *                                  Function Shortcuts
    ***************************************************************************************/

    auto detect_dcd_candidates = [this]
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect DCD Candidates"};
            m_global_trajectory_filter->detect(0.0);
            m_global_trajectory_filter->filter_active();
        }
    };

    auto detect_trajectory_candidates = [this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect Trajectory Candidates"};
            m_global_trajectory_filter->detect(alpha);
        }
    };

    auto filter_dcd_candidates = [this]
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter Contact Candidates"};
            m_global_trajectory_filter->filter_active();
        }
    };

    auto record_friction_candidates = [this]
    {
        if(m_global_trajectory_filter && m_friction_enabled)
        {
            m_global_trajectory_filter->record_friction_candidates();
        }
    };

    auto compute_adaptive_kappa = [this]
    {
        // TODO: now no effect
        if(m_global_contact_manager)
            m_global_contact_manager->compute_adaptive_kappa();
    };

    auto compute_contact = [this]
    {
        if(m_global_contact_manager)
        {
            Timer timer{"Compute Contact"};
            m_global_contact_manager->compute_contact();
        }
    };

    auto cfl_condition = [&cfl_alpha, this](Float alpha)
    {
        if(m_global_contact_manager)
        {
            cfl_alpha = m_global_contact_manager->compute_cfl_condition();
            if(cfl_alpha < alpha)
            {
                spdlog::info("CFL Filter: {} < {}", cfl_alpha, alpha);
                return cfl_alpha;
            }
        }

        return alpha;
    };

    auto filter_toi = [&ccd_alpha, this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter CCD TOI"};
            ccd_alpha = m_global_trajectory_filter->filter_toi(alpha);
            if(ccd_alpha < alpha)
            {
                spdlog::info("CCD Filter: {} < {}", ccd_alpha, alpha);
                return ccd_alpha;
            }
        }

        return alpha;
    };

    auto compute_energy = [this, filter_dcd_candidates](Float alpha) -> Float
    {
        // Step Forward => x = x_0 + alpha * dx
        m_global_vertex_manager->step_forward(alpha);
        m_line_searcher->step_forward(alpha);

        // Update the collision pairs
        filter_dcd_candidates();

        // Compute New Energy => E
        return m_line_searcher->compute_energy(false);
    };

    auto step_animation = [this]()
    {
        if(m_global_animator)
        {
            Timer timer{"Step Animation"};
            m_global_animator->step();
        }
    };

    auto compute_animation_substep_ratio = [this](SizeT newton_iter)
    {
        // compute the ratio to the aim position.
        // dst = prev_position + ratio * (position - prev_position)
        if(m_global_animator)
        {
            m_global_animator->compute_substep_ratio(newton_iter);
            spdlog::info("Animation Substep Ratio: {}", m_global_animator->substep_ratio());
        }
    };

    auto animation_reach_target = [this]()
    {
        if(m_global_animator)
        {
            return m_global_animator->substep_ratio() >= 1.0;
        }
        return true;
    };

    auto record_state = [this]()
    {
        if(m_global_diff_sim_manager && m_global_adjoint_method_replayer)
        {
            bool enable = m_global_diff_sim_manager->is_grad_enabled();
            m_global_adjoint_method_replayer->record(enable, m_current_frame);
        }
    };

    /***************************************************************************************
    *                                  Core Pipeline
    ***************************************************************************************/

    // Abort on exception if the runtime check is enabled for debugging
    constexpr bool AbortOnException = uipc::RUNTIME_CHECK;

    auto pipeline = [&]() noexcept(AbortOnException)
    {
        Timer timer{"Advance Pipeline"};

        ++m_current_frame;

        spdlog::info(R"(>>> Begin Frame: {})", m_current_frame);

        // Rebuild Scene
        {
            Timer timer{"Rebuild Scene"};
            // Trigger the rebuild_scene event, systems register their actions will be called here
            m_state = SimEngineState::RebuildScene;
            {
                event_rebuild_scene();

                // TODO: rebuild the vertex and surface info
                // m_global_vertex_manager->rebuild_vertex_info();
                // m_global_surface_manager->rebuild_surface_info();
            }

            // After the rebuild_scene event, the pending creation or deletion can be solved
            world().scene().solve_pending();
        }

        // Simulation:
        {
            Timer timer{"Simulation"};
            // 1. Update Velocity
            if(frame() > 1)
            {
                // Skip the first frame, since the velocity is the initial velocity
                // We don't want to modify the initial velocity
                m_state = SimEngineState::UpdateVelocity;
                {
                    Timer timer{"Update Velocity"};
                    m_dof_predictor->compute_velocity();
                }
            }

            // 2. Adaptive Parameter Calculation
            AABB vertex_bounding_box =
                m_global_vertex_manager->compute_vertex_bounding_box();
            detect_dcd_candidates();
            compute_adaptive_kappa();

            // 3. Record Friction Candidates at the beginning of the frame
            record_friction_candidates();
            m_global_vertex_manager->record_prev_positions();

            // 4. Predict Motion => x_tilde = x + v * dt
            m_state = SimEngineState::PredictMotion;
            m_dof_predictor->predict();
            step_animation();

            // 5. Nonlinear-Newton Iteration
            Float box_size = vertex_bounding_box.diagonal().norm();
            Float tol      = m_newton_scene_tol * box_size;
            Float res0     = 0.0;
            Float gradient_norm0 = 0.0;
            SizeT precision_streak = 0;
            bool precision_limited = false;
            const char* exit_reason = "iteration_limit";

            SizeT newton_iter = 0;
            for(; newton_iter < m_newton_max_iter; ++newton_iter)
            {
                Timer timer{"Newton Iteration"};

                // 1) Compute animation substep ratio
                compute_animation_substep_ratio(newton_iter);

                // 2) Build Collision Pairs
                if(newton_iter > 0)
                    detect_dcd_candidates();

                // 3) Compute Contact Gradient and Hessian => G:Vector3, H:Matrix3x3
                m_state = SimEngineState::ComputeContact;
                compute_contact();

                // 4) Compute System Gradient and Hessian
                m_state = SimEngineState::ComputeGradientHessian;
                {
                    // If some systems need to precompute some G/H before assembling

                    Timer timer{"Compute Gradient Hessian"};
                    m_gradient_hessian_computer->compute_gradient_hessian();
                }

                // 5) Solve Global Linear System => dx = A^-1 * b
                m_state = SimEngineState::SolveGlobalLinearSystem;
                {
                    // - Assemble the linear system
                    // - Solve the linear system
                    // - Distribute the solution

                    Timer timer{"Solve Global Linear System"};
                    m_global_linear_system->solve();
                }

                // 6) Get Max Movement => dx_max = max(|dx|), if dx_max < tol, break
                m_global_vertex_manager->collect_vertex_displacements();
                Float res = m_global_vertex_manager->compute_axis_max_displacement();
                Float gradient_norm = 0.0;
                Float projected_decrease = 0.0;
                if(safe_newton)
                {
                    // SimEngine is a friend of GlobalLinearSystem. Its forward
                    // b is the assembled gradient, solution is -displacement.
                    auto& system = m_global_linear_system->m_impl;
                    if(!system.empty_system)
                    {
                        gradient_norm = system.ctx.norm(system.b.cview());
                        projected_decrease = 0.5 * system.ctx.dot(system.b.cview(), system.x.cview());
                    }
                    if(newton_iter == 0)
                        gradient_norm0 = gradient_norm;
                }

                // 7) Check Termination Condition
                // TODO: Maybe we can implement a class for termination condition in the future
                bool converged = false;
                {
                    if(newton_iter == 0)
                        res0 = res;  // record the initial residual

                    Float rel_tol = res == 0.0 ? 0.0 : res / res0;

                    spdlog::info(">> Frame {} Newton Iteration {} => Residual/AbsTol/CCDToi: {}/{}/{}",
                                 m_current_frame,
                                 newton_iter,
                                 res,
                                 m_abs_tol,
                                 ccd_alpha);

                    // Old: converged = res <= m_abs_tol || rel_tol <= 0.001;
                    // A zero configured threshold disables the relative shortcut.
                    converged = res <= m_abs_tol
                                || (m_newton_relative_tol > 0.0
                                    && rel_tol <= m_newton_relative_tol);

                    if(dump_surface)
                    {
                        dump_global_surface(fmt::format(
                            "dump_surface.{}.{}", m_current_frame, newton_iter));
                    }

                    if(newton_iter > 0 && converged && ccd_alpha >= m_ccd_tol
                       && animation_reach_target())
                    {
                        exit_reason = "step_tolerance";
                        break;
                    }
                }


                // 8) Begin Line Search
                m_state = SimEngineState::LineSearch;
                {
                    Timer timer{"Line Search"};

                    // Reset Alpha
                    alpha = 1.0;

                    // Record Current State x to x_0
                    m_line_searcher->record_start_point();
                    m_global_vertex_manager->record_start_point();
                    detect_trajectory_candidates(alpha);

                    // Compute Current Energy => E_0
                    Float E0 = m_line_searcher->compute_energy(true);  // initial energy
                    // spdlog::info("Initial Energy: {}", E0);

                    if(audit_curvature && m_current_frame == 2 && res > 0.0
                       && (newton_iter == 5 || newton_iter == 20
                           || newton_iter == 50 || newton_iter == 900))
                    {
                        // Probe the ORIGINAL energy, not the SPD-projected Hessian.
                        // d=dx/max_axis(|dx|); s below has units of metres.
                        // Histories and animator targets remain frozen in line search.
                        const Float center = compute_energy(0.0);
                        // Diagnostic only: b=gradient, x=-dx. For an accurate
                        // linear solve x^T b approximates x^T A_projected x.
                        // Compare on the SAME d=dx/res used by the energy probes.
                        auto& audit_system = m_global_linear_system->m_impl;
                        const Float direction_norm_sq = audit_system.ctx.dot(
                            audit_system.x.cview(), audit_system.x.cview()) / (res * res);
                        const Float projected_curvature_estimate = audit_system.ctx.dot(
                            audit_system.b.cview(), audit_system.x.cview()) / (res * res);
                        const Float predicted_slope = -projected_curvature_estimate * res;
                        const auto center_components = m_line_searcher->m_energy_values;
                        auto audit_fem = m_line_searcher->m_reporters.find<FEMLineSearchReporter>();
                        const auto snapshot_fem = [&]()
                        {
                            return audit_fem ? audit_fem->audit_energy_components()
                                : vector<FEMLineSearchReporter::AuditEnergyComponent>{};
                        };
                        const auto center_fem_components = snapshot_fem();
                        // Preserve reporter order; components are system energies,
                        // NOT necessarily individual constitutions/hinges.
                        vector<std::string> component_names;
                        for(auto* reporter : m_line_searcher->m_reporters.view())
                            component_names.emplace_back(reporter->name());
                        for(const auto& name : m_line_searcher->m_energy_reporter_names)
                            component_names.emplace_back(name);
                        spdlog::info("[newton-projection-audit-v1] frame={} newton={} projected_curvature_estimate={} direction_norm_sq={} predicted_slope={}",
                                     m_current_frame, newton_iter,
                                     projected_curvature_estimate, direction_norm_sq, predicted_slope);
                        for(const Float step : {Float(1e-7), Float(3e-7),
                                                Float(1e-6), Float(3e-6)})
                        {
                            const Float scale = step / res;
                            const Float minus = compute_energy(-scale);
                            const auto minus_components = m_line_searcher->m_energy_values;
                            const auto minus_fem_components = snapshot_fem();
                            const Float plus = compute_energy(scale);
                            const auto plus_components = m_line_searcher->m_energy_values;
                            const auto plus_fem_components = snapshot_fem();
                            const Float second_difference = (plus - center) + (minus - center);
                            const Float restored = compute_energy(0.0);
                            const auto restored_fem_components = snapshot_fem();
                            spdlog::info("[newton-curvature-audit-v1] frame={} newton={} step_m={} Ecenter={} Eminus={} Eplus={} second_difference={} directional_curvature={} directional_slope={} restored_minus_center={} epsilon_energy_scale={}",
                                         m_current_frame, newton_iter, step,
                                         center, minus, plus, second_difference,
                                         second_difference / (step * step),
                                         (plus - minus) / (2.0 * step),
                                         restored - center,
                                         std::numeric_limits<Float>::epsilon() * std::abs(center));
                            for(SizeT component = 0; component < center_components.size(); ++component)
                            {
                                const Float component_second_difference =
                                    (plus_components[component] - center_components[component])
                                    + (minus_components[component] - center_components[component]);
                                spdlog::info("[newton-component-curvature-v1] frame={} newton={} step_m={} component={} name={} directional_curvature={} second_difference={} restored_minus_center={}",
                                             m_current_frame, newton_iter, step, component,
                                             component_names[component],
                                             component_second_difference / (step * step),
                                             component_second_difference,
                                             m_line_searcher->m_energy_values[component] - center_components[component]);
                            }
                            // Producer-level partition distinguishes bending,
                            // membrane, kinetic and animator energies. Keep this
                            // within the contact-off curvature probe guard.
                            UIPC_ASSERT(center_fem_components.size() == minus_fem_components.size()
                                        && center_fem_components.size() == plus_fem_components.size()
                                        && center_fem_components.size() == restored_fem_components.size(),
                                        "FEM curvature audit: component count changed");
                            long double sum_second_difference = 0.0L;
                            for(SizeT component = 0; component < center_fem_components.size(); ++component)
                            {
                                const auto& base = center_fem_components[component];
                                const Float component_second_difference =
                                    (plus_fem_components[component].energy - base.energy)
                                    + (minus_fem_components[component].energy - base.energy);
                                sum_second_difference += component_second_difference;
                                spdlog::info("[newton-fem-curvature-v1] frame={} newton={} step_m={} component={} name={} Ecenter={} Eminus={} Eplus={} directional_curvature={} second_difference={} restored_minus_center={}",
                                             m_current_frame, newton_iter, step, component, base.name,
                                             base.energy, minus_fem_components[component].energy,
                                             plus_fem_components[component].energy,
                                             component_second_difference / (step * step),
                                             component_second_difference,
                                             restored_fem_components[component].energy - base.energy);
                            }
                            if(audit_fem)
                            {
                                Float fem_total_second_difference = 0.0;
                                for(SizeT component = 0; component < component_names.size(); ++component)
                                    if(component_names[component] == std::string(audit_fem->name()))
                                        fem_total_second_difference =
                                            (plus_components[component] - center_components[component])
                                            + (minus_components[component] - center_components[component]);
                                spdlog::info("[newton-fem-curvature-closure-v1] frame={} newton={} step_m={} sum_second_difference={} fem_second_difference={} partition_error={}",
                                             m_current_frame, newton_iter, step,
                                             static_cast<Float>(sum_second_difference),
                                             fem_total_second_difference,
                                             static_cast<Float>(sum_second_difference) - fem_total_second_difference);
                            }
                        }
                        // Restore candidate caches for the normal +dx line search.
                        detect_trajectory_candidates(alpha);
                    }

                    // CCD filter
                    alpha = filter_toi(alpha);

                    // CFL Condition
                    alpha = cfl_condition(alpha);

                    // Compute Test Energy => E
                    Float E  = compute_energy(alpha);
                    Float E1 = E;
                    SizeT audit_backtracks = 0;
                    bool rolled_back = false;

                    if(!converged || safe_newton)
                    {
                        SizeT line_search_iter = 0;
                        while(line_search_iter < m_line_searcher->max_iter())
                        {
                            Timer timer{"Line Search Iteration"};

                            bool energy_decrease = E <= E0;  // Check Energy Decrease

                            // TODO: Inversion Check (Not Implemented Yet)
                            bool no_inversion = true;

                            bool success = energy_decrease && no_inversion;
                            if(success)
                                break;

                            // If not success, then shrink alpha
                            alpha /= 2;
                            E = compute_energy(alpha);

                            line_search_iter++;
                            audit_backtracks = line_search_iter;
                        }

                        // Old '> max_iter' was unreachable. Check the final trial
                        // too: it may succeed on the last permitted reduction.
                        if(line_search_iter >= m_line_searcher->max_iter() && E > E0)
                        {
                            //m_global_linear_system->dump_linear_system(
                            //    fmt::format("{}.{}.{}", workspace(), frame(), newton_iter));

                            spdlog::warn(
                                "Line Search Exits with Max Iteration: {} (Frame={}, Newton={})\n"
                                "E/E0: {}, E1/E0: {}, E0:{}",
                                m_line_searcher->max_iter(),
                                m_current_frame,
                                newton_iter,
                                E / E0,
                                E1 / E0,
                                E0);

                            if(safe_newton)
                            {
                                // Old: a rejected final trial remained in xs.
                                // Restore BOTH FEM/ABD and global vertex positions,
                                // and refresh energy/contact caches at that point.
                                const Float rejected_energy = E;
                                E = compute_energy(0.0);
                                alpha = 0.0;
                                rolled_back = true;
                                spdlog::warn("[newton-safe-step-v1] frame={} newton={} line_search=rollback rejected_delta_E={} restored_delta_E={}",
                                             m_current_frame, newton_iter,
                                             rejected_energy - E0, E - E0);
                            }

                            if(m_strict_mode)
                            {
                                throw SimEngineException("StrictMode: Line Search Exits with Max Iteration");
                            }
                        }
                    }
                    if(safe_newton)
                    {
                        // Conservative heuristic for a numerical precision limit,
                        // NOT proof of a local minimum or an accurate adjoint.
                        // Never use abs(energy) floored at 1: our E is ~1e-8.
                        const Float energy_scale = std::max(std::abs(E0), std::abs(E));
                        const Float energy_budget = 8.0 * std::numeric_limits<Float>::epsilon() * energy_scale;
                        const Float step_floor = std::max(100.0 * m_abs_tol,
                            64.0 * std::numeric_limits<Float>::epsilon() * box_size);
                        const bool small_gradient = gradient_norm0 > 0.0
                            && gradient_norm <= 1e-8 * gradient_norm0;
                        const bool tiny_prediction = std::isfinite(projected_decrease)
                            && projected_decrease >= 0.0 && projected_decrease <= energy_budget;
                        const bool precision_candidate = newton_iter > 0
                            && std::isfinite(res) && res <= step_floor
                            && std::isfinite(gradient_norm) && small_gradient
                            && tiny_prediction && std::abs(E - E0) <= energy_budget
                            && ccd_alpha >= m_ccd_tol && animation_reach_target();
                        precision_streak = precision_candidate ? precision_streak + 1 : 0;
                        if(audit_newton || precision_streak > 0 || rolled_back)
                            spdlog::info("[newton-safe-step-v1] frame={} newton={} gradient_norm={} gradient_relative={} projected_decrease={} energy_budget={} step_floor={} precision_streak={}",
                                         m_current_frame, newton_iter, gradient_norm,
                                         gradient_norm0 > 0.0 ? gradient_norm / gradient_norm0 : 0.0,
                                         projected_decrease, energy_budget, step_floor, precision_streak);
                        if(precision_streak >= 5)
                        {
                            precision_limited = true;
                            exit_reason = "numerical_precision_limit";
                        }
                    }
                    if(audit_newton
                       && (newton_iter < 10 || newton_iter % 50 == 0
                           || E > E0 || newton_iter + 1 == m_newton_max_iter))
                    {
                        // Energy comparisons are recorded without altering acceptance.
                        // epsilon*|E0| is a scale estimate, not an error bound.
                        spdlog::info("[newton-energy-audit-v1] frame={} newton={} dx_max={} alpha={} backtracks={} E0={} E1={} Efinal={} delta_E={} epsilon_energy_scale={} energy_nonincrease={}",
                                     m_current_frame, newton_iter, res, alpha,
                                     audit_backtracks, E0, E1, E, E - E0,
                                     std::numeric_limits<Float>::epsilon() * std::abs(E0),
                                     E <= E0);
                    }
                }
                if(precision_limited)
                    break;
            }

            if(safe_newton)
                spdlog::info("[newton-safe-step-v1] frame={} exit_reason={} strict_convergence_certified=false",
                             m_current_frame, exit_reason);
            if(precision_limited)
            {
                spdlog::warn("Newton reached numerical precision limit (Frame={}, Newton={}); this is NOT certified equilibrium/minimum convergence",
                             m_current_frame, newton_iter);
                status().push_back(core::EngineStatus::warning("Newton reached numerical precision limit; equilibrium/minimum not certified"));
                if(m_strict_mode)
                    throw SimEngineException("StrictMode: Newton reached numerical precision limit");
            }

            // 6. Record State for DiffSim
            record_state();


            // The '< max_iter' loop ends at equality, never greater-than.
            if(newton_iter >= m_newton_max_iter)
            {
                spdlog::warn("Newton Iteration Exits with Max Iteration: {} (Frame={})",
                             m_newton_max_iter,
                             m_current_frame);

                if(m_strict_mode)
                {
                    throw SimEngineException("StrictMode: Newton Iteration Exits with Max Iteration");
                }
            }
        }

        spdlog::info("<<< End Frame: {}", m_current_frame);
    };

    try
    {
        pipeline();
        m_last_solved_frame = m_current_frame;
    }
    catch(const SimEngineException& e)
    {
        spdlog::error("Engine Advance Error: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
}
}  // namespace uipc::backend::cuda
