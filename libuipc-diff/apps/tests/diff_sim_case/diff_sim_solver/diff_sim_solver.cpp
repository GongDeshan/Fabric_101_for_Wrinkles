#include "diff_sim_solver.h"
#include <uipc/io/scene_io.h>
#include <iostream>
#include <uipc/builtin/backend_name.h>

namespace uipc::test_case
{
DiffSimSolver::DiffSimSolver(core::World& world, std::string_view output_path)
    : m_world(world)
    , m_scene(backend::WorldVisitor{world}.scene().ref())
    , m_output_path(output_path)
{
    UIPC_ASSERT(m_world.frame() == 0, "World frame must be 0");
    m_world.dump();
    adjoint_method_feature = m_world.features().find<diff_sim::AdjointMethodFeature>();
    UIPC_ASSERT(adjoint_method_feature, "AdjointMethodFeature is not found");
    enable_grad_feature = m_world.features().find<diff_sim::EnableGradFeature>();
    UIPC_ASSERT(enable_grad_feature, "EnableGradFeature is not found");
    enable_grad_feature->no_grad();  // disable grad
    learning_rate_feature = m_world.features().find<diff_sim::LearningRateFeature>();
    UIPC_ASSERT(learning_rate_feature, "LearningRateFeature is not found");
    update_parameter_feature =
        m_world.features().find<diff_sim::UpdateParameterFeature>();
    UIPC_ASSERT(update_parameter_feature, "UpdateParameterFeature is not found");
}

void DiffSimSolver::forward(bool save_surf)
{
    if(enable_grad_feature->is_grad_enabled())  // disable grad
    {
        enable_grad_feature->no_grad();
    }

    m_world.recover(0);  // recover to frame 0
    _update_parms(_U);   // set parameters

    if(save_surf)
    {
        m_world.retrieve();
        core::SceneIO sio{m_scene};
        sio.write_surface(fmt::format(
            "{}/scene_surface{}.{}.obj", m_output_path, m_epoch, m_world.frame()));
    }

    SizeT sdi_count = 0;
    while(m_world.frame() < m_total_frame_count)
    {
        m_world.advance();

        auto&         sdi = _frame_local_SDI;
        SelectDofInfo info{m_world.frame(), sdi};
        select_Dofs(info);
        sdi_count += sdi.size();
        adjoint_method_feature->select_dofs(m_world.frame(), from_eigen(sdi));

        if(save_surf)
        {
            m_world.retrieve();
            core::SceneIO sio{m_scene};
            sio.write_surface(fmt::format(
                "{}/scene_surface{}.{}.obj", m_output_path, m_epoch, m_world.frame()));
        }

        if(per_frame_callback)
        {
            per_frame_callback();
        }
    }

    Xs.resize(sdi_count);
    Xs.setZero();
    adjoint_method_feature->receive_dofs(from_eigen(Xs));
}

void DiffSimSolver::backward()
{
    UIPC_ASSERT(compute_pLpX != nullptr, "Function dLdX is not set");
    UIPC_ASSERT(compute_Loss != nullptr, "Function Loss is not set");

    m_world.recover(0);
    auto& diff_sim = m_scene.diff_sim();
    auto  P_count  = diff_sim.parameters().size();
    _update_parms(_U);

    core::SceneIO sio{m_scene};
    sio.write_surface(fmt::format(
        "{}/scene_surface{}.{}.obj", m_output_path, m_epoch, m_world.frame()));

    if(enable_grad_feature->is_grad_enabled() == false)
    {
        enable_grad_feature->with_grad();
    }


    SizeT sdi_count = 0;

    while(m_world.frame() < m_total_frame_count)
    {
        m_world.advance();

        auto&         sdi = _frame_local_SDI;
        SelectDofInfo info{m_world.frame(), sdi};
        select_Dofs(info);
        sdi_count += sdi.size();
        adjoint_method_feature->select_dofs(m_world.frame(), from_eigen(sdi));
    }

    Xs.resize(sdi_count);
    Xs.setZero();
    adjoint_method_feature->receive_dofs(from_eigen(Xs));
    // std::cout << "X:\n" << Xs << std::endl;

    _dLdX.resize(sdi_count);
    _dLdX.setZero();

    {
        ComputedLdXInfo info{Xs, _dLdX};
        compute_pLpX(info);
        // std::cout << "dLdX:\n" << _dLdX << std::endl;
    }

    auto& dLdP = _dLdU;  // alias
    dLdP.resize(P_count);
    adjoint_method_feature->compute_dLdP(from_eigen(dLdP), from_eigen(_dLdX));

    // std::cout << "dLdP:\n" << dLdP << std::endl;

    if(compute_dPdU)
    {
        _dPdU.resize(P_count, _U.size());
        _dPdU.setZero();
        {
            ComputedPdUInfo info{_U, _dPdU};
            compute_dPdU(info);
        }
        _dLdU = dLdP.transpose() * _dPdU;
    }

    if(compute_pLpU)
    {
        _pLpU.resize(_dLdU.size());
        _pLpU.setZero();
        ComputepLpUInfo info{_pLpU, _U};
        compute_pLpU(info);
        _dLdU += _pLpU;
    }

    enable_grad_feature->no_grad();  // disable grad
}

void DiffSimSolver::check_functions()
{
    UIPC_ASSERT(compute_Loss != nullptr, "Function Loss is not set");
    UIPC_ASSERT(compute_pLpX != nullptr, "Function dLdX is not set");
    UIPC_ASSERT(select_Dofs != nullptr, "Function select_dofs is not set")

    if(compute_PofU || compute_dPdU)
    {
        UIPC_ASSERT(compute_dPdU && compute_PofU, "Function PofU and dPdU must be set together");
        UIPC_ASSERT(_U.size() > 0, "U (user parameters) is not set");
    }
}

void DiffSimSolver::_update_parms(const VectorX& U)
{
    auto& diff_sim = m_scene.diff_sim();
    if(compute_PofU)
    {
        VectorX P;
        P.resize(diff_sim.parameters().size());
        ComputePofUInfo info{U, P};
        compute_PofU(info);

        update_parameter_feature->update(from_eigen(P));
    }
    else
    {
        update_parameter_feature->update(from_eigen(U));
    }

    std::cout << "U:\n" << U.transpose() << std::endl;
}

Float DiffSimSolver::_compute_learning_rate(const VectorX& Farthest)
{
    m_world.recover(0);
    auto& diff_sim = m_scene.diff_sim();
    auto  _view    = view(diff_sim.parameters());
    Float alpha    = 1.0;
    if(compute_PofU)
    {
        VectorX P;
        P.resize(_view.size());
        ComputePofUInfo info{Farthest, P};
        compute_PofU(info);
        alpha = learning_rate_feature->learning_rate(from_eigen(P));
    }
    else
    {
        UIPC_ASSERT(Farthest.size() == _view.size(), "U size mismatch");
        alpha = learning_rate_feature->learning_rate(from_eigen(Farthest));
    }
    return alpha;
}

bool DiffSimSolver::gradient_descent(IndexT max_iter, std::function<void()> on_finish_once)
{
    check_functions();

    m_max_iter = max_iter;
    m_epoch    = 0;

    if(_U.size() == 0)
    {
        auto& diff_sim = m_scene.diff_sim();
        _U             = diff_sim.parameters().as_eigen();
    }

    while(m_epoch < m_max_iter)
    {
        fmt::println("{}", std::string(80, '-'));
        fmt::println("Start Epoch[{}]", m_epoch);

        backward();

        VectorX Dir = -_dLdU;

        if(allow_exit)
        {
            ExitConditionInfo info{*this};
            bool              ok = m_epoch == 0 ? false : allow_exit(info);

            if(ok)
            {
                fmt::println("Exit Condition Met");
                if(m_save_surf)
                {
                    forward(true);  // save surf
                }
                return true;
            }
        }

        ComputeLossInfo loss_info{Xs};
        Float           current_loss = compute_Loss(loss_info);
        if(m_epoch == 0)
            loss0 = current_loss;
        fmt::println("Epoch {}, Current Loss={}", m_epoch, current_loss);

        _U0 = _U;

        VectorX Farthest = _U0 + Dir;
        Float   alpha    = _compute_learning_rate(Farthest);
        fmt::println("Penetration Free Learning Rate: {}", alpha);


        // backtracking line search
        IndexT line_search_iter = 0;
        Float  new_loss         = current_loss;

        while(true)
        {
            _U = _U0 + alpha * Dir;

            forward(true);  // do not save surf

            ComputeLossInfo loss_info{Xs};
            new_loss = compute_Loss(loss_info);
            fmt::println(">>> LS[{}] in Epoch[{}]: Alpha={}, Loss={}/{}",
                         line_search_iter,
                         m_epoch,
                         alpha,
                         new_loss,
                         current_loss);

            line_search_iter++;

            if(line_search_iter >= m_max_line_search)
            {
                fmt::println("Line Search Iteration Exceeds Max Iteration");
                break;
            }

            if(new_loss <= current_loss)
            {
                break;
            }
            else
            {
                alpha *= 0.5;
            }
        }

        if(m_save_surf)
        {
            forward(true);  // save surf
        }

        fmt::println("Finish Epoch[{}]: Loss={}", m_epoch, new_loss);
        fmt::println("{}", std::string(80, ' '));

        if(on_finish_once)
        {
            on_finish_once();
        }

        m_epoch++;
    }

    return false;
}

}  // namespace uipc::test_case