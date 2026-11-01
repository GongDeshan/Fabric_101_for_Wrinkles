#pragma once
#include <uipc/uipc.h>
#include <Eigen/Sparse>
#include <uipc/diff_sim/enable_grad_feature.h>
#include <uipc/diff_sim/adjoint_method_feature.h>
#include <uipc/diff_sim/learning_rate_feature.h>
#include <uipc/diff_sim/update_parameter_feature.h>
#include <uipc/builtin/backend_name.h>

namespace uipc
{
template <typename T>
backend::BufferView from_eigen(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec)
{
    return backend::BufferView{
        (backend::HandleT)vec.data(), 0, (SizeT)vec.size(), sizeof(T), sizeof(T), builtin::none};
}
}  // namespace uipc

namespace uipc::test_case
{
class DiffSimSolver
{
  public:
    DiffSimSolver(core::World& world, std::string_view output_path);

    bool gradient_descent(IndexT max_iter, std::function<void()> on_finish_once = nullptr);

    void save_surf(bool save) { m_save_surf = save; }

    void max_line_search(IndexT max_iter) { m_max_line_search = max_iter; }

    void total_frame_count(IndexT total_frame_count)
    {
        m_total_frame_count = total_frame_count;
    }

    class SelectDofInfo
    {
      public:
        SelectDofInfo(SizeT frame, VectorXi& local_SDI)
            : m_frame(frame)
            , m_local_SDI(local_SDI)
        {
            m_local_SDI.resize(0);  // clear
        }

        auto frame() const { return m_frame; }
        /**
         * @brief Select dof indices for current frame
         */
        auto& sdi() { return m_local_SDI; }

      private:
        friend class DiffSimSolver;
        SizeT     m_frame;
        VectorXi& m_local_SDI;
    };

    void SelectDofs(std::function<void(SelectDofInfo&)> select_dofs)
    {
        this->select_Dofs = select_dofs;
    }

    class ComputeLossInfo
    {
      public:
        ComputeLossInfo(const VectorX& X)
            : m_X(X)
        {
        }

        auto& X() { return m_X; }

      private:
        friend class DiffSimSolver;
        const VectorX& m_X;
    };

    void L(std::function<Float(ComputeLossInfo&)> compute_loss)
    {
        compute_Loss = compute_loss;
    }

    class ComputedLdXInfo
    {
      public:
        ComputedLdXInfo(const VectorX& X, VectorX& dLdX)
            : m_X(X)
            , m_dLdX(dLdX)
        {
        }

        auto& X() { return m_X; }
        auto& dLdX() { return m_dLdX; }

      private:
        friend class DiffSimSolver;
        const VectorX& m_X;
        VectorX&       m_dLdX;
    };

    void dLdX(std::function<void(ComputedLdXInfo&)> compute_dLdX)
    {
        this->compute_pLpX = compute_dLdX;
    }

    class ComputePofUInfo
    {
      public:
        ComputePofUInfo(const VectorX& U, VectorX& P)
            : m_U(U)
            , m_P(P)
        {
        }

        auto& P() { return m_P; }
        auto& U() { return m_U; }

      private:
        friend class DiffSimSolver;
        const VectorX& m_U;
        VectorX&       m_P;
    };

    void PofU(std::function<void(ComputePofUInfo&)> compute_PofU)
    {
        this->compute_PofU = compute_PofU;
    }

    class ComputedPdUInfo
    {
      public:
        ComputedPdUInfo(const VectorX& U, Eigen::SparseMatrix<Float>& dPdU)
            : m_U(U)
            , m_dPdU(dPdU)
        {
        }

        auto& dPdU() { return m_dPdU; }
        auto& U() { return m_U; }

      private:
        friend class DiffSimSolver;
        const VectorX&              m_U;
        Eigen::SparseMatrix<Float>& m_dPdU;
    };

    void dPdU(std::function<void(ComputedPdUInfo&)> compute_dPdU)
    {
        this->compute_dPdU = compute_dPdU;
    }

    class ComputepLpUInfo
    {
      public:
        ComputepLpUInfo(const VectorX& U, VectorX& pLpU)
            : m_U(U)
            , m_pLpU(pLpU)
        {
        }

        auto& pLpU() { return m_pLpU; }

      private:
        friend class DiffSimSolver;
        const VectorX& m_U;
        VectorX&       m_pLpU;
    };

    void pLpU(std::function<void(ComputepLpUInfo&)> compute_pLpU)
    {
        this->compute_pLpU = compute_pLpU;
    }

    class ExitConditionInfo
    {
      public:
        ExitConditionInfo(DiffSimSolver& solver)
            : m_solver(solver)
        {
        }

        auto& U() { return m_solver._U; }
        auto& X() { return m_solver.Xs; }
        auto& dLdU() { return m_solver._dLdU; }

        Float loss()
        {
            ComputeLossInfo info{m_solver.Xs};
            return m_solver.compute_Loss(info);
        }

        Float rel_loss()
        {
            return m_solver.loss0 > 0 ? loss() / m_solver.loss0 : 1;
        }

      private:
        friend class DiffSimSolver;
        DiffSimSolver& m_solver;
    };

    void ExitCondition(std::function<bool(ExitConditionInfo&)> allow_exit)
    {
        this->allow_exit = allow_exit;
    }

    void PerFrameCallback(std::function<void()> per_frame_callback)
    {
        this->per_frame_callback = per_frame_callback;
    }

    VectorX& U() { return _U; }

  private:
    void  forward(bool save_surf);
    void  backward();
    void  check_functions();
    void  _update_parms(const VectorX& U);
    Float _compute_learning_rate(const VectorX& Farthest);

    bool         m_save_surf       = false;
    IndexT       m_max_iter        = -1;
    IndexT       m_max_line_search = 10;
    IndexT       m_epoch           = 0;
    std::string  m_output_path;
    core::World& m_world;
    core::Scene& m_scene;
    IndexT       m_total_frame_count = 0;

    // 0: SelectDofs
    std::function<void(SelectDofInfo&)> select_Dofs = nullptr;

    // 1: L(U,X)
    std::function<Float(ComputeLossInfo&)> compute_Loss = nullptr;

    // 2: pLpX
    std::function<void(ComputedLdXInfo&)> compute_pLpX = nullptr;

    // 3: P
    std::function<void(ComputePofUInfo&)> compute_PofU = nullptr;

    // 4: dPdU
    std::function<void(ComputedPdUInfo&)> compute_dPdU = nullptr;

    // 5: pLpU
    std::function<void(ComputepLpUInfo&)> compute_pLpU = nullptr;


    std::function<bool(ExitConditionInfo&)> allow_exit         = nullptr;
    std::function<void()>                   per_frame_callback = nullptr;

    VectorXi _SDI;
    VectorXi _frame_local_SDI;
    VectorX  _dLdX;

    VectorX                    _dLdU;
    VectorX                    _pLpU;
    MatrixX                    _dXdU;
    VectorX                    _U;
    VectorX                    _U0;
    Eigen::SparseMatrix<Float> _dPdU;
    Float                      loss0 = 0;


    S<diff_sim::AdjointMethodFeature>   adjoint_method_feature   = nullptr;
    S<diff_sim::EnableGradFeature>      enable_grad_feature      = nullptr;
    S<diff_sim::LearningRateFeature>    learning_rate_feature    = nullptr;
    S<diff_sim::UpdateParameterFeature> update_parameter_feature = nullptr;

    VectorX Xs;
};
}  // namespace uipc::test_case
