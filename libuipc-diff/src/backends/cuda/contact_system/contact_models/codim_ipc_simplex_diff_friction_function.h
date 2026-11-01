#pragma once
#include <type_define.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::cuda
{
namespace sym::codim_ipc_diff_friction
{
    namespace CIC = sym::codim_ipc_contact;

    namespace PP
    {
#include "sym/ipc_diff_friction_PP.inl"
    }

    namespace PE
    {
#include "sym/ipc_diff_friction_PE.inl"
    }

    namespace PT
    {
#include "sym/ipc_diff_friction_PT.inl"
    }

    namespace EE
    {
#include "sym/ipc_diff_friction_EE.inl"
    }

    //tex:
    //$$
    //\tau = T dx
    //$$
    //
    //$$
    //dx = x - x^n
    //$$
    //
    //$$
    //\frac{\partial^2 E}{\partial x_j \partial x^n_k} =
    //
    //\mu \frac{\partial N^n}{\partial x^n_k}\frac{\partial f}{\partial \tau_i} T_{ij}
    //
    //+ \mu N^n \frac{\partial^2 f}{\partial \tau_i \partial \tau_l} (-T_{lk}) T_{ij}
    //
    //+ \mu N^n \frac{\partial f}{\partial \tau_i} \frac{\partial T_{ij}}{\partial x^n}
    //$$

    //tex:
    //$$
    // N = -\frac{\partial B}{\partial d} = -\frac{\partial B}{\partial D} \frac{\partial D}{\partial d}
    // = -2 \frac{\partial B}{\partial D} d
    //$$
    //
    //$$
    // \frac{\partial N}{\partial D}
    // = -2 (\frac{\partial^2 B}{\partial D^2} d + \frac{\partial B}{\partial D} \frac{\partial d}{\partial D})
    // $$
    //
    //The derivative of d with respect to D is:
    //$$
    //\frac{\partial d}{\partial D} = \frac{1}{2d}
    //$$

    inline MUDA_GENERIC Float dNdD(Float kappa, Float d_hat, Float thickness, Float D)
    {
        Float d    = std::sqrt(D);
        Float dddD = 1.0 / (2.0 * d);
        Float dBdD;
        CIC::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        Float ddBddD;
        CIC::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        return -2.0 * (ddBddD * d + dBdD * dddD);
    }

    //tex:
    //$$
    //\frac{\partial^2 E}{\partial x_j \partial x^n_k} =
    //
    //\mu \left(T^T \frac{\partial f}{\partial \tau} \right) \left(\frac{\partial N^n}{\partial D}\frac{\partial D}{\partial x^n} \right)^T
    //
    //+ \mu N^n\left(T^T\frac{\partial^2 f}{\partial \tau^2} \left( A - T\right)  + B\right)
    //$$

    //tex: $$ A_{lk} = \frac{\partial T_{lp}} {\partial x ^ n_k} dx_p $$

    //tex:
    //$$
    //B_{jk} =\frac{\partial f}
    //{
    //    \partial \tau_i
    //}
    //\frac{\partial T_{ij}} {\partial x ^ n_k}
    //$$

    template <int N>
    MUDA_GENERIC Eigen::Matrix<Float, 3 * N, 3 * N> compute_pGpXn(
        Float                                     mu,
        const Eigen::Matrix<Float, 2, 3 * N>&     T,
        const Vector2&                            dfdtau,
        const Matrix2x2&                          ddfddtau,
        Float                                     Nn,
        Float                                     dNdD,
        const Eigen::Vector<Float, 3 * N>&        dDdXn,
        const Eigen::Matrix<Float, 2, 3 * N>&     A,
        const Eigen::Matrix<Float, 3 * N, 3 * N>& B)
    {
        return mu * (T.transpose() * dfdtau) * (dNdD * dDdXn).transpose()
               + mu * Nn * (T.transpose() * ddfddtau * (A - T) + B);
    }

    inline MUDA_GENERIC Vector3 PP_e_hat(const Vector6& X, const Vector6& Xn)
    {
        Vector3 N = Xn.segment<3>(3) - Xn.segment<3>(0);

        Float s1 = N.cross(Vector3::UnitX()).squaredNorm();
        Float s2 = N.cross(Vector3::UnitY()).squaredNorm();

        return (s1 >= s2) ? Vector3::UnitX() : Vector3::UnitY();
    }

    inline MUDA_GENERIC Matrix6x6 PP_pGpXn(Float          kappa,
                                           Float          mu,
                                           Float          eps_vh,
                                           Float          d_hat,
                                           Float          thickness,
                                           const Vector6& X,
                                           const Vector6& Xn)
    {
        Eigen::Matrix<Float, 2, 6> T;
        Vector3                    e_hat = PP_e_hat(X, Xn);
        PP::TR(T, Xn, e_hat);

        Vector6 dX        = X - Xn;
        Vector2 tan_rel_x = T * dX;

        Vector2   dfdtau;
        Matrix2x2 ddfddtau;
        CIC::dfdtau(dfdtau, eps_vh, tan_rel_x);
        CIC::ddfddtau(ddfddtau, eps_vh, tan_rel_x);

        Float D;
        distance::point_point_distance2<Float>(Xn.segment<3>(0), Xn.segment<3>(3), D);

        Float Nn    = CIC::normal_force(kappa, d_hat, thickness, D);
        Float dNndD = dNdD(kappa, d_hat, thickness, D);

        Eigen::Vector<Float, 6> dDdXn;
        distance::point_point_distance2_gradient<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), dDdXn);

        Eigen::Matrix<Float, 2, 6> An;
        PP::An(An, Xn, X, e_hat);

        Eigen::Matrix<Float, 6, 6> Bn;
        PP::Bn(Bn, Xn, X, dfdtau, e_hat);

        return compute_pGpXn<2>(mu, T, dfdtau, ddfddtau, Nn, dNndD, dDdXn, An, Bn);
    }

    inline MUDA_GENERIC void PP_friction_pGpkappa_pGpmu(Vector6&       pGpkappa,
                                                        Vector6&       pGpmu,
                                                        Float          kappa,
                                                        Float          mu,
                                                        const Vector6& X,
                                                        const Vector6& Xn,
                                                        Float          d_hat,
                                                        Float thickness,
                                                        Float eps_vh)
    {
        namespace CIC = codim_ipc_contact;
        Eigen::Matrix<Float, 2, 6> T;
        Vector3                    e_hat = PP_e_hat(X, Xn);
        PP::TR(T, Xn, e_hat);

        Vector6 dX  = X - Xn;
        Vector2 tau = T * dX;

        Float D;
        distance::point_point_distance2<Float>(Xn.segment<3>(0), Xn.segment<3>(3), D);

        Vector2 pGpkappa2, pGpmu2;
        CIC::d_friction_gradient_d_kappa_mu(
            pGpkappa2, pGpmu2, kappa, mu, tau, eps_vh, d_hat, thickness, D);

        pGpkappa = pGpkappa2.transpose() * T;
        pGpmu    = pGpmu2.transpose() * T;
    }

    inline MUDA_GENERIC Matrix9x9 PE_pGpXn(Float          kappa,
                                           Float          mu,
                                           Float          eps_vh,
                                           Float          d_hat,
                                           Float          thickness,
                                           const Vector9& X,
                                           const Vector9& Xn)
    {
        Eigen::Matrix<Float, 2, 9> T;
        PE::TR(T, Xn);

        Vector9 dX        = X - Xn;
        Vector2 tan_rel_x = T * dX;

        Vector2   dfdtau;
        Matrix2x2 ddfddtau;
        CIC::dfdtau(dfdtau, eps_vh, tan_rel_x);
        CIC::ddfddtau(ddfddtau, eps_vh, tan_rel_x);

        Float D;
        distance::point_edge_distance2<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), D);

        Float Nn    = CIC::normal_force(kappa, d_hat, thickness, D);
        Float dNndD = dNdD(kappa, d_hat, thickness, D);

        Eigen::Vector<Float, 9> dDdXn;
        distance::point_edge_distance2_gradient<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), dDdXn);

        Eigen::Matrix<Float, 2, 9> An;
        PE::An(An, Xn, X);

        Eigen::Matrix<Float, 9, 9> Bn;
        PE::Bn(Bn, Xn, X, dfdtau);

        return compute_pGpXn<3>(mu, T, dfdtau, ddfddtau, Nn, dNndD, dDdXn, An, Bn);
    }

    inline MUDA_GENERIC void PE_friction_pGpkappa_pGpmu(Vector9&       pGpkappa,
                                                        Vector9&       pGpmu,
                                                        Float          kappa,
                                                        Float          mu,
                                                        const Vector9& X,
                                                        const Vector9& Xn,
                                                        Float          d_hat,
                                                        Float thickness,
                                                        Float eps_vh)
    {
        namespace CIC = codim_ipc_contact;
        Eigen::Matrix<Float, 2, 9> T;
        PE::TR(T, Xn);

        Vector9 dX  = X - Xn;
        Vector2 tau = T * dX;

        Float D;
        distance::point_edge_distance2<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), D);

        Vector2 pGpkappa2, pGpmu2;
        CIC::d_friction_gradient_d_kappa_mu(
            pGpkappa2, pGpmu2, kappa, mu, tau, eps_vh, d_hat, thickness, D);

        pGpkappa = pGpkappa2.transpose() * T;
        pGpmu    = pGpmu2.transpose() * T;
    }

    inline MUDA_GENERIC Matrix12x12 PT_pGpXn(Float           kappa,
                                             Float           mu,
                                             Float           eps_vh,
                                             Float           d_hat,
                                             Float           thickness,
                                             const Vector12& X,
                                             const Vector12& Xn)
    {
        Eigen::Matrix<Float, 2, 12> T;
        PT::TR(T, Xn);

        Vector12 dX        = X - Xn;
        Vector2  tan_rel_x = T * dX;

        Vector2   dfdtau;
        Matrix2x2 ddfddtau;
        CIC::dfdtau(dfdtau, eps_vh, tan_rel_x);
        CIC::ddfddtau(ddfddtau, eps_vh, tan_rel_x);

        Float D;
        distance::point_triangle_distance2<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), Xn.segment<3>(9), D);

        Float Nn    = CIC::normal_force(kappa, d_hat, thickness, D);
        Float dNndD = dNdD(kappa, d_hat, thickness, D);

        Eigen::Vector<Float, 12> dDdXn;
        distance::point_triangle_distance2_gradient<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), Xn.segment<3>(9), dDdXn);

        Eigen::Matrix<Float, 2, 12> An;
        PT::An(An, Xn, X);

        Eigen::Matrix<Float, 12, 12> Bn;
        PT::Bn(Bn, Xn, X, dfdtau);

        return compute_pGpXn<4>(mu, T, dfdtau, ddfddtau, Nn, dNndD, dDdXn, An, Bn);
    }

    inline MUDA_GENERIC void PT_friction_pGpkappa_pGpmu(Vector12& pGpkappa,
                                                        Vector12& pGpmu,
                                                        Float     kappa,
                                                        Float     mu,
                                                        const Vector12& X,
                                                        const Vector12& Xn,
                                                        Float           d_hat,
                                                        Float thickness,
                                                        Float eps_vh)
    {
        namespace CIC = codim_ipc_contact;
        Eigen::Matrix<Float, 2, 12> T;
        PT::TR(T, Xn);

        Vector12 dX  = X - Xn;
        Vector2  tau = T * dX;

        Float D;
        distance::point_triangle_distance2<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), Xn.segment<3>(9), D);

        Vector2 pGpkappa2, pGpmu2;
        CIC::d_friction_gradient_d_kappa_mu(
            pGpkappa2, pGpmu2, kappa, mu, tau, eps_vh, d_hat, thickness, D);

        pGpkappa = pGpkappa2.transpose() * T;
        pGpmu    = pGpmu2.transpose() * T;
    }

    inline MUDA_GENERIC Matrix12x12 EE_pGpXn(Float           kappa,
                                             Float           mu,
                                             Float           eps_vh,
                                             Float           d_hat,
                                             Float           thickness,
                                             const Vector12& X,
                                             const Vector12& Xn)
    {
        Eigen::Matrix<Float, 2, 12> T;
        EE::TR(T, Xn);

        Vector12 dX        = X - Xn;
        Vector2  tan_rel_x = T * dX;

        Vector2   dfdtau;
        Matrix2x2 ddfddtau;
        CIC::dfdtau(dfdtau, eps_vh, tan_rel_x);
        CIC::ddfddtau(ddfddtau, eps_vh, tan_rel_x);

        Float D;
        distance::edge_edge_distance2<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), Xn.segment<3>(9), D);

        Float Nn    = CIC::normal_force(kappa, d_hat, thickness, D);
        Float dNndD = dNdD(kappa, d_hat, thickness, D);

        Eigen::Vector<Float, 12> dDdXn;
        distance::edge_edge_distance2_gradient<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), Xn.segment<3>(9), dDdXn);

        Eigen::Matrix<Float, 2, 12> An;
        EE::An(An, Xn, X);

        Eigen::Matrix<Float, 12, 12> Bn;
        EE::Bn(Bn, Xn, X, dfdtau);

        return compute_pGpXn<4>(mu, T, dfdtau, ddfddtau, Nn, dNndD, dDdXn, An, Bn);
    }

    inline MUDA_GENERIC void EE_friction_pGpkappa_pGpmu(Vector12& pGpkappa,
                                                        Vector12& pGpmu,
                                                        Float     kappa,
                                                        Float     mu,
                                                        const Vector12& X,
                                                        const Vector12& Xn,
                                                        Float           d_hat,
                                                        Float thickness,
                                                        Float eps_vh)
    {
        namespace CIC = codim_ipc_contact;
        Eigen::Matrix<Float, 2, 12> T;
        EE::TR(T, Xn);

        Vector12 dX  = X - Xn;
        Vector2  tau = T * dX;

        Float D;
        distance::edge_edge_distance2<Float>(
            Xn.segment<3>(0), Xn.segment<3>(3), Xn.segment<3>(6), Xn.segment<3>(9), D);

        Vector2 pGpkappa2, pGpmu2;
        CIC::d_friction_gradient_d_kappa_mu(
            pGpkappa2, pGpmu2, kappa, mu, tau, eps_vh, d_hat, thickness, D);

        pGpkappa = pGpkappa2.transpose() * T;
        pGpmu    = pGpmu2.transpose() * T;
    }
}  // namespace sym::codim_ipc_diff_friction
}  // namespace uipc::backend::cuda
