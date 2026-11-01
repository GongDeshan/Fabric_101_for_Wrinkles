#pragma once
#include <type_define.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>

namespace uipc::backend::cuda
{
namespace sym::ipc_vertex_half_contact
{
#include "sym/vertex_half_plane_distance.inl"

    inline __device__ Float PH_barrier_energy(Float          kappa,
                                              Float          d_hat,
                                              Float          thickness,
                                              const Vector3& v,
                                              const Vector3& P,
                                              const Vector3& N)
    {
        using namespace codim_ipc_contact;

        Float D;
        HalfPlaneD(D, v, P, N);
        Float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);

        return E;
    }

    inline __device__ void PH_barrier_gradient_hessian(Vector3&       G,
                                                       Matrix3x3&     H,
                                                       Float          kappa,
                                                       Float          d_hat,
                                                       Float          thickness,
                                                       const Vector3& v,
                                                       const Vector3& P,
                                                       const Vector3& N)
    {
        using namespace codim_ipc_contact;

        Float D;
        HalfPlaneD(D, v, P, N);

        Float dBdD = 0.0;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Vector3 dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;

        Float ddBddD = 0.0;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix3x3 ddDddx;
        ddHalfPlaneDddx(ddDddx, v, P, N);

        H = ddBddD * dDdx * dDdx.transpose() + dBdD * ddDddx;
    }

    inline __device__ void PH_barrier_gradient(Vector3&       G,
                                               Float          kappa,
                                               Float          d_hat,
                                               Float          thickness,
                                               const Vector3& v,
                                               const Vector3& P,
                                               const Vector3& N)
    {
        using namespace codim_ipc_contact;

        Float D;
        HalfPlaneD(D, v, P, N);

        Float dBdD = 0.0;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Vector3 dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;
    }

    inline __device__ void compute_tan_basis(Vector3& e1, Vector3& e2, const Vector3& N)
    {
        using namespace codim_ipc_contact;

        Vector3 cross_x = -N.cross(Vector3(1, 0, 0));
        Vector3 cross_z = N.cross(Vector3(0, 0, 1));

        if(cross_x.squaredNorm() >= cross_z.squaredNorm())
        {
            e1 = cross_x.normalized();
            e2 = N.cross(cross_x).normalized();
        }
        else
        {
            e1 = cross_z.normalized();
            e2 = N.cross(cross_z).normalized();
        }
    }


    inline __device__ Float PH_friction_energy(Float          kappa,
                                               Float          d_hat,
                                               Float          thickness,
                                               Float          mu,
                                               Float          eps_vh,
                                               const Vector3& prev_x,
                                               const Vector3& x,
                                               const Vector3& P,
                                               const Vector3& N)
    {
        using namespace codim_ipc_contact;

        Float prev_D;
        HalfPlaneD(prev_D, prev_x, P, N);
        Float f = normal_force(kappa, d_hat, thickness, prev_D);

        Vector3 dx = x - prev_x;


        Vector3 e1, e2;
        compute_tan_basis(e1, e2, N);

        Vector2 tan_dV;

        TR(tan_dV, x, prev_x, e1, e2);

        return friction_energy(mu, f, eps_vh, tan_dV);
    }

    inline __device__ Float PH_friction_gradient_hessian(Vector3&   G,
                                                         Matrix3x3& H,
                                                         Float      kappa,
                                                         Float      d_hat,
                                                         Float      thickness,
                                                         Float      mu,
                                                         Float      eps_vh,
                                                         const Vector3& prev_v,
                                                         const Vector3& v,
                                                         const Vector3& P,
                                                         const Vector3& N)
    {
        using namespace codim_ipc_contact;

        Float prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        Float f = normal_force(kappa, d_hat, thickness, prev_D);

        Vector3 dV = v - prev_v;

        Vector3 e1, e2;
        compute_tan_basis(e1, e2, N);

        Vector2 tan_dV;

        TR(tan_dV, v, prev_v, e1, e2);

        Vector2 G2;
        friction_gradient(G2, mu, f, eps_vh, tan_dV);

        Matrix<Float, 2, 3> J;
        dTRdx(J, v, prev_v, e1, e2);

        G = J.transpose() * G2;

        Matrix2x2 H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_dV);

        H = J.transpose() * H2x2 * J;
    }

    inline __device__ void PH_friction_pGpkappa_pGpmu(Vector3&       pGpkappa,
                                                      Vector3&       pGpmu,
                                                      Float          kappa,
                                                      Float          mu,
                                                      const Vector3& prev_x,
                                                      const Vector3& x,
                                                      const Vector3& P,
                                                      const Vector3& N,
                                                      Float          d_hat,
                                                      Float          thickness,
                                                      Float          eps_vh)
    {
        namespace CIC = codim_ipc_contact;
        Vector2 dfdmu;
        Vector3 dx = x - prev_x;
        Vector3 e1, e2;
        compute_tan_basis(e1, e2, N);

        Float D;
        HalfPlaneD(D, prev_x, P, N);


        Matrix<Float, 2, 3> T;
        T.row(0)    = e1.transpose();
        T.row(1)    = e2.transpose();
        Vector2 tau = T * dx;

        Vector2 pGpkappa2, pGpmu2;
        CIC::d_friction_gradient_d_kappa_mu(
            pGpkappa2, pGpmu2, kappa, mu, tau, eps_vh, d_hat, thickness, D);

        pGpkappa = pGpkappa2.transpose() * T;
        pGpmu    = pGpmu2.transpose() * T;
    }

    inline __device__ void PH_barrier_pGpkappa(Vector3&       pGpkappa,
                                               Float          kappa,
                                               const Vector3& x,
                                               const Vector3& P,
                                               const Vector3& N,
                                               Float          d_hat,
                                               Float          thickness)
    {
        namespace CIC = codim_ipc_contact;

        Float D;
        HalfPlaneD(D, x, P, N);
        Vector3 dDdx;
        dHalfPlaneDdx(dDdx, x, P, N);

        //tex:
        //$$
        //G = \frac{\partial(\kappa B)} {\partial x} = \kappa \frac{\partial B} {\partial x}
        //$$
        //
        //$$
        //\frac{\partial G}{\partial \kappa} = \frac{\partial B}{\partial x} = (\frac{\partial B}{\partial D}) \frac{\partial D}{\partial x}
        //$$
        Float dBdD = CIC::d_KappaBarrierGradient_d_kappa(D, d_hat, thickness);
        pGpkappa   = dBdD * dDdx;
    }
}  // namespace sym::ipc_vertex_half_contact
}  // namespace uipc::backend::cuda
