#pragma once
#include <type_define.h>
#include <utils/dihedral_angle.h>
//ref: https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf
namespace uipc::backend::cuda
{
namespace sym::discrete_shell_friction_hardening_plastic_bending
{
#include "sym/discrete_shell_bending.inl"

    inline UIPC_GENERIC Float sign(Float& value)
    {
        if(value > 0.0)
            return 1.0;
        else if(value < 0.0)
            return -1.0;
        else
            return 0.0;
    }

    inline UIPC_GENERIC Float relu(Float& value)
    {
        if(value > 0.0)
            return value;
        else
            return 0.0;
    }

    inline UIPC_GENERIC Float gpu_abs(Float& value)
    {
        if(value > 0.0)
            return value;
        else
            return -value;
    }

    inline UIPC_GENERIC void compute_constants(Float&         L0,
                                               Float&         h_bar,
                                               Float&         theta_bar,
                                               Float&         V_bar,
                                               const Vector3& x0_bar,
                                               const Vector3& x1_bar,
                                               const Vector3& x2_bar,
                                               const Vector3& x3_bar,
                                               Float          thickness0,
                                               Float          thickness1,
                                               Float          thickness2,
                                               Float          thickness3)

    {
        L0         = (x2_bar - x1_bar).norm();
        Vector3 n1 = (x1_bar - x0_bar).cross(x2_bar - x0_bar);
        Vector3 n2 = (x2_bar - x3_bar).cross(x1_bar - x3_bar);
        Float   A  = (n1.norm() + n2.norm()) / 2.0;
        h_bar      = A / 3.0 / L0;
        dihedral_angle(x0_bar, x1_bar, x2_bar, x3_bar, theta_bar);

        Float thickness = (thickness0 + thickness1 + thickness2 + thickness3) / 4.0;
        V_bar = A * thickness;
    }

    inline UIPC_GENERIC Float E(const Vector3& x0,
                                const Vector3& x1,
                                const Vector3& x2,
                                const Vector3& x3,
                                Float          L0,
                                Float          h_bar,
                                Float          theta_bar,
                                Float          theta_yield,
                                Float          theta_plastic,
                                Float          anchor_theta,
                                Float          thres_theta,
                                Float          kappa,
                                Float          friction_stiffness)
    {

        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float elastic_theta = theta - theta_plastic;
        Float theta_var     = theta - anchor_theta;

        Float R;
        DSGFPB::E(R, kappa, elastic_theta, theta_bar, L0, h_bar);

        Float R_friction;
        DSGFPB::E(R_friction, friction_stiffness, theta_var, theta_bar, L0, h_bar);

        return (R + R_friction);
        //return R;
    }

    inline UIPC_GENERIC void dEdx(Vector12&      G,
                                  const Vector3& x0,
                                  const Vector3& x1,
                                  const Vector3& x2,
                                  const Vector3& x3,
                                  Float          L0,
                                  Float          h_bar,
                                  Float          theta_bar,
                                  Float          theta_yield,
                                  Float          theta_plastic,
                                  Float          anchor_theta,
                                  Float          thres_theta,
                                  Float          kappa,
                                  Float          friction_stiffness)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float elastic_theta = theta - theta_plastic;
        Float dEdtheta;

        DSGFPB::dEdtheta(dEdtheta, kappa, elastic_theta, theta_bar, L0, h_bar);

        Float theta_var = theta - anchor_theta;
        Float dEdtheta_friction;

        DSGFPB::dEdtheta(dEdtheta_friction, friction_stiffness, theta_var, theta_bar, L0, h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        G = (dEdtheta + dEdtheta_friction) * dthetadx;
    }

    inline UIPC_GENERIC void ddEddx(Matrix12x12&   H,
                                    const Vector3& x0,
                                    const Vector3& x1,
                                    const Vector3& x2,
                                    const Vector3& x3,
                                    Float          L0,
                                    Float          h_bar,
                                    Float          theta_bar,
                                    Float          theta_yield,
                                    Float          theta_plastic,
                                    Float          anchor_theta,
                                    Float          thres_theta,
                                    Float          kappa,
                                    Float          friction_stiffness)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float elastic_theta = theta - theta_plastic;

        Float dEdtheta;
        DSGFPB::dEdtheta(dEdtheta, kappa, elastic_theta, theta_bar, L0, h_bar);

        Float ddEddtheta;
        DSGFPB::ddEddtheta(ddEddtheta, kappa, elastic_theta, theta_bar, L0, h_bar);

        Float theta_var = theta - anchor_theta;

        Float dEdtheta_friction;
        DSGFPB::dEdtheta(dEdtheta_friction, friction_stiffness, theta_var, theta_bar, L0, h_bar);

        Float ddEddtheta_friction;
        DSGFPB::ddEddtheta(ddEddtheta_friction, friction_stiffness, theta_var, theta_bar, L0, h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Matrix12x12 ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        H = dthetadx * (ddEddtheta + ddEddtheta_friction) * dthetadx.transpose()
            + (dEdtheta + dEdtheta_friction) * ddthetaddx;
    }

    inline UIPC_GENERIC void pGpKappa(Vector12&      pGpP,
                                      const Vector3& x0,
                                      const Vector3& x1,
                                      const Vector3& x2,
                                      const Vector3& x3,
                                      Float          L0,
                                      Float          h_bar,
                                      Float          kappa,
                                      Float          theta_bar,
                                      Float          theta_plastic,
                                      Float          d_plastic_theta_d_k)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float elastic_theta = theta - theta_plastic;

        Float dEdtheta =
            2.0 * L0 * (elastic_theta - theta_bar - kappa * d_plastic_theta_d_k) / h_bar;

        //Float dEdtheta;
        // set kappa 1.0 to get the derivative of the energy with respect to kappa
        //DSGFPB::dEdtheta(dEdtheta, 1.0, elastic_theta, theta_bar, L0, h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta * dthetadx;
    }


    inline UIPC_GENERIC void pGpFrictionStiffness(Vector12&      pGpP,
                                                  const Vector3& x0,
                                                  const Vector3& x1,
                                                  const Vector3& x2,
                                                  const Vector3& x3,
                                                  Float          L0,
                                                  Float          h_bar,
                                                  Float          theta_bar,
                                                  Float          anchor_theta)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Float theta_var = theta - anchor_theta;

        Float dEdtheta_friction;
        DSGFPB::dEdtheta(dEdtheta_friction, 1.0, theta_var, theta_bar, L0, h_bar);

        pGpP = dEdtheta_friction * dthetadx;
    }

    
    inline UIPC_GENERIC void pGpThetaThres(Vector12&      pGpP,
                                           const Vector3& x0,
                                           const Vector3& x1,
                                           const Vector3& x2,
                                           const Vector3& x3,
                                           Float          L0,
                                           Float          h_bar,
                                           Float d_anchor_theta_d_theta_theta,
                                           Float friction_stiffness)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Float dEdtheta_thres =
            -2.0 * L0 * friction_stiffness * d_anchor_theta_d_theta_theta / h_bar;

        pGpP = dEdtheta_thres * dthetadx;
    }

    inline UIPC_GENERIC void pGpThetaY(Vector12&      pGpP,
                                       const Vector3& x0,
                                       const Vector3& x1,
                                       const Vector3& x2,
                                       const Vector3& x3,
                                       Float          L0,
                                       Float          h_bar,
                                       Float          theta_yield,
                                       Float          theta_plastic,
                                       Float          theta_bar,
                                       Float          kappa)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float elastic_theta = theta - theta_plastic;
        Float dis_to_yield  = gpu_abs(elastic_theta) - theta_yield;

        Float dEdtheta_y = relu(dis_to_yield) > 0.0 ?
                               sign(elastic_theta) * 2 * L0 * kappa / h_bar :
                               0.0;  // D energy/ D theta_yield

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_y * dthetadx;
    }


    inline UIPC_GENERIC void pGpYield(Vector12&      pGpP,
                                      const Vector3& x0,
                                      const Vector3& x1,
                                      const Vector3& x2,
                                      const Vector3& x3,
                                      Float          L0,
                                      Float          h_bar,
                                      Float          bending_stiffness,
                                      Float d_theta_plastic_d_yield_ori)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

        Float dEdtheta = -2.0 * L0 * bending_stiffness * d_theta_plastic_d_yield_ori / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta * dthetadx;
    }

    inline UIPC_GENERIC void pGpHardeningK(Vector12&      pGpP,
                                           const Vector3& x0,
                                           const Vector3& x1,
                                           const Vector3& x2,
                                           const Vector3& x3,
                                           Float          L0,
                                           Float          h_bar,
                                           Float          bending_stiffness,
                                           Float d_theta_plastic_d_hardening_k)
    {
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float dEdtheta = -2.0 * L0 * bending_stiffness * d_theta_plastic_d_hardening_k / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta * dthetadx;
    }

}  // namespace sym::discrete_shell_friction_hardening_plastic_bending
}  // namespace uipc::backend::cuda
