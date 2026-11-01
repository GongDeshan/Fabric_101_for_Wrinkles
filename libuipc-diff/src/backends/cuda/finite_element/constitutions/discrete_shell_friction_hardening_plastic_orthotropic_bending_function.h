#pragma once
#include <type_define.h>
#include <utils/dihedral_angle.h>
//ref: https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf
namespace uipc::backend::cuda
{
namespace sym::discrete_shell_friction_hardening_plastic_orthotropic_bending
{
#include "sym/orthotropic_discrete_shell_bending.inl"

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
                                Float          torsion_bar,
                                Float          bias_angle,
                                Float          plastic_theta_warp,
                                Float          plastic_theta_weft,
                                Float          anchor_theta_warp,
                                Float          anchor_theta_weft,
                                Float          kappa_warp,
                                Float          kappa_weft,
                                Float          kappa_torsion,
                                Float          kappa_poisson,
                                Float          friction_stiffness_warp,
                                Float          friction_stiffness_weft)
    {

        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);
        Float sin_angle          = sin(angle_between_warp);

        Float theta_warp    = theta * cos_angle * cos_angle;
        Float theta_weft    = theta * sin_angle * sin_angle;
        Float theta_torsion = theta * cos_angle * sin_angle;

        Float elastic_theta_warp    = theta_warp - plastic_theta_warp;
        Float elastic_theta_weft    = theta_weft - plastic_theta_weft;
        Float elastic_theta_torsion = theta_torsion - torsion_bar;

        Float R;
        DSGFPOB::E(R,
                   kappa_warp,
                   kappa_weft,
                   kappa_torsion,
                   kappa_poisson,
                   elastic_theta_warp,
                   elastic_theta_weft,
                   elastic_theta_torsion,
                   L0,
                   h_bar);

        Float theta_var_warp = theta_warp - anchor_theta_warp;
        Float theta_var_weft = theta_weft - anchor_theta_weft;

        Float R_friction;
        DSGFPOB::E_friction(R_friction,
                            friction_stiffness_warp,
                            friction_stiffness_weft,
                            theta_var_warp,
                            theta_var_weft,
                            L0,
                            h_bar);

        return (R + R_friction);
    }

    inline UIPC_GENERIC void dEdx(Vector12&      G,
                                  const Vector3& x0,
                                  const Vector3& x1,
                                  const Vector3& x2,
                                  const Vector3& x3,
                                  Float          L0,
                                  Float          h_bar,
                                  Float          torsion_bar,
                                  Float          bias_angle,
                                  Float          plastic_theta_warp,
                                  Float          plastic_theta_weft,
                                  Float          anchor_theta_warp,
                                  Float          anchor_theta_weft,
                                  Float          kappa_warp,
                                  Float          kappa_weft,
                                  Float          kappa_torsion,
                                  Float          kappa_poisson,
                                  Float          friction_stiffness_warp,
                                  Float          friction_stiffness_weft)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);
        Float sin_angle          = sin(angle_between_warp);

        Float theta_warp    = theta * cos_angle * cos_angle;
        Float theta_weft    = theta * sin_angle * sin_angle;
        Float theta_torsion = theta * cos_angle * sin_angle;

        Float elastic_theta_warp    = theta_warp - plastic_theta_warp;
        Float elastic_theta_weft    = theta_weft - plastic_theta_weft;
        Float elastic_theta_torsion = theta_torsion - torsion_bar;

        Float dEdtheta;
        DSGFPOB::dEdtheta(dEdtheta,
                          kappa_warp,
                          kappa_weft,
                          kappa_torsion,
                          kappa_poisson,
                          elastic_theta_warp,
                          elastic_theta_weft,
                          elastic_theta_torsion,
                          cos_angle,
                          sin_angle,
                          L0,
                          h_bar);

        Float theta_var_warp = theta_warp - anchor_theta_warp;
        Float theta_var_weft = theta_weft - anchor_theta_weft;

        Float dEdtheta_friction;
        DSGFPOB::dE_friction_dtheta(dEdtheta_friction,
                                    friction_stiffness_warp,
                                    friction_stiffness_weft,
                                    cos_angle,
                                    sin_angle,
                                    theta_var_warp,
                                    theta_var_weft,
                                    L0,
                                    h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        //printf("grad: \r\n %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f \r\n",
        //       dthetadx(0),
        //       dthetadx(1),
        //       dthetadx(2),
        //       dthetadx(3),
        //       dthetadx(4),
        //       dthetadx(5),
        //       dthetadx(6),
        //       dthetadx(7),
        //       dthetadx(8),
        //       dthetadx(9),
        //       dthetadx(10),
        //       dthetadx(11));

        G = (dEdtheta + dEdtheta_friction) * dthetadx;
    }

    inline UIPC_GENERIC void ddEddx(Matrix12x12&   H,
                                    const Vector3& x0,
                                    const Vector3& x1,
                                    const Vector3& x2,
                                    const Vector3& x3,
                                    Float          L0,
                                    Float          h_bar,
                                    Float          bias_angle,
                                    Float          plastic_theta_warp,
                                    Float          plastic_theta_weft,
                                    Float          torsion_bar,
                                    Float          anchor_theta_warp,
                                    Float          anchor_theta_weft,
                                    Float          kappa_warp,
                                    Float          kappa_weft,
                                    Float          kappa_torsion,
                                    Float          kappa_poisson,
                                    Float          friction_stiffness_warp,
                                    Float          friction_stiffness_weft)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);
        Float sin_angle          = sin(angle_between_warp);

        Float theta_warp    = theta * cos_angle * cos_angle;
        Float theta_weft    = theta * sin_angle * sin_angle;
        Float theta_torsion = theta * cos_angle * sin_angle;

        Float dEdtheta;
        DSGFPOB::dEdtheta(dEdtheta,
                          kappa_warp,
                          kappa_weft,
                          kappa_torsion,
                          kappa_poisson,
                          theta_warp - plastic_theta_warp,
                          theta_weft - plastic_theta_weft,
                          theta_torsion - torsion_bar,
                          cos_angle,
                          sin_angle,
                          L0,
                          h_bar);

        Float ddEddtheta;
        DSGFPOB::ddEddtheta(
            ddEddtheta, kappa_warp, kappa_weft, kappa_torsion, kappa_poisson, cos_angle, sin_angle, L0, h_bar);

        Float theta_var_warp = theta_warp - anchor_theta_warp;
        Float theta_var_weft = theta_weft - anchor_theta_weft;

        Float dEdtheta_friction;
        DSGFPOB::dE_friction_dtheta(dEdtheta_friction,
                                    friction_stiffness_warp,
                                    friction_stiffness_weft,
                                    cos_angle,
                                    sin_angle,
                                    theta_var_warp,
                                    theta_var_weft,
                                    L0,
                                    h_bar);

        Float ddEddtheta_friction;
        DSGFPOB::ddE_friction_ddtheta(ddEddtheta_friction,
                                      friction_stiffness_warp,
                                      friction_stiffness_weft,
                                      cos_angle,
                                      sin_angle,
                                      L0,
                                      h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Matrix12x12 ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        H = dthetadx * (ddEddtheta + ddEddtheta_friction) * dthetadx.transpose()
            + (dEdtheta + dEdtheta_friction) * ddthetaddx;
    }

    inline UIPC_GENERIC void pGpKappaWarp(Vector12&      pGpP,
                                          const Vector3& x0,
                                          const Vector3& x1,
                                          const Vector3& x2,
                                          const Vector3& x3,
                                          Float          L0,
                                          Float          h_bar,
                                          Float          bias_angle,
                                          Float          bending_stiffness_warp,
                                          Float          theta_plastic_warp,
                                          Float d_plastic_theta_d_k_warp)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);

        Float theta_warp = theta * cos_angle * cos_angle;

        Float elastic_theta_warp = theta_warp - theta_plastic_warp;

        Float dEdtheta_warp = 2.0 * L0
                              * (elastic_theta_warp * std::pow(cos_angle, 2)
                                 - bending_stiffness_warp * d_plastic_theta_d_k_warp
                                       * std::pow(cos_angle, 2))
                              / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_warp * dthetadx;
    }


    inline UIPC_GENERIC void pGpKappaWeft(Vector12&      pGpP,
                                          const Vector3& x0,
                                          const Vector3& x1,
                                          const Vector3& x2,
                                          const Vector3& x3,
                                          Float          L0,
                                          Float          h_bar,
                                          Float          bias_angle,
                                          Float          bending_stiffness_weft,
                                          Float          theta_plastic_weft,
                                          Float d_plastic_theta_d_k_weft)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float angle_between_warp = bias_angle + 1.570796;
        Float sin_angle          = sin(angle_between_warp);
        Float theta_weft         = theta * sin_angle * sin_angle;
        Float elastic_theta_weft = theta_weft - theta_plastic_weft;

        //R = 2.0 * L0
        //        * (kappa_warp * theta_warp * std::pow(cos_angle, 2)
        //           + kappa_weft * theta_weft * std::pow(sin_angle, 2)
        //           + kappa_poisson
        //                 * (theta_warp * std::pow(sin_angle, 2) + theta_weft * std::pow(cos_angle, 2))
        //           + kappa_torsion * theta_torsion * sin_angle * cos_angle)
        //        / h_bar;

        Float dEdtheta_weft = 2.0 * L0
                              * (elastic_theta_weft * std::pow(sin_angle, 2)
                                 - bending_stiffness_weft * d_plastic_theta_d_k_weft
                                       * std::pow(sin_angle, 2))
                              / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_weft * dthetadx;
    }


    inline UIPC_GENERIC void pGpFrictionStiffnessWarp(Vector12&      pGpP,
                                                      const Vector3& x0,
                                                      const Vector3& x1,
                                                      const Vector3& x2,
                                                      const Vector3& x3,
                                                      Float          L0,
                                                      Float          h_bar,
                                                      Float          bias_angle,
                                                      Float anchor_theta_warp)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);

        Float theta_warp = theta * cos_angle * cos_angle;

        Float theta_var_warp = theta_warp - anchor_theta_warp;

        Float dEdtheta_friction_k_warp =
            2.0 * L0 * (theta_var_warp * std::pow(cos_angle, 2)) / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_friction_k_warp * dthetadx;
    }


    inline UIPC_GENERIC void pGpFrictionStiffnessWeft(Vector12&      pGpP,
                                                      const Vector3& x0,
                                                      const Vector3& x1,
                                                      const Vector3& x2,
                                                      const Vector3& x3,
                                                      Float          L0,
                                                      Float          h_bar,
                                                      Float          bias_angle,
                                                      Float anchor_theta_weft)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float angle_between_warp = bias_angle + 1.570796;
        Float sin_angle          = sin(angle_between_warp);
        Float theta_weft         = theta * sin_angle * sin_angle;

        Float theta_var_weft = theta_weft - anchor_theta_weft;

        Float dEdtheta_friction_k_weft =
            2.0 * L0 * (theta_var_weft * std::pow(sin_angle, 2)) / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_friction_k_weft * dthetadx;
    }


    inline UIPC_GENERIC void pGpThetaThresWarp(Vector12&      pGpP,
                                               const Vector3& x0,
                                               const Vector3& x1,
                                               const Vector3& x2,
                                               const Vector3& x3,
                                               Float          L0,
                                               Float          h_bar,
                                               Float          bias_angle,
                                               Float d_anchor_theta_d_theta_theta_warp,
                                               Float friction_stiffness_warp)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Float dEdtheta_thres_warp = -2.0 * L0 * friction_stiffness_warp * d_anchor_theta_d_theta_theta_warp
                                    * std::pow(cos_angle, 2) / h_bar;

        pGpP = dEdtheta_thres_warp * dthetadx;
    }


    inline UIPC_GENERIC void pGpThetaThresWeft(Vector12&      pGpP,
                                               const Vector3& x0,
                                               const Vector3& x1,
                                               const Vector3& x2,
                                               const Vector3& x3,
                                               Float          L0,
                                               Float          h_bar,
                                               Float          bias_angle,
                                               Float d_anchor_theta_d_theta_theta_weft,
                                               Float friction_stiffness_weft)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float angle_between_warp = bias_angle + 1.570796;
        Float sin_angle          = sin(angle_between_warp);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Float dEdtheta_thres_weft = -2.0 * L0 * friction_stiffness_weft * d_anchor_theta_d_theta_theta_weft
                                    * std::pow(sin_angle, 2) / h_bar;

        pGpP = dEdtheta_thres_weft * dthetadx;
    }

    inline UIPC_GENERIC void pGpYieldOriWarp(Vector12&      pGpP,
                                             const Vector3& x0,
                                             const Vector3& x1,
                                             const Vector3& x2,
                                             const Vector3& x3,
                                             Float          L0,
                                             Float          bias_angle,
                                             Float          h_bar,
                                             Float bending_stiffness_warp,
                                             Float d_theta_plastic_d_yield_ori_warp)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);

        Float dEdtheta_warp = -L0 * bending_stiffness_warp * 2 * d_theta_plastic_d_yield_ori_warp
                              * std::pow(cos_angle, 2) / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_warp * dthetadx;
    }

    inline UIPC_GENERIC void pGpYieldOriWeft(Vector12&      pGpP,
                                             const Vector3& x0,
                                             const Vector3& x1,
                                             const Vector3& x2,
                                             const Vector3& x3,
                                             Float          L0,
                                             Float          bias_angle,
                                             Float          h_bar,
                                             Float bending_stiffness_weft,
                                             Float d_theta_plastic_d_yield_ori_weft)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float angle_between_warp = bias_angle + 1.570796;
        Float sin_angle          = sin(angle_between_warp);

        Float dEdtheta_weft = -L0 * bending_stiffness_weft * 2 * d_theta_plastic_d_yield_ori_weft
                              * std::pow(sin_angle, 2) / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_weft * dthetadx;
    }

    inline UIPC_GENERIC void pGpHardeningKWarp(Vector12&      pGpP,
                                               const Vector3& x0,
                                               const Vector3& x1,
                                               const Vector3& x2,
                                               const Vector3& x3,
                                               Float          L0,
                                               Float          h_bar,
                                               Float          bias_angle,
                                               Float bending_stiffness_warp,
                                               Float d_theta_plastic_d_hardening_k_warp)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float angle_between_warp = bias_angle + 1.570796;
        Float cos_angle          = cos(angle_between_warp);

        Float dEdtheta_warp = -L0 * bending_stiffness_warp * 2 * d_theta_plastic_d_hardening_k_warp
                              * std::pow(cos_angle, 2) / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_warp * dthetadx;
    }

    inline UIPC_GENERIC void pGpHardeningKWeft(Vector12&      pGpP,
                                               const Vector3& x0,
                                               const Vector3& x1,
                                               const Vector3& x2,
                                               const Vector3& x3,
                                               Float          L0,
                                               Float          h_bar,
                                               Float          bias_angle,
                                               Float bending_stiffness_weft,
                                               Float d_theta_plastic_d_hardening_k_weft)
    {
        namespace DSGFPOB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        Float angle_between_warp = bias_angle + 1.570796;
        Float sin_angle          = sin(angle_between_warp);

        Float dEdtheta_weft = -L0 * bending_stiffness_weft * 2 * d_theta_plastic_d_hardening_k_weft
                              * std::pow(sin_angle, 2) / h_bar;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        pGpP = dEdtheta_weft * dthetadx;
    }

}  // namespace sym::discrete_shell_friction_hardening_plastic_orthotropic_bending
}  // namespace uipc::backend::cuda
