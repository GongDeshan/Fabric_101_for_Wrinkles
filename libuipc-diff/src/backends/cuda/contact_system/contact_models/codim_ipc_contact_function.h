#pragma once
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/ext/eigen/evd.h>
#include <utils/distance.h>
#include <utils/distance/distance_flagged.h>
#include <muda/ext/eigen/evd.h>
#include <utils/friction_utils.h>

namespace uipc::backend::cuda
{
namespace sym::codim_ipc_contact
{
#include "sym/codim_ipc_contact.inl"
}

namespace sym::codim_ipc_contact
{
    inline UIPC_GENERIC ContactCoeff PT_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                      const Vector4i& cids)
    {
        Float kappa = 0.0;
        Float mu    = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
            mu += coeff.mu;
        }
        return {kappa / 3.0, mu / 3.0};
    }

    inline UIPC_GENERIC DiffContactCoeff PT_diff_contact_coeff(
        const muda::CDense2D<DiffContactCoeff>& table, const Vector4i& cids)
    {
        MUDA_ASSERT(cids[1] == cids[2] && cids[2] == cids[3],
                    "(In Diff Sim) The contact ids should be the same for PT contact. But yours (%d,%d,%d)",
                    cids[1],
                    cids[2],
                    cids[3]);

        return table(cids[0], cids[1]);
    }

    inline UIPC_GENERIC ContactCoeff EE_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                      const Vector4i& cids)
    {
        Float kappa = 0.0;
        Float mu    = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                ContactCoeff coeff = table(cids[j], cids[k]);
                kappa += coeff.kappa;
                mu += coeff.mu;
            }
        }
        return {kappa / 4.0, mu / 4.0};
    }

    inline UIPC_GENERIC DiffContactCoeff EE_diff_contact_coeff(
        const muda::CDense2D<DiffContactCoeff>& table, const Vector4i& cids)
    {
        MUDA_ASSERT(cids[0] == cids[1] && cids[2] == cids[3],
                    "(In Diff Sim) The contact ids should be the same for EE contact. But yours (%d,%d)-(%d,%d)",
                    cids[0],
                    cids[1],
                    cids[2],
                    cids[3]);

        return table(cids[0], cids[2]);
    }

    inline UIPC_GENERIC ContactCoeff PE_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                      const Vector3i& cids)
    {
        Float kappa = 0.0;
        Float mu    = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
            mu += coeff.mu;
        }
        return {kappa / 2.0, mu / 2.0};
    }

    inline UIPC_GENERIC DiffContactCoeff PE_diff_contact_coeff(
        const muda::CDense2D<DiffContactCoeff>& table, const Vector3i& cids)
    {
        MUDA_ASSERT(cids[1] == cids[2],
                    "(In Diff Sim) The contact ids should be the same for PE contact. But yours (%d,%d)",
                    cids[1],
                    cids[2]);

        return table(cids[0], cids[1]);
    }

    inline UIPC_GENERIC ContactCoeff PP_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                      const Vector2i& cids)
    {
        return table(cids[0], cids[1]);
    }

    inline UIPC_GENERIC DiffContactCoeff PP_diff_contact_coeff(
        const muda::CDense2D<DiffContactCoeff>& table, const Vector2i& cids)
    {
        return table(cids[0], cids[1]);
    }

    // C1 clamping
    inline UIPC_GENERIC void f0(Float x2, Float epsvh, Float& f0)
    {
        if(x2 >= epsvh * epsvh)
        {
            //tex: $$y$$
            f0 = std::sqrt(x2);
        }
        else
        {
            //tex: $$\frac{y^{2}}{\epsilon_{x}} + \frac{1}{3 \epsilon_{x}} - \frac{y^{3}}{3 \epsilon_{x}^{2}}$$
            f0 = x2 * (-std::sqrt(x2) / 3.0 + epsvh) / (epsvh * epsvh) + epsvh / 3.0;
        }
    }

    inline UIPC_GENERIC void f1_div_rel_dx_norm(Float x2, Float epsvh, Float& result)
    {
        if(x2 >= epsvh * epsvh)
        {
            //tex: $$ \frac{1}{y}$$
            result = 1 / std::sqrt(x2);
        }
        else
        {
            //tex: $$ \frac{2 \epsilon_{x} - y}{ \epsilon_{x}^{2}}$$
            result = (-std::sqrt(x2) + 2.0 * epsvh) / (epsvh * epsvh);
        }
    }

    inline UIPC_GENERIC void f2_term(Float x2, Float epsvh, Float& term)
    {
        term = -1 / (epsvh * epsvh);
        // same for x2 >= epsvh * epsvh for C1 clamped friction
    }

    inline UIPC_GENERIC void dfdtau(Vector2& R, Float eps_vh, Vector2 tan_rel_x)
    {
        Float f1_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_val);
        R = f1_val * tan_rel_x;
    }

    inline UIPC_GENERIC void make_spd(Matrix2x2& mat)
    {
        Vector2   eigen_values;
        Matrix2x2 eigen_vectors;
        muda::eigen::template evd(mat, eigen_values, eigen_vectors);
#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            auto& v = eigen_values(i);
            v       = v < 0.0 ? 0.0 : v;
        }
        mat = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
    };

    inline UIPC_GENERIC void ddfddtau(Matrix2x2& R, Float eps_vh, Vector2 tan_rel_x)
    {
        Float sq_norm = tan_rel_x.squaredNorm();
        Float epsvh2  = eps_vh * eps_vh;
        Float f1_div_rel_dx_norm_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_div_rel_dx_norm_val);

        if(sq_norm >= epsvh2)
        {
            // no SPD projection needed
            Vector2 ubar(-tan_rel_x[1], tan_rel_x[0]);
            R = ubar * (f1_div_rel_dx_norm_val / sq_norm) * ubar.transpose();
        }
        else
        {
            if(sq_norm == 0.0)
            {
                // no SPD projection needed
                R = f1_div_rel_dx_norm_val * Matrix2x2::Identity();
            }
            else
            {
                Float f2_term_val;
                f2_term(sq_norm, eps_vh, f2_term_val);

                Float relDXNorm = std::sqrt(sq_norm);

                // only need to project the inner 2x2 matrix to SPD
                R = tan_rel_x * (f2_term_val / relDXNorm) * tan_rel_x.transpose();
                R.diagonal().array() += f1_div_rel_dx_norm_val;

                make_spd(R);
            }
        }
    }


    inline UIPC_GENERIC Float friction_energy(Float mu, Float lambda, Float eps_vh, Vector2 tan_rel_x)
    {
        Float f0_val;
        f0(tan_rel_x.squaredNorm(), eps_vh, f0_val);
        return mu * lambda * f0_val;
    }

    inline UIPC_GENERIC void friction_gradient(Vector2& G2, Float mu, Float lambda, Float eps_vh, Vector2 tan_rel_x)
    {
        Float f1_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_val);
        G2 = mu * lambda * f1_val * tan_rel_x;
    }

    inline UIPC_GENERIC void friction_hessian(
        Matrix2x2& H2x2, Float mu, Float lambda, Float eps_vh, Vector2 tan_rel_x)
    {
        Float sq_norm = tan_rel_x.squaredNorm();
        Float epsvh2  = eps_vh * eps_vh;
        Float f1_div_rel_dx_norm_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_div_rel_dx_norm_val);

        if(sq_norm >= epsvh2)
        {
            // no SPD projection needed
            Vector2 ubar(-tan_rel_x[1], tan_rel_x[0]);
            H2x2 = ubar * (mu * lambda * f1_div_rel_dx_norm_val / sq_norm)
                   * ubar.transpose();
        }
        else
        {
            if(sq_norm == 0.0)
            {
                // no SPD projection needed
                H2x2 = (mu * lambda * f1_div_rel_dx_norm_val) * Matrix2x2::Identity();
            }
            else
            {
                Float f2_term_val;
                f2_term(sq_norm, eps_vh, f2_term_val);

                Float relDXNorm = std::sqrt(sq_norm);

                // only need to project the inner 2x2 matrix to SPD
                H2x2 = tan_rel_x * (f2_term_val / relDXNorm) * tan_rel_x.transpose();
                H2x2.diagonal().array() += f1_div_rel_dx_norm_val;

                make_spd(H2x2);

                H2x2 *= mu * lambda;
            }
        }
    }

    //tex:
    //$$
    // N = \frac{\partial B}{\partial d} = \frac{\partial B}{\partial D} \frac{\partial D}{\partial d}
    //$$
    inline UIPC_GENERIC Float normal_force(Float kappa, Float d_hat, Float thickness, Float D)
    {
        using namespace distance;
        Float d = std::sqrt(D);
        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        Float dDdd = 2.0 * d;
        Float dBdd = dBdD * dDdd;
        return -dBdd;  // > 0
    }

    //tex: $$\frac{\partial N}{\partial \kappa}$$
    inline UIPC_GENERIC Float d_normal_force_d_kappa(Float d_hat, Float thickness, Float D)
    {
        // just set kappa to 1.0, cuz it's a constant multiplier
        return normal_force(1.0, d_hat, thickness, D);
    }

    inline UIPC_GENERIC Float d_normal_force_dD(Float kappa, Float d_hat, Float thickness, Float D)
    {
        using namespace distance;
        Float d    = std::sqrt(D);
        Float dDdd = 2.0 * d;
        Float dddD = 0.5 / d;
        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        Float dNdD = -2.0 * (ddBddD * d + dBdD * dddD);
        return dNdD;
    }

    inline UIPC_GENERIC void d_friction_gradient_d_kappa_mu(Vector2& pGpkappa,
                                                            Vector2& pGpmu,
                                                            Float    kappa,
                                                            Float    mu,
                                                            const Vector2& tau,
                                                            Float eps_vh,
                                                            Float d_hat,
                                                            Float thickness,
                                                            Float D)
    {
        Float lambda = normal_force(kappa, d_hat, thickness, D);

        Vector2 _dfdtau;
        dfdtau(_dfdtau, eps_vh, tau);
        //tex:
        //$$
        //\frac{\partial G}{\partial \mu} = N^n \frac{\partial f}{\partial \tau}
        //$$
        pGpmu = lambda * _dfdtau;
        //tex:
        //$$
        //\frac{\partial G}{\partial \kappa} = \mu \frac{\partial N^n}{\partial \kappa} \frac{\partial f}{\partial \tau}
        //$$
        Float dldkappa = d_normal_force_d_kappa(d_hat, thickness, D);
        pGpkappa       = mu * dldkappa * _dfdtau;
    }

    inline UIPC_GENERIC Float d_KappaBarrierGradient_d_kappa(Float D, Float d_hat, Float thickness)
    {
        Float R;
        //tex:
        //$$
        //\frac{\partial G}{\partial \kappa} = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        //we just return the $\frac{\partial B}{\partial D}$ here
        //the caller should multiply the $\frac{\partial D}{\partial x}$ by themselves
        dKappaBarrierdD(R, 1.0, D, d_hat, thickness);
        return R;
    }
}  // namespace sym::codim_ipc_contact
}  // namespace uipc::backend::cuda
