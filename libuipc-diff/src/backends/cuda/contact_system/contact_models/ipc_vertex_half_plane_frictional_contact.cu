#include <contact_system/vertex_half_plane_frictional_contact.h>
#include <implicit_geometry/half_plane.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <contact_system/contact_diff_dof_reporter.h>
#include <kernel_cout.h>
#include <collision_detection/global_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <contact_system/vertex_half_plane_frictional_contact_diff_dof.h>
#include <contact_system/vertex_half_plane_frictional_contact_diff_parm.h>

namespace uipc::backend::cuda
{
class IPCVertexHalfPlaneFrictionalContact final : public VertexHalfPlaneFrictionalContact
{
  public:
    using VertexHalfPlaneFrictionalContact::VertexHalfPlaneFrictionalContact;

    virtual void do_build(BuildInfo& info) override
    {
        auto constitution =
            world().scene().info()["contact"]["constitution"].get<std::string>();
        if(constitution != "ipc")
        {
            throw SimSystemException("Constitution is not IPC");
        }

        half_plane = &require<HalfPlane>();
    }

    virtual void do_compute_energy(EnergyInfo& info)
    {
        using namespace muda;
        using namespace sym::ipc_vertex_half_contact;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_PHs().size(),
                   [Es  = info.energies().viewer().name("Es"),
                    PHs = info.friction_PHs().viewer().name("PHs"),
                    plane_positions = half_plane->positions().viewer().name("plane_positions"),
                    plane_normals = half_plane->normals().viewer().name("plane_normals"),
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    half_plane_contact_ids = half_plane->contact_ids().viewer().name("half_plane_contact_ids"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int I) mutable
                   {
                       Vector2i PH = PHs(I);

                       IndexT vI = PH(0);
                       IndexT HI = PH(1);

                       Vector3 x      = Ps(vI);
                       Vector3 prev_x = prev_Ps(vI);
                       Vector3 P      = plane_positions(HI);
                       Vector3 N      = plane_normals(HI);

                       ContactCoeff coeff =
                           table(contact_ids(vI), half_plane_contact_ids(HI));
                       Float kappa = coeff.kappa * dt * dt;
                       Float mu    = coeff.mu;

                       Float D_hat = d_hat * d_hat;

                       Float thickness = thicknesses(vI);

                       Es(I) = PH_friction_energy(
                           kappa, d_hat, thickness, mu, eps_v * dt, prev_x, x, P, N);
                   });
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using namespace sym::ipc_vertex_half_contact;

        if(info.friction_PHs().size())
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(info.friction_PHs().size(),
                       [Grad = info.gradients().viewer().name("Grad"),
                        Hess = info.hessians().viewer().name("Hess"),
                        PHs  = info.friction_PHs().viewer().name("PHs"),
                        plane_positions = half_plane->positions().viewer().name("plane_positions"),
                        plane_normals = half_plane->normals().viewer().name("plane_normals"),
                        table = info.contact_tabular().viewer().name("contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        half_plane_contact_ids =
                            half_plane->contact_ids().viewer().name("half_plane_contact_ids"),
                        Ps = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        eps_v = info.eps_velocity(),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int I) mutable
                       {
                           Vector2i PH = PHs(I);

                           IndexT vI = PH(0);
                           IndexT HI = PH(1);

                           Vector3 v      = Ps(vI);
                           Vector3 prev_v = prev_Ps(vI);
                           Vector3 P      = plane_positions(HI);
                           Vector3 N      = plane_normals(HI);

                           ContactCoeff coeff =
                               table(contact_ids(vI), half_plane_contact_ids(HI));
                           Float kappa = coeff.kappa;
                           Float mu    = coeff.mu;

                           Float thickness = thicknesses(vI);

                           Float     dt2 = dt * dt;
                           Vector3   G;
                           Matrix3x3 H;

                           PH_friction_gradient_hessian(
                               G, H, kappa, d_hat, thickness, mu, eps_v * dt, prev_v, v, P, N);

                           Grad(I) = G * dt2;
                           Hess(I) = H * dt2;
                       });
        }
    }

    HalfPlane* half_plane = nullptr;
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneFrictionalContact);

class IPCVertexHalfPlaneFrictionalContactDiffDof final : public VertexHalfPlaneFrictionalContactDiffDof
{
  public:
    using VertexHalfPlaneFrictionalContactDiffDof::VertexHalfPlaneFrictionalContactDiffDof;

    using Matrix2x3 = Eigen::Matrix<Float, 2, 3>;

    virtual void do_build(BuildInfo&) override
    {
        half_plane = &require<HalfPlane>();
    }

    virtual void do_init(InitInfo&) override {}


    virtual void do_assemble(DiffDofInfo& info) override
    {
        using namespace muda;
        namespace CIC = sym::codim_ipc_contact;
        namespace VHC = sym::ipc_vertex_half_contact;

        auto half_plane_positions   = half_plane->positions();
        auto half_plane_normals     = half_plane->normals();
        auto half_plane_contact_ids = half_plane->contact_ids();
        auto eps_vh                 = info.eps_velocity() * info.dt();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                info.friction_PHs().size(),
                [PHs = info.friction_PHs().viewer().name("PHs"),
                 contact_tabular = info.contact_tabular().viewer().name("contact_tabluar"),
                 contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                 half_plane_contact_ids = half_plane_contact_ids.viewer().name("half_plane_contact_ids"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                 Ps          = info.positions().viewer().name("Ps"),
                 half_plane_Ps = half_plane_positions.viewer().name("half_plane_Ps"),
                 half_plane_Ns = half_plane_normals.viewer().name("half_plane_Ns"),
                 DiffDof = info.H().viewer().name("H"),
                 d_hat   = info.d_hat(),
                 dt      = info.dt(),
                 eps_vh  = eps_vh] __device__(int I) mutable
                {
                    Vector2i PH = PHs(I);

                    IndexT vI = PH(0);
                    IndexT HI = PH(1);

                    Vector3 plane_pos    = half_plane_Ps(HI);
                    Vector3 plane_normal = half_plane_Ns(HI);

                    Float thickness = thicknesses(vI);

                    ContactCoeff coeff =
                        contact_tabular(contact_ids(vI), half_plane_contact_ids(HI));
                    Float mu    = coeff.mu;
                    Float kappa = coeff.kappa;

                    Matrix2x3 T;
                    Vector3   e1, e2;
                    VHC::compute_tan_basis(e1, e2, plane_normal);

                    T.row(0) = e1.transpose();
                    T.row(1) = e2.transpose();

                    Vector3 xn = prev_Ps(vI);
                    Vector3 x  = Ps(vI);

                    Vector2 tau = T * (x - xn);

                    Vector2 dfdtau;
                    CIC::dfdtau(dfdtau, eps_vh, tau);


                    Matrix2x2 ddfddtau;
                    CIC::ddfddtau(ddfddtau, eps_vh, tau);

                    Float D;
                    VHC::HalfPlaneD(D, xn, plane_pos, plane_normal);
                    Vector3 dDdx;
                    VHC::dHalfPlaneDdx(dDdx, xn, plane_pos, plane_normal);

                    Float N = CIC::normal_force(kappa, d_hat, thickness, D);
                    Float dNdD = CIC::d_normal_force_dD(kappa, d_hat, thickness, D);

                    Float dt2 = dt * dt;

                    //tex:
                    //$$
                    //\frac{\partial^2 E}{\partial x_j \partial x^n_k} =
                    //
                    //\mu \Delta t^2 \left(T^T \frac{\partial f}{\partial \tau} \right) \left(\frac{\partial N^n}{\partial D}\frac{\partial D}{\partial x^n} \right)^T
                    //
                    //+ \mu \Delta t^2 N^n\left(-T^T\frac{\partial^2 f}{\partial \tau^2} T\right)
                    //$$
                    Matrix3x3 H = dt2 * mu
                                  * ((T.transpose() * dfdtau) * dNdD * dDdx.transpose()
                                     + N * (-T.transpose() * ddfddtau * T));

                    DiffDof(I).write(vI, vI, H);
                });
    }

    HalfPlane* half_plane = nullptr;
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneFrictionalContactDiffDof);

class IPCVertexHalfPlaneFrictionalContactDiffParm final : public VertexHalfPlaneFrictionalContactDiffParm
{
  public:
    using VertexHalfPlaneFrictionalContactDiffParm::VertexHalfPlaneFrictionalContactDiffParm;

    using Matrix2x3 = Eigen::Matrix<Float, 2, 3>;

    HalfPlane* half_plane = nullptr;

    virtual void do_build(BuildInfo& info) override
    {
        half_plane = &require<HalfPlane>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        namespace CIC = sym::codim_ipc_contact;
        namespace VHC = sym::ipc_vertex_half_contact;

        if(info.friction_PHs().size())
        {
            auto eps_vh = info.eps_velocity() * info.dt();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(info.friction_PHs().size(),
                       [pGpPs = info.pGpP().viewer().name("pGpPs"),
                        PHs   = info.friction_PHs().viewer().name("PHs"),
                        half_plane_Ps = half_plane->positions().viewer().name("plane_positions"),
                        half_plane_Ns = half_plane->normals().viewer().name("plane_normals"),
                        contact_tabular = info.contact_tabular().viewer().name("contact_tabular"),
                        diff_contact_tabular =
                            info.diff_contact_tabular().viewer().name("diff_contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        half_plane_contact_ids =
                            half_plane->contact_ids().viewer().name("half_plane_contact_ids"),
                        Ps = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        eps_v = info.eps_velocity(),
                        d_hat = info.d_hat(),
                        dt    = info.dt(),
                        eps_vh] __device__(int I) mutable
                       {
                           Vector2i PH = PHs(I);

                           IndexT vI = PH(0);
                           IndexT HI = PH(1);

                           Vector3 plane_pos    = half_plane_Ps(HI);
                           Vector3 plane_normal = half_plane_Ns(HI);

                           Float thickness = thicknesses(vI);

                           ContactCoeff coeff =
                               contact_tabular(contact_ids(vI), half_plane_contact_ids(HI));
                           Float mu    = coeff.mu;
                           Float kappa = coeff.kappa;

                           DiffContactCoeff diff_coeff =
                               diff_contact_tabular(contact_ids(vI),
                                                    half_plane_contact_ids(HI));
                           Float diff_mu    = diff_coeff.diff_mu;
                           Float diff_kappa = diff_coeff.diff_kappa;

                           Vector3 pGpkappa = Vector3::Zero();
                           Vector3 pGpmu    = Vector3::Zero();


                           VHC::PH_friction_pGpkappa_pGpmu(pGpkappa,
                                                           pGpmu,
                                                           kappa,
                                                           mu,
                                                           prev_Ps(vI),
                                                           Ps(vI),
                                                           plane_pos,
                                                           plane_normal,
                                                           d_hat,
                                                           thickness,
                                                           eps_vh);

                           pGpmu *= dt * dt;
                           pGpkappa *= dt * dt;


                           // write `0s` to safe place if diff_mu or diff_kappa is negative
                           if(diff_kappa < 0)
                           {
                               pGpkappa.setZero();
                               diff_kappa = 0;
                           }

                           if(diff_mu < 0)
                           {
                               pGpmu.setZero();
                               diff_mu = 0;
                           }

                           pGpPs(2 * I + 0).write(vI, diff_kappa, pGpkappa);
                           pGpPs(2 * I + 1).write(vI, diff_mu, pGpmu);

                           // cout << "pGpmu: " << pGpmu.transpose().eval() << "\n";
                       });
        }
    }
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneFrictionalContactDiffParm);
}  // namespace uipc::backend::cuda
