#include <contact_system/vertex_half_plane_normal_contact.h>
#include <implicit_geometry/half_plane.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <contact_system/vertex_half_plane_normal_contact_diff_parm.h>
#include <kernel_cout.h>
#include <affine_body/constitutions/diff_area.h>
#include <affine_body/abd_diff_sim_manager.h>
#include <utils/matrix_assembly_utils.h>

namespace uipc::backend::cuda
{
class IPCVertexHalfPlaneNormalContact final : public VertexHalfPlaneNormalContact
{
  public:
    using VertexHalfPlaneNormalContact::VertexHalfPlaneNormalContact;

    HalfPlane*                     half_plane                 = nullptr;
    GlobalSimpicialSurfaceManager* simplicial_surface_manager = nullptr;

    virtual void do_build(BuildInfo& info) override
    {
        auto constitution =
            world().scene().info()["contact"]["constitution"].get<std::string>();
        if(constitution != "ipc")
        {
            throw SimSystemException("Constitution is not IPC");
        }

        half_plane                 = &require<HalfPlane>();
        simplicial_surface_manager = &require<GlobalSimpicialSurfaceManager>();
    }

    virtual void do_compute_energy(EnergyInfo& info)
    {
        using namespace muda;

        auto ssm = simplicial_surface_manager;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.PHs().size(),
                   [Es  = info.energies().viewer().name("Es"),
                    PHs = info.PHs().viewer().name("PHs"),
                    plane_positions = half_plane->positions().viewer().name("plane_positions"),
                    plane_normals = half_plane->normals().viewer().name("plane_normals"),
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    half_plane_contact_ids = half_plane->contact_ids().viewer().name("half_plane_contact_ids"),
                    Ps = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt(),
                    v2s   = ssm->v2s().viewer().name("v2s"),
                    vf    = ssm->surf_verts_triangles().viewer().name("vf"),
                    vfo = ssm->surf_verts_triangle_offsets().viewer().name("vfo"),
                    vfc = ssm->surf_verts_triangle_counts().viewer().name("vfc"),
                    rest_positions = info.rest_positions().viewer().name("rest_positions"),
                    surf_triangles = ssm->surf_triangles().viewer().name(
                        "surf_triangles")] __device__(int I) mutable
                   {
                       Vector2i PH = PHs(I);

                       IndexT vI = PH(0);
                       IndexT HI = PH(1);

                       Vector3 v = Ps(vI);
                       Vector3 P = plane_positions(HI);
                       Vector3 N = plane_normals(HI);

                       Float kappa =
                           table(contact_ids(vI), half_plane_contact_ids(HI)).kappa;

                       Float thickness = thicknesses(vI);

                       Float  A      = 0.0;
                       IndexT svI    = v2s(vI);
                       IndexT count  = vfc(svI);
                       IndexT offset = vfo(svI);

                       for(IndexT i = 0; i < count; ++i)
                       {
                           IndexT   FI = vf(offset + i);
                           Vector3i F  = surf_triangles(FI);

                           Vector9 X0;
                           X0.segment<3>(0) = rest_positions(F(0));
                           X0.segment<3>(3) = rest_positions(F(1));
                           X0.segment<3>(6) = rest_positions(F(2));

                           Float local_A;
                           sym::diff_area::A(local_A, X0);
                           A += local_A;
                       }

                       A /= 3;  // A is the sum of the areas of all triangles / 3

                       Es(I) = /*A **/ dt * dt
                               * sym::ipc_vertex_half_contact::PH_barrier_energy(
                                   kappa, d_hat, thickness, v, P, N);
                   });
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;

        auto ssm = simplicial_surface_manager;

        if(info.PHs().size())
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    info.PHs().size(),
                    [Grad = info.gradients().viewer().name("Grad"),
                     Hess = info.hessians().viewer().name("Hess"),
                     PHs  = info.PHs().viewer().name("PHs"),
                     plane_positions = half_plane->positions().viewer().name("plane_positions"),
                     plane_normals = half_plane->normals().viewer().name("plane_normals"),
                     table = info.contact_tabular().viewer().name("contact_tabular"),
                     contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                     half_plane_contact_ids = half_plane->contact_ids().viewer().name("half_plane_contact_ids"),
                     Ps = info.positions().viewer().name("Ps"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     eps_v = info.eps_velocity(),
                     d_hat = info.d_hat(),
                     dt    = info.dt(),
                     v2s   = ssm->v2s().viewer().name("v2s"),
                     vf    = ssm->surf_verts_triangles().viewer().name("vf"),
                     vfo = ssm->surf_verts_triangle_offsets().viewer().name("vfo"),
                     vfc = ssm->surf_verts_triangle_counts().viewer().name("vfc"),
                     rest_positions = info.rest_positions().viewer().name("rest_positions"),
                     surf_triangles = ssm->surf_triangles().viewer().name(
                         "surf_triangles")] __device__(int I) mutable
                    {
                        Vector2i PH = PHs(I);

                        IndexT vI = PH(0);
                        IndexT HI = PH(1);

                        Vector3 v = Ps(vI);
                        Vector3 P = plane_positions(HI);
                        Vector3 N = plane_normals(HI);

                        Float kappa =
                            table(contact_ids(vI), half_plane_contact_ids(HI)).kappa;

                        Float thickness = thicknesses(vI);

                        Float  A      = 0.0;
                        IndexT svI    = v2s(vI);
                        IndexT count  = vfc(svI);
                        IndexT offset = vfo(svI);

                        for(IndexT i = 0; i < count; ++i)
                        {
                            IndexT   FI = vf(offset + i);
                            Vector3i F  = surf_triangles(FI);

                            Vector9 X0;
                            X0.segment<3>(0) = rest_positions(F(0));
                            X0.segment<3>(3) = rest_positions(F(1));
                            X0.segment<3>(6) = rest_positions(F(2));

                            Float local_A;
                            sym::diff_area::A(local_A, X0);
                            A += local_A;
                        }

                        A /= 3;  // A is the sum of the areas of all triangles / 3

                        sym::ipc_vertex_half_contact::PH_barrier_gradient_hessian(
                            Grad(I), Hess(I), kappa, d_hat, thickness, v, P, N);

                        Float Adt2 = /*A **/ dt * dt;
                        Grad(I) *= Adt2;
                        Hess(I) *= Adt2;
                    });
        }
    }
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneNormalContact);

class IPCVertexHalfPlaneNormalContactDiffParm final : public VertexHalfPlaneNormalContactDiffParm
{
  public:
    using VertexHalfPlaneNormalContactDiffParm::VertexHalfPlaneNormalContactDiffParm;

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

        if(info.PHs().size())
        {
            auto eps_vh = info.eps_velocity() * info.dt();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    info.PHs().size(),
                    [pGpPs = info.pGpP().viewer().name("pGpPs"),
                     PHs   = info.PHs().viewer().name("PHs"),
                     half_plane_Ps = half_plane->positions().viewer().name("plane_positions"),
                     half_plane_Ns = half_plane->normals().viewer().name("plane_normals"),
                     contact_tabular = info.contact_tabular().viewer().name("contact_tabular"),
                     diff_contact_tabular = info.diff_contact_tabular().viewer().name("diff_contact_tabular"),
                     contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                     half_plane_contact_ids = half_plane->contact_ids().viewer().name("half_plane_contact_ids"),
                     Ps      = info.positions().viewer().name("Ps"),
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
                        Float diff_kappa = diff_coeff.diff_kappa;

                        Vector3 pGpkappa = Vector3::Zero();


                        VHC::PH_barrier_pGpkappa(
                            pGpkappa, kappa, Ps(I), plane_pos, plane_normal, d_hat, thickness);
                        pGpkappa *= dt * dt;


                        // write `0s` to safe place if diff_mu or diff_kappa is negative
                        if(diff_kappa < 0)
                        {
                            pGpkappa.setZero();
                            diff_kappa = 0;
                        }

                        pGpPs(I).write(vI, diff_kappa, pGpkappa);
                    });
        }
    }
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneNormalContactDiffParm);
}  // namespace uipc::backend::cuda
