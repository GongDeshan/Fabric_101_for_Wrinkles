#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/StVK_Nonlinear_Friction_Hardening_Plastic_Shear_shell_2d_function.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <Eigen/Dense>
#include <muda/ext/eigen/inverse.h>
#include <utils/codim_thickness.h>
#include <utils/matrix_assembler.h>
#include <utils/matrix_unpacker.h>
#include <utils/matrix_assembly_utils.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/codim_2d_constitution_diff_parm_reporter.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::cuda
{
static constexpr U64 ConstitutionUID = 68;

class NonlinearFrictionHardeningPlasticStVKShell2D final : public Codim2DConstitution
{
  public:
    using Codim2DConstitution::Codim2DConstitution;

    vector<Float> h_k11s, h_k22s, h_k12s, h_k33_0s, h_k33_1s;
    vector<Float> h_strain_uus, h_strain_vvs, h_strain_uvs;
    // Friction Parameters
    vector<Float> h_friction_k33s;
    vector<Float> h_anchor_uus, h_anchor_vvs, h_anchor_uvs;  // Rest friction strains
    vector<Float> h_thres_uvs;
    // Plastic Parameters
    vector<Float> h_hardening_k33s;
    vector<Float> h_plastic_uus, h_plastic_vvs, h_plastic_uvs;  // Rest plastic strains
    vector<Float>    h_hardening_plastic_uvs;
    vector<Float>    h_yield_uvs;
    vector<Vector6>  h_rest_xs;
    vector<Vector3i> h_tri_idxs;

    muda::DeviceBuffer<Float> k11s, k22s, k12s, k33_0s, k33_1s;
    muda::DeviceBuffer<Float> strain_uus, strain_vvs, strain_uvs;
    // Friction Parameters (CUDA)
    muda::DeviceBuffer<Float> friction_k33s;
    muda::DeviceBuffer<Float> anchor_uus, anchor_vvs, anchor_uvs;  // Rest friction strains
    muda::DeviceBuffer<Float> thres_uvs;
    // Plastic Parameters (CUDA)
    muda::DeviceBuffer<Float> hardening_k33s;
    muda::DeviceBuffer<Float> plastic_uus, plastic_vvs, plastic_uvs;  // Rest plastic strains
    muda::DeviceBuffer<Float> hardening_plastic_uvs;
    muda::DeviceBuffer<Float> yield_ori_uvs, yield_uvs;  // Original Yeild Strain (For Hardening)
    muda::DeviceBuffer<Vector6>  rest_xs;
    muda::DeviceBuffer<Vector3i> tri_idxs;

    FiniteElementMethod*                 fem;
    GlobalLinearSystem* audit_linear_system = nullptr;
    std::span<S<geometry::GeometrySlot>> geo_slots;

    vector<IndexT> filter_geo_slot_indices;
    vector<SizeT>  vertex_counts, vertex_offset;

    void anchor_plastic_strain_update()
    {
        //std::cout << "Friction hardening plastic stvk strain updata" << std::endl;

        using namespace muda;
        using namespace sym::stvk_nonlinear_friction_hardening_plastic_shear_2d;

        namespace STVKFHPS = sym::stvk_nonlinear_friction_hardening_plastic_shear_2d;

        // Used when single object
        auto geo = geo_slots.front()->geometry().as<geometry::SimplicialComplex>();

        // Used when muptile objects
        //auto geo = geo_slots[filter_geo_slot_indices[0]]->geometry().as<geometry::SimplicialComplex>();

        auto rest_anchor_plastic =
            geo->triangles().find<I32>("reset_anchor_plastic")->view();

        if(rest_anchor_plastic[0] != 0)
        {
            //std::cout << "Rest Anchor Strain and Plastic Strains" << std::endl;

            auto anchor_strain_uv_view =
                geo->triangles().find<Float>("anchor_strain_uv")->view();

            //auto yield_strain_uv_view =
            //    geo->triangles().find<Float>("yield_uv")->view();

            auto plastic_strain_uv_view =
                geo->triangles().find<Float>("plastic_strain_uv")->view();

            for(int f = 0; f < h_anchor_uus.size(); ++f)
            {
                // Rest Friction Anchor Strain
                h_anchor_uvs[f] = anchor_strain_uv_view[f];
                // Rest Plastic Strain and Yield
                //h_yield_uvs[f]   = yield_strain_uv_view[f];
                h_plastic_uvs[f] = plastic_strain_uv_view[f];
            }

            anchor_uvs.view().copy_from(h_anchor_uvs.data());

            //yield_uvs.view().copy_from(h_yield_uvs.data());
            plastic_uvs.view().copy_from(h_plastic_uvs.data());
            hardening_plastic_uvs.fill(0.0);

            yield_uvs = yield_ori_uvs;
        }

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                tri_idxs.size(),
                [tri_idxs      = tri_idxs.viewer().name("tri_idx"),
                 strain_uus    = strain_uus.viewer().name("strain_uus"),
                 strain_vvs    = strain_vvs.viewer().name("strain_vvs"),
                 strain_uvs    = strain_uvs.viewer().name("strain_uvs"),
                 thres_uvs     = thres_uvs.viewer().name("thres_uvs"),
                 anchor_uus    = anchor_uus.viewer().name("anchor_uus"),
                 anchor_vvs    = anchor_vvs.viewer().name("anchor_vvs"),
                 anchor_uvs    = anchor_uvs.viewer().name("anchor_uvs"),
                 plastic_uus   = plastic_uus.viewer().name("plastic_uus"),
                 plastic_vvs   = plastic_vvs.viewer().name("plastic_vvs"),
                 plastic_uvs   = plastic_uvs.viewer().name("plastic_uvs"),
                 yield_uvs     = yield_uvs.viewer().name("yield_uvs"),
                 yield_ori_uvs = yield_ori_uvs.viewer().name("yield_ori_uvs"),
                 hardening_plastic_uvs = hardening_plastic_uvs.viewer().name("hardening_plastic_uvs"),
                 k11s   = k11s.viewer().name("k11s"),
                 k22s   = k22s.viewer().name("k22s"),
                 k33_0s = k33_0s.viewer().name("k33_0s"),
                 k33_1s = k33_1s.viewer().name("k33_1s"),
                 hardening_k33s = hardening_k33s.viewer().name("hardening_k33s"),
                 fem_yield_strain_uv = fem->yield_strain_uvs.viewer().name("fem_yeild_strain_uv"),
                 fem_plastic_strain_uv = fem->plastic_strain_uvs.viewer().name("fem_plastic_strain_uv"),
                 fem_hardening_plastic_strain_uv =
                     fem->plastic_strain_uvs.viewer().name("fem_hardening_plastic_strain_uv"),
                 fem_anchor_strain_uv = fem->anchor_strain_uvs.viewer().name("fem_anchor_strain_uv"),
                 fem_d_anchor_uv_d_thres_uv =
                     fem->d_anchor_uv_d_thres_uv.viewer().name("fem_d_anchor_uv_d_thres_uv"),
                 fem_d_plastic_uv_d_yield_ori =
                     fem->d_plastic_uv_d_yield_ori.viewer().name("d_plastic_uv_d_yield_ori"),
                 fem_d_plastic_uv_d_hardening_k33 =
                     fem->d_plastic_uv_d_hardening_k33.viewer().name("d_plastic_uv_d_hardening_k33"),
                 fem_d_plastic_uv_d_k33 = fem->d_plastic_uv_d_k33.viewer().name("d_plastic_uv_d_k33"),
                 rest_xs = rest_xs.viewer().name("rest_x"),
                 xs      = fem->xs().viewer().name("xs")] __device__(int I)
                {
                    Vector3i idx = tri_idxs(I);

                    Vector6 rest_x;

                    rest_x = rest_xs(I);

                    Vector9 curr_x;

                    curr_x(0) = xs(idx(0))(0);
                    curr_x(1) = xs(idx(0))(1);
                    curr_x(2) = xs(idx(0))(2);

                    curr_x(3) = xs(idx(1))(0);
                    curr_x(4) = xs(idx(1))(1);
                    curr_x(5) = xs(idx(1))(2);

                    curr_x(6) = xs(idx(2))(0);
                    curr_x(7) = xs(idx(2))(1);
                    curr_x(8) = xs(idx(2))(2);

                    Matrix2x2 G;
                    STVKFHPS::Strain(G, curr_x, rest_x);

                    strain_uus(I) = G(0, 0);
                    strain_vvs(I) = G(1, 1);
                    strain_uvs(I) = G(0, 1);

                    //Float yield_strain_uv     = yield_uvs(I);

                    Float yield_strain_uv_ori = yield_ori_uvs(I);
                    Float plastic_strain_uv   = plastic_uvs(I);
                    Float hardening_plastic_strain_uv = hardening_plastic_uvs(I);


                    //if(I == 0)
                    //{
                    //    printf("Yield Ori: %f", yield_strain_uv_ori);
                    //    printf("Yield UV: %f", yield_uvs(I));
                    //}

                    Float k33 = k33_0s(I) + abs(G(0, 1)) * k33_1s(I);

                    Float yield_strain_uv =
                        yield_strain_uv_ori
                        + hardening_plastic_strain_uv * (hardening_k33s(I) / k33);

                    Float elastic_strain_uv;
                    elastic_strain_uv = G(0, 1) - plastic_strain_uv;

                    bool is_yield_uv = abs(elastic_strain_uv) > yield_strain_uv;

                    if(is_yield_uv)
                    {
                        plastic_uvs(I) = plastic_strain_uv
                                         + sign(elastic_strain_uv)
                                               * (abs(elastic_strain_uv) - yield_strain_uv)
                                               * (k33 / (k33 + hardening_k33s(I)));

                        hardening_plastic_uvs(I) =
                            hardening_plastic_strain_uv
                            + (abs(elastic_strain_uv) - yield_strain_uv)
                                  * (k33 / (k33 + hardening_k33s(I)));

                        yield_uvs(I) =
                            yield_strain_uv_ori
                            + hardening_plastic_uvs(I) * (hardening_k33s(I) / k33);

                        fem_d_plastic_uv_d_yield_ori(I) =
                            -sign(elastic_strain_uv) * (k33 / (k33 + hardening_k33s(I)));

                        fem_d_plastic_uv_d_k33(I) =
                            sign(elastic_strain_uv)
                            * (hardening_plastic_strain_uv
                                   * (hardening_k33s(I) / (k33 * k33))
                                   * (k33 / (k33 + hardening_k33s(I)))
                               + (abs(elastic_strain_uv) - yield_strain_uv)
                                     * (hardening_k33s(I)
                                        / ((k33 + hardening_k33s(I))
                                           * (k33 + hardening_k33s(I)))));

                        fem_d_plastic_uv_d_hardening_k33(I) =
                            sign(elastic_strain_uv)
                            * ((-hardening_plastic_strain_uv / k33)
                                   * (k33 / (k33 + hardening_k33s(I)))
                               + (abs(elastic_strain_uv) - yield_strain_uv)
                                     * (-k33 / ((k33 + hardening_k33s(I)) * (k33 + hardening_k33s(I)))));
                    }
                    else
                    {
                        fem_d_plastic_uv_d_yield_ori(I)     = 0.0;
                        fem_d_plastic_uv_d_hardening_k33(I) = 0.0;
                        fem_d_plastic_uv_d_k33(I)           = 0.0;
                    }

                    fem_plastic_strain_uv(I) = plastic_uvs(I);
                    fem_hardening_plastic_strain_uv(I) = hardening_plastic_uvs(I);
                    fem_yield_strain_uv(I) = yield_uvs(I);

                    Float thres_strain_uv  = thres_uvs(I);
                    Float anchor_strain_uv = anchor_uvs(I);

                    Float friction_var_strain_uv;
                    friction_var_strain_uv = G(0, 1) - anchor_strain_uv;

                    bool is_slide_uv = abs(friction_var_strain_uv) > thres_strain_uv;

                    if(is_slide_uv)
                    {
                        anchor_uvs(I) = anchor_strain_uv
                                        + sign(friction_var_strain_uv)
                                              * (abs(friction_var_strain_uv) - thres_strain_uv);

                        fem_d_anchor_uv_d_thres_uv(I) = -sign(friction_var_strain_uv);

                        //printf("Slide Friction: d_anchor_d_thres: {%f}\r\n",
                        //       fem_d_anchor_uv_d_thres_uv(I));
                    }
                    else
                    {
                        fem_d_anchor_uv_d_thres_uv(I) = 0.0;
                    }

                    fem_anchor_strain_uv(I) = anchor_uvs(I);
                });

        strain_uus.view().copy_to(h_strain_uus.data());
        strain_vvs.view().copy_to(h_strain_vvs.data());
        strain_uvs.view().copy_to(h_strain_uvs.data());

        anchor_uvs.view().copy_to(h_anchor_uvs.data());
        plastic_uvs.view().copy_to(h_plastic_uvs.data());
        hardening_plastic_uvs.view().copy_to(h_hardening_plastic_uvs.data());
        yield_uvs.view().copy_to(h_yield_uvs.data());

        auto strain_uu_view = view(*geo->triangles().find<Float>("strain_uu"));
        auto strain_vv_view = view(*geo->triangles().find<Float>("strain_vv"));
        auto strain_uv_view = view(*geo->triangles().find<Float>("strain_uv"));

        auto anchor_strain_uv_view =
            view(*geo->triangles().find<Float>("anchor_strain_uv"));
        auto yield_strain_uv_view = view(*geo->triangles().find<Float>("yield_uv"));
        auto plastic_strain_uv_view =
            view(*geo->triangles().find<Float>("plastic_strain_uv"));
        auto hardening_plastic_uv_view =
            view(*geo->triangles().find<Float>("hardening_plastic_uv"));

        auto strain_uu_iter = strain_uu_view.begin();
        auto strain_vv_iter = strain_vv_view.begin();
        auto strain_uv_iter = strain_uv_view.begin();

        auto yield_uv_iter             = yield_strain_uv_view.begin();
        auto anchor_strain_uv_iter     = anchor_strain_uv_view.begin();
        auto plastic_strain_uv_iter    = plastic_strain_uv_view.begin();
        auto hardening_plastic_uv_iter = hardening_plastic_uv_view.begin();

        for(int i = 0; i < h_strain_uus.size(); ++i)
        {
            *strain_uu_iter++ = h_strain_uus[i];
            *strain_vv_iter++ = h_strain_vvs[i];
            *strain_uv_iter++ = h_strain_uvs[i];

            *anchor_strain_uv_iter++ = h_anchor_uvs[i];

            *yield_uv_iter++             = h_yield_uvs[i];
            *plastic_strain_uv_iter++    = h_plastic_uvs[i];
            *hardening_plastic_uv_iter++ = h_hardening_plastic_uvs[i];
        }

        //std::cout << "Friction hardening plastic stvk strain updata Done" << std::endl;
    }

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override
    {
        //std::cout << "Stvk build" << std::endl;

        fem = &require<FiniteElementMethod>();
        audit_linear_system = &require<GlobalLinearSystem>();

        auto& dof_predictor = require<DofPredictor>();

        dof_predictor.on_predict(*this,
                                 [&](DofPredictor::PredictInfo& info)
                                 { anchor_plastic_strain_update(); });  // Update Plastic Theta
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        std::cout << "In the (nonlinear) friction hardening plastic stvk Initialization"
                  << std::endl;

        namespace STVKFHPS = sym::stvk_nonlinear_friction_hardening_plastic_shear_2d;

        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        for(auto geo_if : info.geo_infos())
        {
            filter_geo_slot_indices.push_back(geo_if.geo_slot_index);
            vertex_counts.push_back(geo_if.vertex_count);
            vertex_offset.push_back(geo_if.vertex_offset);
        }

        // CPU resize
        h_k11s.resize(N);
        h_k22s.resize(N);
        h_k12s.resize(N);
        h_k33_0s.resize(N);
        h_k33_1s.resize(N);

        h_strain_uus.resize(N);
        h_strain_vvs.resize(N);
        h_strain_uvs.resize(N);

        // Friction
        h_friction_k33s.resize(N);
        h_thres_uvs.resize(N);

        h_anchor_uus.resize(N);
        h_anchor_vvs.resize(N);
        h_anchor_uvs.resize(N);

        // Plastic
        h_hardening_k33s.resize(N);
        h_yield_uvs.resize(N);

        h_plastic_uus.resize(N);
        h_plastic_vvs.resize(N);
        h_plastic_uvs.resize(N);

        h_hardening_plastic_uvs.resize(N);

        h_rest_xs.resize(N);
        h_tri_idxs.resize(N);

        // CUDA resize
        k11s.resize(N);
        k22s.resize(N);
        k12s.resize(N);
        k33_0s.resize(N);
        k33_1s.resize(N);

        strain_uus.resize(N);
        strain_vvs.resize(N);
        strain_uvs.resize(N);
        // Friction (CUDA)
        friction_k33s.resize(N);
        thres_uvs.resize(N);

        anchor_uus.resize(N);
        anchor_vvs.resize(N);
        anchor_uvs.resize(N);
        // Plastic (CUDA)
        hardening_k33s.resize(N);
        hardening_plastic_uvs.resize(N);
        yield_ori_uvs.resize(N);
        yield_uvs.resize(N);

        plastic_uus.resize(N);
        plastic_vvs.resize(N);
        plastic_uvs.resize(N);

        rest_xs.resize(N);
        tri_idxs.resize(N);

        list<Float>    k11_list, k12_list, k22_list, k33_0_list, k33_1_list;
        list<Float>    hardening_k33_list;
        list<Float>    friction_k33_list;
        list<Float>    thres_uv_list;
        list<Float>    yield_uv_list;
        list<Vector6>  rest_xs_list;
        list<Vector3i> tri_idx_list;

        // Save current strain
        list<Float> strain_uu_list, strain_vv_list, strain_uv_list;

        info.for_each(
            geo_slots,
            [&](geometry::SimplicialComplex& sc) -> auto
            {
                auto k11 = sc.triangles().find<Float>("k11");
                UIPC_ASSERT(k11, "k11 not found, why?");
                auto k11_view = k11->view();

                auto k12 = sc.triangles().find<Float>("k12");
                UIPC_ASSERT(k12, "k12 not found, why?");
                auto k12_view = k12->view();

                auto k22 = sc.triangles().find<Float>("k22");
                UIPC_ASSERT(k22, "k22 not found, why?");
                auto k22_view = k22->view();

                // Nonlinear Shearing
                auto k33_0 = sc.triangles().find<Float>("k33_0");
                UIPC_ASSERT(k33_0, "k33_0 not found, why?");
                auto k33_0_view = k33_0->view();

                auto k33_1 = sc.triangles().find<Float>("k33_1");
                UIPC_ASSERT(k33_1, "k33_1 not found, why?");
                auto k33_1_view = k33_1->view();

                // Friction Properties (Mesh triangles)
                auto friction_k33 = sc.triangles().find<Float>("friction_k33");
                UIPC_ASSERT(friction_k33, "friction_k33 not found, why?");
                auto friction_k33_view = friction_k33->view();

                auto anchor_uv = sc.triangles().find<Float>("anchor_strain_uv");
                UIPC_ASSERT(anchor_uv, "anchor_strain_uv not found, why?");
                auto anchor_uv_view = view(*anchor_uv);  // This property is updated later
                auto anchor_uv_view_iter = anchor_uv_view.begin();

                auto thres_uv = sc.triangles().find<Float>("thres_uv");
                UIPC_ASSERT(thres_uv, "thres_uv not found, why?");
                auto thres_uv_view = thres_uv->view();

                // Plastic Properties (Mesh triangles)
                auto hardening_k33 = sc.triangles().find<Float>("hardening_k33");
                UIPC_ASSERT(hardening_k33, "hardening_k33 not found, why?");
                auto hardening_k33_view = hardening_k33->view();

                auto plastic_uv = sc.triangles().find<Float>("plastic_strain_uv");
                UIPC_ASSERT(plastic_uv, "plastic_uv not found, why?");
                auto plastic_uv_view = view(*plastic_uv);  // This property is updated later
                auto plastic_uv_view_iter = plastic_uv_view.begin();

                auto yield_uv = sc.triangles().find<Float>("yield_uv");
                UIPC_ASSERT(yield_uv, "yield_uv not found, why?");
                auto yield_uv_view = yield_uv->view();

                //    Create plastic properties (Hardening Plastic UV)
                auto hardening_plastic_uv_attr =
                    sc.triangles().find<Float>("hardening_plastic_uv");
                if(!hardening_plastic_uv_attr)
                    hardening_plastic_uv_attr =
                        sc.triangles().create<Float>("hardening_plastic_uv", 0.0);
                std::ranges::fill(geometry::view(*hardening_plastic_uv_attr), 0.0);

                // Add strain to triangle properties
                auto strain_uu_attr = sc.triangles().find<Float>("strain_uu");
                if(!strain_uu_attr)
                    strain_uu_attr = sc.triangles().create<Float>("strain_uu", 0.0);
                std::ranges::fill(geometry::view(*strain_uu_attr), 0.0);

                auto strain_vv_attr = sc.triangles().find<Float>("strain_vv");
                if(!strain_vv_attr)
                    strain_vv_attr = sc.triangles().create<Float>("strain_vv", 0.0);
                std::ranges::fill(geometry::view(*strain_vv_attr), 0.0);

                auto strain_uv_attr = sc.triangles().find<Float>("strain_uv");
                if(!strain_uv_attr)
                    strain_uv_attr = sc.triangles().create<Float>("strain_uv", 0.0);
                std::ranges::fill(geometry::view(*strain_uv_attr), 0.0);

                // Attribute for telling if rest the friction anchor and plastic strains
                auto reset_anchor_plastic = sc.triangles().find<I32>("reset_anchor_plastic");
                if(!reset_anchor_plastic)
                    reset_anchor_plastic =
                        sc.triangles().create<I32>("reset_anchor_plastic", 0);
                std::ranges::fill(geometry::view(*reset_anchor_plastic), 0);

                auto rest_x_attr = sc.triangles().find<Vector6>("rest_x");
                UIPC_ASSERT(rest_x_attr, "rest_x not found, why?");
                auto rest_x_view = rest_x_attr->view();

                auto triangles = sc.triangles().topo().view();

                auto position_view =
                    sc.vertices().find<Vector3>(builtin::position)->view();

                for(unsigned int tri = 0; tri < sc.triangles().size(); ++tri)
                {
                    Float k11 = k11_view[tri];
                    k11_list.push_back(k11);

                    Float k12 = k12_view[tri];
                    k12_list.push_back(k12);

                    Float k22 = k22_view[tri];
                    k22_list.push_back(k22);

                    Float k33_0 = k33_0_view[tri];
                    k33_0_list.push_back(k33_0);

                    Float k33_1 = k33_1_view[tri];
                    k33_1_list.push_back(k33_1);

                    Float hardening_k33 = hardening_k33_view[tri];
                    hardening_k33_list.push_back(hardening_k33);

                    Float friction_k33 = friction_k33_view[tri];
                    friction_k33_list.push_back(friction_k33);

                    Float thres_uv = thres_uv_view[tri];
                    thres_uv_list.push_back(thres_uv);

                    Float yield_uv = yield_uv_view[tri];
                    yield_uv_list.push_back(yield_uv);

                    // Tri Vertices Indices
                    Vector3i tri_v_idx = triangles[tri];
                    Vector3i tri_v_idx_offset =
                        tri_v_idx + Vector3i::Ones() * vertex_offset[0];
                    tri_idx_list.push_back(tri_v_idx_offset);

                    //printf("face index: %d, %d, %d\r\n",
                    //       tri_v_idx_offset(0),
                    //       tri_v_idx_offset(1),
                    //       tri_v_idx_offset(2));

                    Vector6 rest_x = rest_x_view[tri];
                    rest_xs_list.push_back(rest_x);

                    //printf("face rest x:  %f, %f, %f, %f, %f, %f\r\n",
                    //       rest_x(0),
                    //       rest_x(1),
                    //       rest_x(2),
                    //       rest_x(3),
                    //       rest_x(4),
                    //       rest_x(5));

                    Vector9 curr_x;

                    curr_x(0) = position_view[tri_v_idx(0)](0);
                    curr_x(1) = position_view[tri_v_idx(0)](1);
                    curr_x(2) = position_view[tri_v_idx(0)](2);

                    curr_x(3) = position_view[tri_v_idx(1)](0);
                    curr_x(4) = position_view[tri_v_idx(1)](1);
                    curr_x(5) = position_view[tri_v_idx(1)](2);

                    curr_x(6) = position_view[tri_v_idx(2)](0);
                    curr_x(7) = position_view[tri_v_idx(2)](1);
                    curr_x(8) = position_view[tri_v_idx(2)](2);

                    Matrix2x2 G;
                    STVKFHPS::Strain(G, curr_x, rest_x);

                    Float strain_uu = G(0, 0);
                    Float strain_vv = G(1, 1);
                    Float strain_uv = G(0, 1);

                    // Strain
                    strain_uu_list.push_back(strain_uu);
                    strain_vv_list.push_back(strain_vv);
                    strain_uv_list.push_back(strain_uv);

                    // Set the friction anchor strain and plastic strain to the current strain (no elastic deformation nor relative sliding friction)
                    *anchor_uv_view_iter++  = strain_uv;
                    *plastic_uv_view_iter++ = strain_uv;
                }
            });

        // Move from Host to Device
        std::ranges::move(k11_list, h_k11s.begin());
        std::ranges::move(k12_list, h_k12s.begin());
        std::ranges::move(k22_list, h_k22s.begin());
        std::ranges::move(k33_0_list, h_k33_0s.begin());
        std::ranges::move(k33_1_list, h_k33_1s.begin());

        std::ranges::move(strain_uu_list, h_strain_uus.begin());
        std::ranges::move(strain_vv_list, h_strain_vvs.begin());
        std::ranges::move(strain_uv_list, h_strain_uvs.begin());

        std::ranges::move(friction_k33_list, h_friction_k33s.begin());
        std::ranges::move(thres_uv_list, h_thres_uvs.begin());

        std::ranges::move(strain_uu_list, h_anchor_uus.begin());
        std::ranges::move(strain_vv_list, h_anchor_vvs.begin());
        std::ranges::move(strain_uv_list, h_anchor_uvs.begin());

        std::ranges::move(hardening_k33_list, h_hardening_k33s.begin());
        std::ranges::move(yield_uv_list, h_yield_uvs.begin());

        std::ranges::move(strain_uu_list, h_plastic_uus.begin());
        std::ranges::move(strain_vv_list, h_plastic_vvs.begin());
        std::ranges::move(strain_uv_list, h_plastic_uvs.begin());

        std::ranges::move(rest_xs_list, h_rest_xs.begin());
        std::ranges::move(tri_idx_list, h_tri_idxs.begin());

        k11s.view().copy_from(h_k11s.data());
        k12s.view().copy_from(h_k12s.data());
        k22s.view().copy_from(h_k22s.data());
        k33_0s.view().copy_from(h_k33_0s.data());
        k33_1s.view().copy_from(h_k33_1s.data());

        strain_uus.view().copy_from(h_strain_uus.data());
        strain_vvs.view().copy_from(h_strain_vvs.data());
        strain_uvs.view().copy_from(h_strain_uvs.data());

        friction_k33s.view().copy_from(h_friction_k33s.data());

        anchor_uus.view().copy_from(h_anchor_uus.data());
        anchor_vvs.view().copy_from(h_anchor_vvs.data());
        anchor_uvs.view().copy_from(h_anchor_uvs.data());

        fem->anchor_strain_uvs.resize(h_anchor_uvs.size());
        fem->anchor_strain_uvs.view().copy_from(h_anchor_uvs.data());

        thres_uvs.view().copy_from(h_thres_uvs.data());

        hardening_k33s.view().copy_from(h_hardening_k33s.data());

        plastic_uus.view().copy_from(h_plastic_uus.data());
        plastic_vvs.view().copy_from(h_plastic_vvs.data());
        plastic_uvs.view().copy_from(h_plastic_uvs.data());

        fem->plastic_strain_uvs.resize(h_plastic_uvs.size());
        fem->plastic_strain_uvs.view().copy_from(h_plastic_uvs.data());

        hardening_plastic_uvs.fill(0.0);
        hardening_plastic_uvs.view().copy_to(h_hardening_plastic_uvs.data());  // Copy from Device to Host
        fem->hardening_plastic_strain_uvs.reserve(N);
        fem->hardening_plastic_strain_uvs.fill(0.0);

        yield_uvs.view().copy_from(h_yield_uvs.data());
        yield_ori_uvs.view().copy_from(h_yield_uvs.data());
        fem->yield_strain_uvs.resize(h_yield_uvs.size());
        fem->yield_strain_uvs.view().copy_from(h_yield_uvs.data());

        // Initialize the replayer properties for calculate gradient
        fem->d_anchor_uv_d_thres_uv.reserve(N);
        fem->d_anchor_uv_d_thres_uv.fill(0.0);
        fem->d_plastic_uv_d_yield_ori.reserve(N);
        fem->d_plastic_uv_d_yield_ori.fill(0.0);
        fem->d_plastic_uv_d_k33.reserve(N);
        fem->d_plastic_uv_d_k33.fill(0.0);
        fem->d_plastic_uv_d_hardening_k33.reserve(N);
        fem->d_plastic_uv_d_hardening_k33.fill(0.0);

        rest_xs.view().copy_from(h_rest_xs.data());
        tri_idxs.view().copy_from(h_tri_idxs.data());

        //std::cout << "friction hardening plastic stvk initialization done" << std::endl;
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace STVKFHPS = sym::stvk_nonlinear_friction_hardening_plastic_shear_2d;

        //std::cout << "Computer Energy" << std::endl;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [k11s   = k11s.cviewer().name("k11"),
                    k12s   = k12s.cviewer().name("k12"),
                    k22s   = k22s.cviewer().name("k22"),
                    k33_0s = k33_0s.cviewer().name("k33_0"),
                    k33_1s = k33_1s.cviewer().name("k33_1"),
                    friction_k33s = friction_k33s.cviewer().name("friction_k33"),
                    anchor_uus  = anchor_uus.viewer().name("anchor_uus"),
                    anchor_vvs  = anchor_vvs.viewer().name("anchor_vvs"),
                    anchor_uvs  = anchor_uvs.viewer().name("anchor_uvs"),
                    plastic_uus = plastic_uus.viewer().name("plastic_uus"),
                    plastic_vvs = plastic_vvs.viewer().name("plastic_vvs"),
                    plastic_uvs = plastic_uvs.viewer().name("plastic_uvs"),
                    rest_xs     = rest_xs.viewer().name("rest_x"),
                    rest_areas  = info.rest_areas().viewer().name("rest_area"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    energies = info.energies().viewer().name("energies"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    x_bars   = info.x_bars().viewer().name("x_bars"),
                    dt       = info.dt()] __device__(int I)
                   {
                       Float k11   = k11s(I);
                       Float k12   = k12s(I);
                       Float k22   = k22s(I);
                       Float k33_0 = k33_0s(I);
                       Float k33_1 = k33_1s(I);

                       //printf("k11: %f, k22: %f\r\n", k11, k22);

                       Float friction_k33 = friction_k33s(I);

                       Vector6 rest_x = rest_xs(I);

                       Vector3i idx = indices(I);

                       Vector9 curr_x;

                       curr_x(0) = xs(idx(0))(0);
                       curr_x(1) = xs(idx(0))(1);
                       curr_x(2) = xs(idx(0))(2);

                       curr_x(3) = xs(idx(1))(0);
                       curr_x(4) = xs(idx(1))(1);
                       curr_x(5) = xs(idx(1))(2);

                       curr_x(6) = xs(idx(2))(0);
                       curr_x(7) = xs(idx(2))(1);
                       curr_x(8) = xs(idx(2))(2);

                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float plastic_strain_uu = plastic_uus(I);
                       Float plastic_strain_vv = plastic_vvs(I);
                       Float plastic_strain_uv = plastic_uvs(I);

                       Float anchor_strain_uu = anchor_uus(I);
                       Float anchor_strain_vv = anchor_vvs(I);
                       Float anchor_strain_uv = anchor_uvs(I);

                       Float E;
                       STVKFHPS::StVKFrictionPlasticNonlinearShearEnergy(
                           E,
                           k11,
                           k12,
                           k22,
                           k33_0,
                           k33_1,
                           friction_k33,
                           anchor_strain_uu,
                           anchor_strain_vv,
                           anchor_strain_uv,
                           plastic_strain_uu,
                           plastic_strain_vv,
                           plastic_strain_uv,
                           curr_x,
                           rest_x);

                       energies(I) = E * rest_area * thickness * dt * dt;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace STVKFHPS = sym::stvk_nonlinear_friction_hardening_plastic_shear_2d;

        //std::cout << "Computer Hessian" << std::endl;
        const bool audit_raw = info.is_adjoint() && audit_linear_system
                               && audit_linear_system->diagnostic_raw_material_hessian();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [k11s   = k11s.cviewer().name("k11"),
                    k12s   = k12s.cviewer().name("k12"),
                    k22s   = k22s.cviewer().name("k22"),
                    k33_0s = k33_0s.cviewer().name("k33_0"),
                    k33_1s = k33_1s.cviewer().name("k33_1"),
                    friction_k33s = friction_k33s.cviewer().name("friction_k33"),
                    anchor_uus  = anchor_uus.viewer().name("anchor_uus"),
                    anchor_vvs  = anchor_vvs.viewer().name("anchor_vvs"),
                    anchor_uvs  = anchor_uvs.viewer().name("anchor_uvs"),
                    plastic_uus = plastic_uus.viewer().name("plastic_uus"),
                    plastic_vvs = plastic_vvs.viewer().name("plastic_vvs"),
                    plastic_uvs = plastic_uvs.viewer().name("plastic_uvs"),
                    rest_xs     = rest_xs.viewer().name("rest_x"),
                    indices     = info.indices().viewer().name("indices"),
                    xs          = info.xs().viewer().name("xs"),
                    x_bars      = info.x_bars().viewer().name("x_bars"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    G3s        = info.gradients().viewer().name("gradient"),
                    H3x3s      = info.hessians().viewer().name("hessian"),
                    rest_areas = info.rest_areas().viewer().name("volumes"),
                    is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                    dt         = info.dt(),
                    is_adjoint = info.is_adjoint(),
                    audit_raw = audit_raw] __device__(int I) mutable
                   {
                       Float k11   = k11s(I);
                       Float k12   = k12s(I);
                       Float k22   = k22s(I);
                       Float k33_0 = k33_0s(I);
                       Float k33_1 = k33_1s(I);

                       Float friction_k33 = friction_k33s(I);

                       Vector6 rest_x = rest_xs(I);

                       Vector3i idx = indices(I);

                       //printf("face index: %d, %d, %d", idx(0), idx(1), idx(2));

                       Vector9 curr_x;

                       curr_x(0) = xs(idx(0))(0);
                       curr_x(1) = xs(idx(0))(1);
                       curr_x(2) = xs(idx(0))(2);

                       curr_x(3) = xs(idx(1))(0);
                       curr_x(4) = xs(idx(1))(1);
                       curr_x(5) = xs(idx(1))(2);

                       curr_x(6) = xs(idx(2))(0);
                       curr_x(7) = xs(idx(2))(1);
                       curr_x(8) = xs(idx(2))(2);

                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float Vdt2 = rest_area * thickness * dt * dt;

                       Float plastic_strain_uu = plastic_uus(I);
                       Float plastic_strain_vv = plastic_vvs(I);
                       Float plastic_strain_uv = plastic_uvs(I);

                       Float anchor_strain_uu = anchor_uus(I);
                       Float anchor_strain_vv = anchor_vvs(I);
                       Float anchor_strain_uv = anchor_uvs(I);

                       Vector9 grad;
                       STVKFHPS::StVKFrictionPlasticNonlinearShearGradient(
                           grad,
                           k11,
                           k12,
                           k22,
                           k33_0,
                           k33_1,
                           friction_k33,
                           anchor_strain_uu,
                           anchor_strain_vv,
                           anchor_strain_uv,
                           plastic_strain_uu,
                           plastic_strain_vv,
                           plastic_strain_uv,
                           curr_x,
                           rest_x);

                       grad *= Vdt2;

                       Vector3i ignore = {
                           is_fixed(idx(0)), is_fixed(idx(1)), is_fixed(idx(2))};

                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<3>(I * 3).write(idx, ignore, grad);

                       Matrix9x9 H;
                       STVKFHPS::StVKFrictionPlasticNonlinearShearHessian(
                           H,
                           k11,
                           k12,
                           k22,
                           k33_0,
                           k33_1,
                           friction_k33,
                           anchor_strain_uu,
                           anchor_strain_vv,
                           anchor_strain_uv,
                           plastic_strain_uu,
                           plastic_strain_vv,
                           plastic_strain_uv,
                           curr_x,
                           rest_x);

                       H *= Vdt2;
                       // 2026-10-08: restore SPD projection in forward AND backward.
                       // Backward still uses PCG; raw indefinite Hessians are unsafe.
                       // This intentionally restores a projected approximate adjoint.
                       // Previous exact-Jacobian experiment:
                       // if(!is_adjoint)
                       //     make_spd(H);
                       // Old/default: make_spd(H). Bypass only for matrix export;
                       // no indefinite Hessian is passed to Newton/adjoint PCG.
                       if(!audit_raw)
                           make_spd(H);

                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.block<3, 3>(I * 3 * 3).write(idx, ignore, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(NonlinearFrictionHardeningPlasticStVKShell2D);

class DiffNonlinearFrictionHardeningPlasticStVKShell2D : public Codim2DConstitutionDiffParmReporter
{
  public:
    using Base = Codim2DConstitutionDiffParmReporter;
    using Base::Base;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    NonlinearFrictionHardeningPlasticStVKShell2D& constitution() noexcept
    {
        return static_cast<NonlinearFrictionHardeningPlasticStVKShell2D&>(Base::constitution());
    }

    vector<IndexT> h_k11_parm_indices;
    vector<IndexT> h_k22_parm_indices;
    vector<IndexT> h_k12_parm_indices;
    vector<IndexT> h_k33_0_parm_indices;
    vector<IndexT> h_k33_1_parm_indices;

    vector<IndexT> h_k11_tri_indices;
    vector<IndexT> h_k22_tri_indices;
    vector<IndexT> h_k12_tri_indices;
    vector<IndexT> h_k33_0_tri_indices;
    vector<IndexT> h_k33_1_tri_indices;

    vector<IndexT> h_friction_k33_parm_indices;
    vector<IndexT> h_friction_k33_tri_indices;
    vector<IndexT> h_thres_uv_parm_indices;
    vector<IndexT> h_thres_uv_tri_indices;

    vector<IndexT> h_yield_uv_ori_parm_indices;
    vector<IndexT> h_yield_uv_ori_tri_indices;
    vector<IndexT> h_hardening_k33_parm_indices;
    vector<IndexT> h_hardening_k33_tri_indices;

    muda::DeviceBuffer<IndexT> k11_parm_indices;
    muda::DeviceBuffer<IndexT> k22_parm_indices;
    muda::DeviceBuffer<IndexT> k12_parm_indices;
    muda::DeviceBuffer<IndexT> k33_0_parm_indices;
    muda::DeviceBuffer<IndexT> k33_1_parm_indices;

    muda::DeviceBuffer<IndexT> k11_tri_indices;
    muda::DeviceBuffer<IndexT> k22_tri_indices;
    muda::DeviceBuffer<IndexT> k12_tri_indices;
    muda::DeviceBuffer<IndexT> k33_0_tri_indices;
    muda::DeviceBuffer<IndexT> k33_1_tri_indices;

    muda::DeviceBuffer<IndexT> friction_k33_parm_indices;
    muda::DeviceBuffer<IndexT> friction_k33_tri_indices;
    muda::DeviceBuffer<IndexT> thres_uv_parm_indices;
    muda::DeviceBuffer<IndexT> thres_uv_tri_indices;

    muda::DeviceBuffer<IndexT> yield_uv_ori_parm_indices;
    muda::DeviceBuffer<IndexT> yield_uv_ori_tri_indices;
    muda::DeviceBuffer<IndexT> hardening_k33_parm_indices;
    muda::DeviceBuffer<IndexT> hardening_k33_tri_indices;

    SizeT k11_pGpP_triplet_count   = 0;
    SizeT k22_pGpP_triplet_count   = 0;
    SizeT k12_pGpP_triplet_count   = 0;
    SizeT k33_0_pGpP_triplet_count = 0;
    SizeT k33_1_pGpP_triplet_count = 0;

    SizeT friction_k33_pGpP_triplet_count = 0;
    SizeT thres_uv_pGpP_triplet_count     = 0;

    SizeT yield_uv_ori_pGpP_triplet_count  = 0;
    SizeT hardening_k33_pGpP_triplet_count = 0;

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        //std::cout << "In the Backward Initialization" << std::endl;

        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots  = world().scene().geometries();
        auto prim_count = info.primitive_count();

        // k11
        {
            h_k11_parm_indices.reserve(prim_count);
            h_k11_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k11 = sc.triangles().find<IndexT>("diff/k11");
                    return diff_k11 ? diff_k11->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k11)
                {
                    if(diff_k11 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k11_parm_indices.push_back(diff_k11);
                        // record the corresponding triangle index
                        h_k11_tri_indices.push_back(I.global_index());

                        //std::cout << "k11 index: " << diff_k11 << std::endl;
                    }
                });

            h_k11_parm_indices.shrink_to_fit();
            h_k11_tri_indices.shrink_to_fit();

            k11_parm_indices.resize(h_k11_parm_indices.size());
            k11_parm_indices.view().copy_from(h_k11_parm_indices.data());

            k11_tri_indices.resize(h_k11_tri_indices.size());
            k11_tri_indices.view().copy_from(h_k11_tri_indices.data());
        }

        std::cout << "K11 Parameter Size: " << k11_parm_indices.size() << std::endl;

        // k22
        {
            h_k22_parm_indices.reserve(prim_count);
            h_k22_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k22 = sc.triangles().find<IndexT>("diff/k22");
                    return diff_k22 ? diff_k22->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k22)
                {
                    if(diff_k22 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k22_parm_indices.push_back(diff_k22);
                        // record the corresponding triangle index
                        h_k22_tri_indices.push_back(I.global_index());

                        //std::cout << "k22 index: " << diff_k22 << std::endl;
                    }
                });

            h_k22_parm_indices.shrink_to_fit();
            h_k22_tri_indices.shrink_to_fit();

            k22_parm_indices.resize(h_k22_parm_indices.size());
            k22_parm_indices.view().copy_from(h_k22_parm_indices.data());

            k22_tri_indices.resize(h_k22_tri_indices.size());
            k22_tri_indices.view().copy_from(h_k22_tri_indices.data());
        }

        std::cout << "K22 Parameter Size: " << k22_parm_indices.size() << std::endl;

        // k12
        {
            h_k12_parm_indices.reserve(prim_count);
            h_k12_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k12 = sc.triangles().find<IndexT>("diff/k12");
                    return diff_k12 ? diff_k12->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k12)
                {
                    if(diff_k12 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k12_parm_indices.push_back(diff_k12);
                        // record the corresponding triangle index
                        h_k12_tri_indices.push_back(I.global_index());
                    }
                });

            h_k12_parm_indices.shrink_to_fit();
            h_k12_tri_indices.shrink_to_fit();

            k12_parm_indices.resize(h_k12_parm_indices.size());
            k12_parm_indices.view().copy_from(h_k12_parm_indices.data());

            k12_tri_indices.resize(h_k12_tri_indices.size());
            k12_tri_indices.view().copy_from(h_k12_tri_indices.data());
        }

        std::cout << "K12 Parameter Size: " << k12_parm_indices.size() << std::endl;

        // k33_0
        {
            h_k33_0_parm_indices.reserve(prim_count);
            h_k33_0_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k33_0 = sc.triangles().find<IndexT>("diff/k33_0");
                    return diff_k33_0 ? diff_k33_0->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k33_0)
                {
                    if(diff_k33_0 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k33_0_parm_indices.push_back(diff_k33_0);
                        // record the corresponding triangle index
                        h_k33_0_tri_indices.push_back(I.global_index());
                    }
                });

            h_k33_0_parm_indices.shrink_to_fit();
            h_k33_0_tri_indices.shrink_to_fit();

            k33_0_parm_indices.resize(h_k33_0_parm_indices.size());
            k33_0_parm_indices.view().copy_from(h_k33_0_parm_indices.data());

            k33_0_tri_indices.resize(h_k33_0_tri_indices.size());
            k33_0_tri_indices.view().copy_from(h_k33_0_tri_indices.data());
        }

        std::cout << "K33_0 Parameter Size: " << k33_0_parm_indices.size() << std::endl;

        // k33_1
        {
            h_k33_1_parm_indices.reserve(prim_count);
            h_k33_1_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k33_1 = sc.triangles().find<IndexT>("diff/k33_1");
                    return diff_k33_1 ? diff_k33_1->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k33_1)
                {
                    if(diff_k33_1 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k33_1_parm_indices.push_back(diff_k33_1);
                        // record the corresponding triangle index
                        h_k33_1_tri_indices.push_back(I.global_index());
                    }
                });

            h_k33_1_parm_indices.shrink_to_fit();
            h_k33_1_tri_indices.shrink_to_fit();

            k33_1_parm_indices.resize(h_k33_1_parm_indices.size());
            k33_1_parm_indices.view().copy_from(h_k33_1_parm_indices.data());

            k33_1_tri_indices.resize(h_k33_1_tri_indices.size());
            k33_1_tri_indices.view().copy_from(h_k33_1_tri_indices.data());
        }

        std::cout << "K33_1 Parameter Size: " << k33_1_parm_indices.size() << std::endl;

        // friction k33
        {
            h_friction_k33_parm_indices.reserve(prim_count);
            h_friction_k33_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_friction_k33 =
                        sc.triangles().find<IndexT>("diff/friction_k33");
                    return diff_friction_k33 ? diff_friction_k33->view() :
                                               span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_friction_k33)
                {
                    if(diff_friction_k33 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_friction_k33_parm_indices.push_back(diff_friction_k33);
                        // record the corresponding triangle index
                        h_friction_k33_tri_indices.push_back(I.global_index());
                    }
                });

            h_friction_k33_parm_indices.shrink_to_fit();
            h_friction_k33_tri_indices.shrink_to_fit();

            friction_k33_parm_indices.resize(h_friction_k33_parm_indices.size());
            friction_k33_parm_indices.view().copy_from(
                h_friction_k33_parm_indices.data());

            friction_k33_tri_indices.resize(h_friction_k33_tri_indices.size());
            friction_k33_tri_indices.view().copy_from(h_friction_k33_tri_indices.data());
        }

        std::cout << "Friction K33 Parameter Size: " << friction_k33_tri_indices.size()
                  << std::endl;

        // thres strain uv
        {
            h_thres_uv_parm_indices.reserve(prim_count);
            h_thres_uv_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_thres_uv = sc.triangles().find<IndexT>("diff/thres_uv");
                    return diff_thres_uv ? diff_thres_uv->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_thres_uv)
                {
                    if(diff_thres_uv >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_thres_uv_parm_indices.push_back(diff_thres_uv);
                        // record the corresponding triangle index
                        h_thres_uv_tri_indices.push_back(I.global_index());
                    }
                });

            h_thres_uv_parm_indices.shrink_to_fit();
            h_thres_uv_tri_indices.shrink_to_fit();

            thres_uv_parm_indices.resize(h_thres_uv_parm_indices.size());
            thres_uv_parm_indices.view().copy_from(h_thres_uv_parm_indices.data());

            thres_uv_tri_indices.resize(h_thres_uv_tri_indices.size());
            thres_uv_tri_indices.view().copy_from(h_thres_uv_tri_indices.data());
        }

        std::cout << "Thres Strain UV Parameter Size: " << thres_uv_tri_indices.size()
                  << std::endl;

        // yield uv ori
        {
            h_yield_uv_ori_parm_indices.reserve(prim_count);
            h_yield_uv_ori_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_yield_uv_ori =
                        sc.triangles().find<IndexT>("diff/yield_ori_uv");
                    return diff_yield_uv_ori ? diff_yield_uv_ori->view() :
                                               span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_yield_uv_ori)
                {
                    if(diff_yield_uv_ori >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_yield_uv_ori_parm_indices.push_back(diff_yield_uv_ori);
                        // record the corresponding triangle index
                        h_yield_uv_ori_tri_indices.push_back(I.global_index());
                    }
                });

            h_yield_uv_ori_parm_indices.shrink_to_fit();
            h_yield_uv_ori_tri_indices.shrink_to_fit();

            yield_uv_ori_parm_indices.resize(h_yield_uv_ori_parm_indices.size());
            yield_uv_ori_parm_indices.view().copy_from(
                h_yield_uv_ori_parm_indices.data());

            yield_uv_ori_tri_indices.resize(h_yield_uv_ori_tri_indices.size());
            yield_uv_ori_tri_indices.view().copy_from(h_yield_uv_ori_tri_indices.data());
        }

        std::cout << "Yield UV Ori Parameter Size: " << yield_uv_ori_tri_indices.size()
                  << std::endl;


        // hardening k33
        {
            h_hardening_k33_parm_indices.reserve(prim_count);
            h_hardening_k33_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_hardening_k33 =
                        sc.triangles().find<IndexT>("diff/hardening_k33");
                    return diff_hardening_k33 ? diff_hardening_k33->view() :
                                                span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_hardening_k33)
                {
                    if(diff_hardening_k33 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_hardening_k33_parm_indices.push_back(diff_hardening_k33);
                        // record the corresponding triangle index
                        h_hardening_k33_tri_indices.push_back(I.global_index());
                    }
                });

            h_hardening_k33_parm_indices.shrink_to_fit();
            h_hardening_k33_tri_indices.shrink_to_fit();

            hardening_k33_parm_indices.resize(h_hardening_k33_parm_indices.size());
            hardening_k33_parm_indices.view().copy_from(
                h_hardening_k33_parm_indices.data());

            hardening_k33_tri_indices.resize(h_hardening_k33_tri_indices.size());
            hardening_k33_tri_indices.view().copy_from(
                h_hardening_k33_tri_indices.data());
        }

        std::cout << "Hardening k33 Parameter Size: "
                  << hardening_k33_tri_indices.size() << std::endl;

        //std::cout << "Finish the Backward Initialization" << std::endl;
    }

    virtual void do_update(DiffParmUpdateInfo& info) override
    {
        //std::cout << "In the Backward Update" << std::endl;

        using namespace muda;

        // update k11
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k11_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k11_parm_indices = k11_parm_indices.cviewer().name("k11_parm_indices"),
                        k11_tri_indices = k11_tri_indices.cviewer().name("k11_tri_indices"),
                        k11s = constitution().k11s.viewer().name("k11")] __device__(int I)
                       {
                           // Get the data from parameters and update the k11
                           auto& k11 = k11s(k11_tri_indices(I));
                           k11       = parameters(k11_parm_indices(I));

                           //printf("Update k11 index: %d\r\n", k11_parm_indices(I));
                           //printf("Update k11 selected parameters: %f\r\n",
                           //       parameters(k11_parm_indices(I)));
                       });
        }

        // update k22
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k22_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k22_parm_indices = k22_parm_indices.cviewer().name("k22_parm_indices"),
                        k22_tri_indices = k22_tri_indices.cviewer().name("k22_tri_indices"),
                        k22s = constitution().k22s.viewer().name("k22")] __device__(int I)
                       {
                           // Get the data from parameters and update the k22
                           auto& k22 = k22s(k22_tri_indices(I));
                           k22       = parameters(k22_parm_indices(I));


                           //printf("Update k22 index: %d\r\n", k22_parm_indices(I));
                           //printf("Update k22 selected parameters: %f\r\n",
                           //       parameters(k22_parm_indices(I)));
                       });
        }

        // update k12
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k12_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k12_parm_indices = k12_parm_indices.cviewer().name("k12_parm_indices"),
                        k12_tri_indices = k12_tri_indices.cviewer().name("k12_tri_indices"),
                        k12s = constitution().k12s.viewer().name("k12")] __device__(int I)
                       {
                           // Get the data from parameters and update the k12
                           auto& k12 = k12s(k12_tri_indices(I));
                           k12       = parameters(k12_parm_indices(I));
                       });
        }

        // update k33_0
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k33_0_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k33_0_parm_indices = k33_0_parm_indices.cviewer().name("k33_0_parm_indices"),
                        k33_0_tri_indices = k33_0_tri_indices.cviewer().name("k33_0_tri_indices"),
                        k33_0s = constitution().k33_0s.viewer().name("k33_0")] __device__(int I)
                       {
                           // Get the data from parameters and update the k33_0
                           auto& k33_0 = k33_0s(k33_0_tri_indices(I));
                           k33_0       = parameters(k33_0_parm_indices(I));
                       });
        }

        // update k33_1
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k33_1_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k33_1_parm_indices = k33_1_parm_indices.cviewer().name("k33_1_parm_indices"),
                        k33_1_tri_indices = k33_1_tri_indices.cviewer().name("k33_1_tri_indices"),
                        k33_1s = constitution().k33_1s.viewer().name("k33_1")] __device__(int I)
                       {
                           // Get the data from parameters and update the k33_1
                           auto& k33_1 = k33_1s(k33_1_tri_indices(I));
                           k33_1       = parameters(k33_1_parm_indices(I));
                       });
        }

        // update friction k33
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(friction_k33_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        friction_k33_parm_indices =
                            friction_k33_parm_indices.cviewer().name("friction_k33_parm_indices"),
                        friction_k33_tri_indices =
                            friction_k33_tri_indices.cviewer().name("friction_k33_tri_indices"),
                        friction_k33s = constitution().friction_k33s.viewer().name(
                            "friction_k33")] __device__(int I)
                       {
                           // Get the data from parameters and update the friction k33
                           auto& friction_k33 =
                               friction_k33s(friction_k33_tri_indices(I));
                           friction_k33 = parameters(friction_k33_parm_indices(I));
                       });
        }

        // update Thres UV
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(thres_uv_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        thres_uv_parm_indices = thres_uv_parm_indices.cviewer().name("thres_uv_parm_indices"),
                        thres_uv_tri_indices = thres_uv_tri_indices.cviewer().name("thres_uv_tri_indices"),
                        thres_uvs = constitution().thres_uvs.viewer().name(
                            "friction_k33")] __device__(int I)
                       {
                           // Get the data from parameters and update the thres strain uv
                           auto& thres_uv = thres_uvs(thres_uv_tri_indices(I));
                           thres_uv = parameters(thres_uv_parm_indices(I));
                       });
        }

        // update Yield Ori
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(yield_uv_ori_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        yield_uv_ori_parm_indices =
                            yield_uv_ori_parm_indices.cviewer().name("yield_uv_ori_parm_indices"),
                        yield_uv_ori_tri_indices =
                            yield_uv_ori_tri_indices.cviewer().name("yield_uv_ori_tri_indices"),
                        yield_ori_uvs = constitution().yield_ori_uvs.viewer().name(
                            "yield_ori_uvs")] __device__(int I)
                       {
                           // Get the data from parameters and update the thres strain uv
                           auto& yield_uv_ori =
                               yield_ori_uvs(yield_uv_ori_tri_indices(I));
                           yield_uv_ori = parameters(yield_uv_ori_parm_indices(I));

                           //printf("Yield Ori Update:%f \n\r", yield_uv_ori);
                       });
        }

        // update Hardening k33
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(hardening_k33_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        hardening_k33_parm_indices =
                            hardening_k33_parm_indices.cviewer().name("hardening_k33_parm_indices"),
                        hardening_k33_tri_indices =
                            hardening_k33_tri_indices.cviewer().name("hardening_k33_tri_indices"),
                        hardening_k33s = constitution().hardening_k33s.viewer().name(
                            "hardening_k33s")] __device__(int I)
                       {
                           // Get the data from parameters and update the thres strain uv
                           auto& hardening_k33 =
                               hardening_k33s(hardening_k33_tri_indices(I));
                           hardening_k33 = parameters(hardening_k33_parm_indices(I));
                       });
        }

        //std::cout << "Finish the Backward Update" << std::endl;
    }

    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        //std::cout << "In the Backward Report extend" << std::endl;

        // every k11 related to 3 vertices, pGpk11 is 3 * matrix3x1
        {
            auto stencil_dim       = 3;
            auto count             = k11_parm_indices.size();
            k11_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k11_pGpP_triplet_count " << k11_pGpP_triplet_count << std::endl;
        }

        //std::cout << "k11 pGpP Size: " << k11_pGpP_triplet_count << std::endl;

        // every k22 related to 3 vertices, pGpk22 is 3 * matrix3x1
        {
            auto stencil_dim       = 3;
            auto count             = k22_parm_indices.size();
            k22_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k22_pGpP_triplet_count " << k22_pGpP_triplet_count << std::endl;
        }

        // every k12 related to 3 vertices, pGpk12 is 3 * matrix3x1
        {
            auto stencil_dim       = 3;
            auto count             = k12_parm_indices.size();
            k12_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k12_pGpP_triplet_count " << k12_pGpP_triplet_count << std::endl;
        }

        // every k33_0 related to 3 vertices, pGpk33 is 3 * matrix3x1
        {
            auto stencil_dim         = 3;
            auto count               = k33_0_parm_indices.size();
            k33_0_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k33_pGpP_triplet_count " << k33_pGpP_triplet_count << std::endl;
        }

        // every k33_1 related to 3 vertices, pGpk33 is 3 * matrix3x1
        {
            auto stencil_dim         = 3;
            auto count               = k33_1_parm_indices.size();
            k33_1_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k33_pGpP_triplet_count " << k33_pGpP_triplet_count << std::endl;
        }

        // every friction k33 related to 3 vertices, pGpk11 is 3 * matrix3x1
        {
            auto stencil_dim                = 3;
            auto count                      = friction_k33_parm_indices.size();
            friction_k33_pGpP_triplet_count = count * stencil_dim;
        }

        // every thres_uv related to 3 vertices, pGpk11 is 3 * matrix3x1
        {
            auto stencil_dim            = 3;
            auto count                  = thres_uv_parm_indices.size();
            thres_uv_pGpP_triplet_count = count * stencil_dim;
        }

        // every yield_ori related to 3 vertices, pGpk11 is 3 * matrix3x1
        {
            auto stencil_dim                = 3;
            auto count                      = yield_uv_ori_parm_indices.size();
            yield_uv_ori_pGpP_triplet_count = count * stencil_dim;
        }

        // every hardening k33 related to 3 vertices, pGpk11 is 3 * matrix3x1
        {
            auto stencil_dim = 3;
            auto count       = hardening_k33_parm_indices.size();
            hardening_k33_pGpP_triplet_count = count * stencil_dim;
        }

        info.triplet_count(k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count
                           + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count + friction_k33_pGpP_triplet_count
                           + thres_uv_pGpP_triplet_count + yield_uv_ori_pGpP_triplet_count
                           + hardening_k33_pGpP_triplet_count);

        //std::cout << "Finish the Backward Report extend" << std::endl;
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        //std::cout << "In Do Assemble" << std::endl;

        using namespace muda;
        namespace svtkfhps = sym::stvk_nonlinear_friction_hardening_plastic_shear_2d;

        auto pGpP = info.pGpP();

        // assemble k11
        {
            auto  N          = k11_parm_indices.size();
            SizeT k11_offset = 0;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k11_parm_indices = k11_parm_indices.cviewer().name("k11_parm_indices"),
                     k11_tri_indices = k11_tri_indices.cviewer().name("k11_tri_indices"),
                     pGpP = pGpP.viewer().name("pGpP"),
                     plastic_uus = constitution().plastic_uus.viewer().name("plastic_uus"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     k11_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k11_tri_indices(I);
                        auto parm_idx = k11_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Vector9 fuu = Du * F.col(0);

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Float plastic_strain_uu = plastic_uus(tri_idx);

                        Float elastic_strain_uu;
                        elastic_strain_uu = G(0, 0) - plastic_strain_uu;

                        Vector9 pGpk11;

                        pGpk11 = elastic_strain_uu * fuu * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk11, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k11_offset + I * 3 + k).write(vI, J, pGpk11.segment<3>(k * 3));
                        }
                    });
        }

        // assemble k22
        {
            auto  N          = k22_parm_indices.size();
            SizeT k22_offset = k11_pGpP_triplet_count;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k22_parm_indices = k22_parm_indices.cviewer().name("k22_parm_indices"),
                     k22_tri_indices = k22_tri_indices.cviewer().name("k22_tri_indices"),
                     pGpP = pGpP.viewer().name("pGpP"),
                     plastic_vvs = constitution().plastic_vvs.viewer().name("plastic_vvs"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     k22_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k22_tri_indices(I);
                        auto parm_idx = k22_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Eigen::Matrix<double, 9, 3> Dv;
                        Dv.setZero();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        Vector9 fvv = Dv * F.col(1);

                        Float elastic_strain_vv;
                        elastic_strain_vv = G(1, 1) - plastic_vvs(tri_idx);

                        Vector9 pGpk22;

                        pGpk22 = elastic_strain_vv * fvv * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk22, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k22_offset + I * 3 + k).write(vI, J, pGpk22.segment<3>(k * 3));
                        }
                    });
        }

        // assemble k12
        {
            auto  N          = k12_parm_indices.size();
            SizeT k12_offset = k11_pGpP_triplet_count + k22_pGpP_triplet_count;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k12_parm_indices = k12_parm_indices.cviewer().name("k12_parm_indices"),
                     k12_tri_indices = k12_tri_indices.cviewer().name("k12_tri_indices"),
                     pGpP = pGpP.viewer().name("pGpP"),
                     plastic_uus = constitution().plastic_uus.viewer().name("plastic_uus"),
                     plastic_vvs = constitution().plastic_vvs.viewer().name("plastic_vvs"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     k12_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k12_tri_indices(I);
                        auto parm_idx = k12_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(I);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        Vector9 fuu = Du * F.col(0);
                        Vector9 fvv = Dv * F.col(1);

                        Float plastic_strain_uu = plastic_uus(tri_idx);
                        Float plastic_strain_vv = plastic_vvs(tri_idx);

                        Float elastic_strain_uu, elastic_strain_vv;
                        elastic_strain_uu = G(0, 0) - plastic_strain_uu;
                        elastic_strain_vv = G(1, 1) - plastic_strain_vv;

                        Vector9 pGpk12;

                        pGpk12 = (elastic_strain_uu * fvv + elastic_strain_vv * fuu) * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk12, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k12_offset + I * 3 + k).write(vI, J, pGpk12.segment<3>(k * 3));
                        }
                    });
        }

        // assemble k33_0
        {
            auto  N            = k33_0_parm_indices.size();
            SizeT k33_0_offset = k11_pGpP_triplet_count + k22_pGpP_triplet_count
                                 + k12_pGpP_triplet_count;

            //std::cout << "Assemable k33" << std::endl;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k33_0_parm_indices = k33_0_parm_indices.cviewer().name("k33_0_parm_indices"),
                     k33_0_tri_indices = k33_0_tri_indices.cviewer().name("k33_0_tri_indices"),
                     pGpP = pGpP.viewer().name("pGpP"),
                     plastic_uvs = info.plastic_uvs().viewer().name("plastic_uvs"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     d_plastic_uv_d_k33 = info.d_plastic_uv_d_k33().viewer().name("d_plastic_uv_d_k33"),
                     dt = info.dt(),
                     k33_0_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k33_0_tri_indices(I);
                        auto parm_idx = k33_0_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        // Guv=.5*Fu.dot(Fv); the old expression was twice its derivative.
                        Vector9 fuv = 0.5 * (Du * F.col(1) + Dv * F.col(0));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Float elastic_strain_uv;
                        elastic_strain_uv = G(0, 1) - plastic_uvs(tri_idx);

                        Vector9 pGpk33_0;
                        pGpk33_0 = elastic_strain_uv * fuv * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk33_0, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k33_0_offset + I * 3 + k).write(vI, J, pGpk33_0.segment<3>(k * 3));
                        }
                    });
        }


        // assemble k33_1
        {
            auto  N            = k33_1_parm_indices.size();
            SizeT k33_1_offset = k11_pGpP_triplet_count + k22_pGpP_triplet_count
                                 + k12_pGpP_triplet_count + k33_0_pGpP_triplet_count;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k33_1_parm_indices = k33_1_parm_indices.cviewer().name("k33_1_parm_indices"),
                     k33_1_tri_indices = k33_1_tri_indices.cviewer().name("k33_1_tri_indices"),
                     pGpP = pGpP.viewer().name("pGpP"),
                     plastic_uvs = info.plastic_uvs().viewer().name("plastic_uvs"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     d_plastic_uv_d_k33 = info.d_plastic_uv_d_k33().viewer().name("d_plastic_uv_d_k33"),
                     dt = info.dt(),
                     k33_1_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k33_1_tri_indices(I);
                        auto parm_idx = k33_1_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        // Guv=.5*Fu.dot(Fv); the old expression was twice its derivative.
                        Vector9 fuv = 0.5 * (Du * F.col(1) + Dv * F.col(0));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Float elastic_strain_uv;
                        elastic_strain_uv = G(0, 1) - plastic_uvs(tri_idx);

                        Vector9 pGpk33_1;
                        // d/dk33_1 of dW/dGuv, for W=.5*(k0+k1*abs(q))*z^2.
                        pGpk33_1 = (abs(G(0, 1)) * elastic_strain_uv
                                    + 0.5 * sym::stvk_nonlinear_friction_hardening_plastic_shear_2d::sign<Float>(G(0, 1)) * elastic_strain_uv * elastic_strain_uv)
                                   * fuv * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk33_1, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k33_1_offset + I * 3 + k).write(vI, J, pGpk33_1.segment<3>(k * 3));
                        }
                    });
        }


        // assemble friction k33
        {
            auto  N = friction_k33_parm_indices.size();
            SizeT friction_k33_offset =
                k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count
                + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [friction_k33_parm_indices =
                         friction_k33_parm_indices.cviewer().name("friction_k33_parm_indices"),
                     friction_k33_tri_indices =
                         friction_k33_tri_indices.cviewer().name("friction_k33_tri_indices"),
                     pGpP       = pGpP.viewer().name("pGpP"),
                     anchor_uvs = info.anchor_uvs().viewer().name("anchor_uvs"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     friction_k33_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = friction_k33_tri_indices(I);
                        auto parm_idx = friction_k33_parm_indices(I);

                        Vector6  rest_x = rest_xs(tri_idx);
                        Vector3i idx    = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));
                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        // Guv=.5*Fu.dot(Fv); the old expression was twice its derivative.
                        Vector9 fuv = 0.5 * (Du * F.col(1) + Dv * F.col(0));

                        Float friction_var_strain_uv;
                        friction_var_strain_uv = G(0, 1) - anchor_uvs(tri_idx);

                        Vector9 pGpFrick33;

                        pGpFrick33 = friction_var_strain_uv * fuv * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpFrick33, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(friction_k33_offset + I * 3 + k)
                                .write(vI, J, pGpFrick33.segment<3>(k * 3));
                        }
                    });
        }

        // assamble thres uv
        {
            auto  N = thres_uv_parm_indices.size();
            SizeT thres_uv_offset =
                k11_pGpP_triplet_count + k22_pGpP_triplet_count
                + k12_pGpP_triplet_count + k33_0_pGpP_triplet_count
                + k33_1_pGpP_triplet_count + friction_k33_pGpP_triplet_count;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [thres_uv_parm_indices = thres_uv_parm_indices.cviewer().name("thres_uv_parm_indices"),
                     thres_uv_tri_indices = thres_uv_tri_indices.cviewer().name("thres_uv_tri_indices"),
                     pGpP = pGpP.viewer().name("pGpP"),
                     friction_k33s = constitution().friction_k33s.cviewer().name("friction_k33"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     d_anchor_uv_d_thres_uvs =
                         info.d_anchor_uv_d_thres_uv().viewer().name("d_anchor_uv_d_thres_uv"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     thres_uv_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = thres_uv_tri_indices(I);
                        auto parm_idx = thres_uv_parm_indices(I);

                        Float friction_k33 = friction_k33s(tri_idx);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;

                        //printf("F: {%f}", F(0, 0));
                        //printf("F: {%f}", F(0, 1));
                        //printf("F: {%f}", F(1, 0));
                        //printf("F: {%f}", F(1, 1));

                        //auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        // Guv=.5*Fu.dot(Fv); the old expression was twice its derivative.
                        Vector9 fuv = 0.5 * (Du * F.col(1) + Dv * F.col(0));

                        Float d_anchor_uv_d_thres_uv = d_anchor_uv_d_thres_uvs(tri_idx);

                        //d_anchor_uv_d_thres_uv = 1.0;

                        //printf("In Assemble: d_anchor_d_thres: {%f}", d_anchor_uv_d_thres_uv);

                        Vector9 pGpk_thres_uv;

                        //fuv = 1000000 * Vector9::Ones(9);

                        pGpk_thres_uv = -friction_k33 * d_anchor_uv_d_thres_uv * fuv * Vdt2;

                        //printf("In Assemble: friction_k33: {%f}", friction_k33);

                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(0));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(1));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(2));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(3));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(4));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(5));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(6));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(7));
                        //printf("In Assemble: fuv: {%f}\r\n", pGpk_thres_uv(8));

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk_thres_uv, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(thres_uv_offset + I * 3 + k)
                                .write(vI, J, pGpk_thres_uv.segment<3>(k * 3));
                        }
                    });
        }


        // assemble yield uv ori
        {
            auto  N = yield_uv_ori_parm_indices.size();
            SizeT yield_uv_offset =
                k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count
                + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count
                + friction_k33_pGpP_triplet_count + thres_uv_pGpP_triplet_count;

            //std::cout << "Assemable k33" << std::endl;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [yield_uv_ori_parm_indices =
                         yield_uv_ori_parm_indices.cviewer().name("yield_uv_ori_parm_indices"),
                     yield_uv_ori_tri_indices =
                         yield_uv_ori_tri_indices.cviewer().name("yield_uv_ori_tri_indices"),
                     pGpP   = pGpP.viewer().name("pGpP"),
                     k33_0s = constitution().k33_0s.cviewer().name("k33_0"),
                     k33_1s = constitution().k33_1s.cviewer().name("k33_1"),
                     plastic_uvs = info.plastic_uvs().viewer().name("plastic_uvs"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     d_plastic_uv_d_yield_ori =
                         info.d_plastic_uv_d_yield_ori().viewer().name("d_plastic_uv_d_yield_ori"),
                     dt = info.dt(),
                     yield_uv_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = yield_uv_ori_tri_indices(I);
                        auto parm_idx = yield_uv_ori_parm_indices(I);

                        Float k33_0 = k33_0s(tri_idx);
                        Float k33_1 = k33_1s(tri_idx);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float strain_uv = G(0, 1);

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        // Guv=.5*Fu.dot(Fv); the old expression was twice its derivative.
                        Vector9 fuv = 0.5 * (Du * F.col(1) + Dv * F.col(0));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Vector9 pGpYield;
                        pGpYield = -(k33_0 + k33_1 * abs(strain_uv)
                                      + k33_1 * sym::stvk_nonlinear_friction_hardening_plastic_shear_2d::sign<Float>(strain_uv) * (strain_uv - plastic_uvs(tri_idx)))
                                   * d_plastic_uv_d_yield_ori(tri_idx) * fuv * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpYield, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(yield_uv_offset + I * 3 + k).write(vI, J, pGpYield.segment<3>(k * 3));
                        }
                    });
        }


        // assemble hardening k33
        {
            auto N = hardening_k33_tri_indices.size();
            SizeT hardening_k33_offset =
                k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count
                + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count
                + friction_k33_pGpP_triplet_count + thres_uv_pGpP_triplet_count+yield_uv_ori_pGpP_triplet_count;

            //std::cout << "Assemable k33" << std::endl;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [hardening_k33_parm_indices =
                         hardening_k33_parm_indices.cviewer().name("hardening_k33_parm_indices"),
                     hardening_k33_tri_indices =
                         hardening_k33_tri_indices.cviewer().name("hardening_k33_tri_indices"),
                     pGpP   = pGpP.viewer().name("pGpP"),
                     k33_0s = constitution().k33_0s.cviewer().name("k33_0"),
                     k33_1s = constitution().k33_1s.cviewer().name("k33_1"),
                     plastic_uvs = info.plastic_uvs().viewer().name("plastic_uvs"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("volumes"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     d_plastic_uv_d_hardening_k33 =
                         info.d_plastic_uv_d_hardening_k33().viewer().name("d_plastic_uv_d_hardening_k33"),
                     dt = info.dt(),
                     hardening_k33_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = hardening_k33_tri_indices(I);
                        auto parm_idx = hardening_k33_parm_indices(I);

                        Float k33_0 = k33_0s(tri_idx);
                        Float k33_1 = k33_1s(tri_idx);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float strain_uv = G(0, 1);

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        // Guv=.5*Fu.dot(Fv); the old expression was twice its derivative.
                        Vector9 fuv = 0.5 * (Du * F.col(1) + Dv * F.col(0));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Vector9 pGpk33;
                        pGpk33 = -(k33_0 + k33_1 * abs(strain_uv)
                                      + k33_1 * sym::stvk_nonlinear_friction_hardening_plastic_shear_2d::sign<Float>(strain_uv) * (strain_uv - plastic_uvs(tri_idx)))
                                 * d_plastic_uv_d_hardening_k33(tri_idx) * fuv * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk33, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(hardening_k33_offset + I * 3 + k)
                                .write(vI, J, pGpk33.segment<3>(k * 3));
                        }
                    });
        }
        //std::cout << "Finish Do Assemble" << std::endl;
    }
};

REGISTER_SIM_SYSTEM(DiffNonlinearFrictionHardeningPlasticStVKShell2D);
}  // namespace uipc::backend::cuda


