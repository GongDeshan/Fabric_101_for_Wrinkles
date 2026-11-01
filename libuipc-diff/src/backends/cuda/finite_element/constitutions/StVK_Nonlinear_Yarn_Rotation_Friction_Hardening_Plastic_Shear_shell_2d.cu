#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/stvk_yarn_rotation_math.h>
#include <diff_sim/adjoint_method_replayer.h>
#include <finite_element/finite_element_diff_dof_reporter.h>
#include <finite_element/constitutions/stvk_yarn_rotation_history.h>
#include <sim_engine.h>
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
static constexpr U64 ConstitutionUID = 1068;

class YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D final : public Codim2DConstitution
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

    muda::DeviceBuffer<I32> rotation_enabled;
    muda::DeviceBuffer<Vector9> plastic_parameter_jacobians,anchor_parameter_jacobians;
    muda::DeviceBuffer<Vector9> hardening_parameter_jacobians;
    using Matrix3x9History=Eigen::Matrix<Float,3,9>;
    struct HistoryFrame {vector<Matrix3x3> state;vector<Matrix3x9History> source;};
    vector<HistoryFrame> position_history;
    muda::DeviceBuffer<Matrix3x3> state_transition;
    muda::DeviceBuffer<Matrix3x9History> position_transition;
    muda::DeviceBuffer<Float> history_measures;
    GlobalDiffSimManager* history_manager=nullptr;
    bool history_position_gradient_enabled=true;
    void record_position_history()
    {
        const auto frame=engine().frame();
        if(frame==1)position_history.clear();
        if(!history_position_gradient_enabled || !history_manager || !history_manager->is_grad_enabled())return;
        UIPC_ASSERT(position_history.size()+1==frame,"Yarn shear history frame mismatch");
        HistoryFrame snapshot;const auto N=tri_idxs.size();
        snapshot.state.resize(N);snapshot.source.resize(N);
        state_transition.view().copy_to(snapshot.state.data());
        position_transition.view().copy_to(snapshot.source.data());
        position_history.push_back(std::move(snapshot));
    }
    void anchor_plastic_strain_update()
    {
        using namespace muda;
        const auto N=tri_idxs.size();
        if(!N) return;

        yield_ori_uvs.view().copy_to(h_yield_uvs.data());
        SizeT offset=0;
        for(auto slot:filter_geo_slot_indices)
        {
            auto geo=geo_slots[slot]->geometry().as<geometry::SimplicialComplex>();
            const auto count=geo->triangles().size();
            auto reset=geo->triangles().find<I32>("reset_anchor_plastic")->view();
            if(count && reset[0])
            {
                auto a=geo->triangles().find<Float>("anchor_strain_uv")->view();
                auto p=geo->triangles().find<Float>("plastic_strain_uv")->view();
                auto h=geo->triangles().find<Float>("hardening_plastic_uv")->view();
                for(SizeT j=0;j<count;++j)
                {
                    h_anchor_uvs[offset+j]=a[j];h_plastic_uvs[offset+j]=p[j];
                    h_hardening_plastic_uvs[offset+j]=h[j];
                }
                anchor_uvs.view(offset,count).copy_from(h_anchor_uvs.data()+offset);
                plastic_uvs.view(offset,count).copy_from(h_plastic_uvs.data()+offset);
                hardening_plastic_uvs.view(offset,count).copy_from(h_hardening_plastic_uvs.data()+offset);
                yield_uvs.view(offset,count).copy_from(h_yield_uvs.data()+offset);

                muda::ParallelFor().file_line(__FILE__,__LINE__).apply(count,
                    [dp=plastic_parameter_jacobians.viewer(),dh=hardening_parameter_jacobians.viewer(),
                     da=anchor_parameter_jacobians.viewer(),offset] __device__(int j) mutable
                    {dp(offset+j).setZero();dh(offset+j).setZero();da(offset+j).setZero();});
            }
            offset+=count;
        }
        ParallelFor().file_line(__FILE__,__LINE__).apply(N,
            [idx=tri_idxs.cviewer(),X=rest_xs.cviewer(),xs=fem->xs().viewer(),
             su=strain_uus.viewer(),sv=strain_vvs.viewer(),sq=strain_uvs.viewer(),
             a=anchor_uvs.viewer(),p=plastic_uvs.viewer(),h=hardening_plastic_uvs.viewer(),y=yield_uvs.viewer(),
             y0=yield_ori_uvs.cviewer(),threshold=thres_uvs.cviewer(),kh=hardening_k33s.cviewer(),
             k22=k22s.cviewer(),k0=k33_0s.cviewer(),k1=k33_1s.cviewer(),enabled=rotation_enabled.cviewer(),
             dp=plastic_parameter_jacobians.viewer(),da=anchor_parameter_jacobians.viewer(),
             dh=hardening_parameter_jacobians.viewer(),A=state_transition.viewer(),B=position_transition.viewer(),
             fa=fem->anchor_strain_uvs.viewer(),fp=fem->plastic_strain_uvs.viewer(),
             fh=fem->hardening_plastic_strain_uvs.viewer(),fy=fem->yield_strain_uvs.viewer(),
             fdy=fem->d_plastic_uv_d_yield_ori.viewer(),fdk=fem->d_plastic_uv_d_k33.viewer(),
             fdh=fem->d_plastic_uv_d_hardening_k33.viewer(),fda=fem->d_anchor_uv_d_thres_uv.viewer()]
            __device__(int I) mutable
            {
                Vector9 x;
                const auto ids=idx(I);
                for(int j=0;j<3;++j) x.segment<3>(j*3)=xs(ids(j));
                auto g=yarn_rotation::geometry(x,X(I));
                const Float q=g.strain(2);
                su(I)=g.strain(0);sv(I)=g.strain(1);sq(I)=q;
                const bool rotate=enabled(I)!=0;
                const auto transition=yarn_rotation::history_transition(g.strain,p(I),h(I),a(I),
                    k22(I),k0(I),k1(I),kh(I),y0(I),threshold(I),rotate);
                A(I)=transition.state;B(I)=transition.strain*g.jacobian;
                const Vector9 previous_dp=dp(I),previous_dh=dh(I),previous_da=da(I);
                const Float effective=yarn_rotation::effective_shear(g.strain(0),g.strain(1),q,k22(I),k0(I),k1(I),rotate);
                const Float k=effective>1e-12?effective:1e-12;
                const Float hard=kh(I),old_h=h(I),elastic=q-p(I);
                const Float sign=elastic>0?1.0:(elastic<0?-1.0:0.0);
                const Float yield=y0(I)+old_h*hard/k;
                const Float excess=std::abs(elastic)-yield;
                Vector9 dplastic=Vector9::Zero(),danchor=Vector9::Zero();
                fdy(I)=0;fdk(I)=0;fdh(I)=0;fda(I)=0;
                if(excess>0)
                {
                    const Float ratio=k/(k+hard),increment=excess*ratio;
                    p(I)+=sign*increment;h(I)+=increment;
                    fdy(I)=-sign*ratio;
                    fdk(I)=sign*(old_h*hard/(k*k)*ratio+excess*hard/((k+hard)*(k+hard)));
                    fdh(I)=sign*(-old_h/k*ratio-excess*k/((k+hard)*(k+hard)));
                    const Float t=rotate?yarn_rotation::cosine_squared(
                        yarn_rotation::Jet<Float>(g.strain(0)),yarn_rotation::Jet<Float>(g.strain(1)),
                        yarn_rotation::Jet<Float>(q)).value:0.0;
                    const Float u=1.0-t;
                    if(effective>1e-12)
                    {
                        dplastic(1)=fdk(I)*t*u;
                        dplastic(3)=fdk(I)*u;
                        dplastic(4)=fdk(I)*std::abs(q)*u;
                    }
                    dplastic(7)=fdy(I);dplastic(8)=fdh(I);
                }
                y(I)=y0(I)+h(I)*hard/k;
                const Float distance=q-a(I);
                if(std::abs(distance)>threshold(I))
                {
                    const Float slip_sign=distance>0?1.0:-1.0;
                    a(I)=q-slip_sign*threshold(I);
                    danchor(6)=-slip_sign;fda(I)=danchor(6);
                }


                const Vector9 direct_h=sign*dplastic;
                dp(I)=transition.state(0,0)*previous_dp+transition.state(0,1)*previous_dh+dplastic;
                dh(I)=transition.state(1,0)*previous_dp+transition.state(1,1)*previous_dh+direct_h;
                da(I)=transition.state(2,2)*previous_da+danchor;
                fa(I)=a(I);fp(I)=p(I);fh(I)=h(I);fy(I)=y(I);
            });
        strain_uus.view().copy_to(h_strain_uus.data());strain_vvs.view().copy_to(h_strain_vvs.data());
        strain_uvs.view().copy_to(h_strain_uvs.data());anchor_uvs.view().copy_to(h_anchor_uvs.data());
        plastic_uvs.view().copy_to(h_plastic_uvs.data());hardening_plastic_uvs.view().copy_to(h_hardening_plastic_uvs.data());
        yield_uvs.view().copy_to(h_yield_uvs.data());
        offset=0;
        for(auto slot:filter_geo_slot_indices)
        {
            auto geo=geo_slots[slot]->geometry().as<geometry::SimplicialComplex>();
            auto count=geo->triangles().size();
            auto su=view(*geo->triangles().find<Float>("strain_uu"));
            auto sv=view(*geo->triangles().find<Float>("strain_vv"));
            auto sq=view(*geo->triangles().find<Float>("strain_uv"));
            auto a=view(*geo->triangles().find<Float>("anchor_strain_uv"));
            auto p=view(*geo->triangles().find<Float>("plastic_strain_uv"));
            auto y=view(*geo->triangles().find<Float>("yield_uv"));
            auto h=view(*geo->triangles().find<Float>("hardening_plastic_uv"));
            for(SizeT j=0;j<count;++j)
            {
                su[j]=h_strain_uus[offset+j];sv[j]=h_strain_vvs[offset+j];sq[j]=h_strain_uvs[offset+j];
                a[j]=h_anchor_uvs[offset+j];p[j]=h_plastic_uvs[offset+j];
                y[j]=h_yield_uvs[offset+j];h[j]=h_hardening_plastic_uvs[offset+j];
            }
            offset+=count;
        }
        record_position_history();
    }

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override
    {
        //std::cout << "Stvk build" << std::endl;

        fem = &require<FiniteElementMethod>();
        audit_linear_system = &require<GlobalLinearSystem>();
        history_manager=find<GlobalDiffSimManager>();
        history_position_gradient_enabled=world().scene().info()["diff_sim"].value(
            "yarn_rotation_shear_history_gradient",true);
        std::cout<<"[yarn-shear-history-gradient-v1] history-position adjoint="
                 <<(history_position_gradient_enabled?"ON":"OFF (approximate)")
                 <<"; physical history and fixed-position parameter sensitivities retained"<<std::endl;

        auto& dof_predictor = require<DofPredictor>();

        dof_predictor.on_predict(*this,
                                 [&](DofPredictor::PredictInfo& info)
                                 { anchor_plastic_strain_update(); });  // Update Plastic Theta
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        std::cout << "In the (nonlinear) friction hardening plastic stvk Initialization"
                  << std::endl;



        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        geo_slots = world().scene().geometries();

        auto N = info.primitive_count();
        if(!N)return;
        rotation_enabled.resize(N);
        plastic_parameter_jacobians.resize(N);anchor_parameter_jacobians.resize(N);
        hardening_parameter_jacobians.resize(N);hardening_parameter_jacobians.fill(Vector9::Zero());
        state_transition.resize(N);position_transition.resize(N);
        plastic_parameter_jacobians.fill(Vector9::Zero());anchor_parameter_jacobians.fill(Vector9::Zero());
        vector<I32> enabled_values;enabled_values.reserve(N);
        vector<Float> history_measure_values;history_measure_values.reserve(N);

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

        SizeT current_geometry=0;
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
                    enabled_values.push_back(sc.triangles().find<I32>("enable_yarn_rotation")->view()[tri]);
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
                    const Vector3 e1=position_view[tri_v_idx(1)]-position_view[tri_v_idx(0)];
                    const Vector3 e2=position_view[tri_v_idx(2)]-position_view[tri_v_idx(0)];
                    history_measure_values.push_back(0.5*e1.cross(e2).norm()*
                        sc.vertices().find<Float>(builtin::thickness)->view()[tri_v_idx(0)]);
                    Vector3i tri_v_idx_offset =
                        tri_v_idx + Vector3i::Ones() * vertex_offset[current_geometry];
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
                    auto initial_G=yarn_rotation::geometry(curr_x,rest_x).strain;
                    G << initial_G(0),initial_G(2),initial_G(2),initial_G(1);

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
                ++current_geometry;
            });
        rotation_enabled.view().copy_from(enabled_values.data());
        history_measures.resize(N);history_measures.view().copy_from(history_measure_values.data());

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
        fem->hardening_plastic_strain_uvs.resize(N);
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
        ParallelFor().file_line(__FILE__,__LINE__).apply(info.indices().size(),
            [k11=k11s.cviewer(),k22=k22s.cviewer(),k12=k12s.cviewer(),k0=k33_0s.cviewer(),k1=k33_1s.cviewer(),
             kf=friction_k33s.cviewer(),pu=plastic_uus.cviewer(),pv=plastic_vvs.cviewer(),pq=plastic_uvs.cviewer(),
             aq=anchor_uvs.cviewer(),enabled=rotation_enabled.cviewer(),X=rest_xs.cviewer(),
             idx=info.indices().viewer(),xs=info.xs().viewer(),energies=info.energies().viewer(),
             thickness=info.thicknesses().viewer(),areas=info.rest_areas().viewer(),dt=info.dt()]
            __device__(int I) mutable
            {
                auto ids=idx(I);Vector9 x,parameters;
                for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                parameters<<k11(I),k22(I),k12(I),k0(I),k1(I),kf(I),0,0,0;
                const auto g=yarn_rotation::geometry(x,X(I));
                const auto e=yarn_rotation::energy_in_strain(g.strain,parameters,pu(I),pv(I),pq(I),aq(I),enabled(I)!=0);
                energies(I)=e.value*areas(I)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
            });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        const bool audit_raw=info.is_adjoint() && audit_linear_system
                              && audit_linear_system->diagnostic_raw_material_hessian();
        ParallelFor().file_line(__FILE__,__LINE__).apply(info.indices().size(),
            [k11=k11s.cviewer(),k22=k22s.cviewer(),k12=k12s.cviewer(),k0=k33_0s.cviewer(),k1=k33_1s.cviewer(),
             kf=friction_k33s.cviewer(),pu=plastic_uus.cviewer(),pv=plastic_vvs.cviewer(),pq=plastic_uvs.cviewer(),
             aq=anchor_uvs.cviewer(),enabled=rotation_enabled.cviewer(),X=rest_xs.cviewer(),
             idx=info.indices().viewer(),xs=info.xs().viewer(),G3s=info.gradients().viewer(),H3x3s=info.hessians().viewer(),
             thickness=info.thicknesses().viewer(),areas=info.rest_areas().viewer(),
             fixed=info.is_fixed().viewer(),dt=info.dt(),audit_raw]
            __device__(int I) mutable
            {
                auto ids=idx(I);Vector9 x,parameters;
                for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                parameters<<k11(I),k22(I),k12(I),k0(I),k1(I),kf(I),0,0,0;
                auto e=yarn_rotation::evaluate(x,X(I),parameters,pu(I),pv(I),pq(I),aq(I),enabled(I)!=0);
                const Float scale=areas(I)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                Vector9 grad=e.gradient*scale;Matrix9x9 H=e.hessian*scale;
                if(!audit_raw) make_spd(H);
                Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                DoubletVectorAssembler DVA{G3s};
                DVA.segment<3>(I*3).write(ids,ignore,grad);
                TripletMatrixAssembler TMA{H3x3s};
                TMA.block<3,3>(I*9).write(ids,ignore,H);
            });
    }
};

REGISTER_SIM_SYSTEM(YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D);

class YarnRotationShearHistoryDiffDofReporter final : public FiniteElementDiffDofReporter
{
  public:
    using FiniteElementDiffDofReporter::FiniteElementDiffDofReporter;
    using Matrix3x9H = Eigen::Matrix<Float,3,9>;
    YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D* model=nullptr;
    vector<Vector2i> source_pairs;
    vector<Matrix3x9H> source_jacobians;
    muda::DeviceBuffer<Vector2i> pairs;
    muda::DeviceBuffer<Matrix3x9H> jacobians;
    muda::DeviceBuffer<Vector9> plastic_force,anchor_force;
    void do_build(BuildInfo&) override
    {
        model=&require<YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D>();
    }
    void do_report_extent(DiffDofExtentInfo& info) override
    {
        source_pairs.clear();source_jacobians.clear();
        const SizeT F=info.frame(),N=model->tri_idxs.size();
        if(!model->history_position_gradient_enabled || F<=1 || !N)
        {info.triplet_count(0);return;}
        auto& history=model->position_history;
        UIPC_ASSERT(history.size()>=F,"Missing yarn shear transition history");
        for(SizeT I=0;I<N;++I)
        {
            Matrix3x3 transfer=Matrix3x3::Identity();
            for(SizeT update=F;update>=2;--update)
            {
                const auto& step=history[update-1];
                Matrix3x9H source=transfer*step.source[I];
                if(source.cwiseAbs().maxCoeff()!=0.0)
                {
                    source_pairs.push_back(Vector2i{IndexT(I),IndexT(update-1)});
                    source_jacobians.push_back(source);
                }
                transfer=(transfer*step.state[I]).eval();
                if(transfer.cwiseAbs().maxCoeff()==0.0)break;
            }
        }
        info.triplet_count(source_pairs.size()*9);
    }
    void do_assemble(DiffDofInfo& info) override
    {
        using namespace muda;
        const auto count=source_pairs.size();
        if(!count)return;
        auto mapped=source_pairs;
        for(auto& pair:mapped)pair[1]=info.dof_offset(pair[1])/3;
        pairs.resize(count);jacobians.resize(count);
        pairs.view().copy_from(mapped.data());jacobians.view().copy_from(source_jacobians.data());
        const auto N=model->tri_idxs.size();plastic_force.resize(N);anchor_force.resize(N);
        ParallelFor().file_line(__FILE__,__LINE__).apply(N,
            [ids=model->tri_idxs.cviewer(),xs=model->fem->xs().cviewer(),X=model->rest_xs.cviewer(),
             k11=model->k11s.cviewer(),k22=model->k22s.cviewer(),k12=model->k12s.cviewer(),
             k0=model->k33_0s.cviewer(),k1=model->k33_1s.cviewer(),kf=model->friction_k33s.cviewer(),
             pu=model->plastic_uus.cviewer(),pv=model->plastic_vvs.cviewer(),pq=model->plastic_uvs.cviewer(),
             aq=model->anchor_uvs.cviewer(),enabled=model->rotation_enabled.cviewer(),
             rp=plastic_force.viewer(),ra=anchor_force.viewer()]
            __device__(int I) mutable
            {
                const auto idx=ids(I);Vector9 x,p;
                for(int j=0;j<3;++j)x.segment<3>(3*j)=xs(idx(j));
                p<<k11(I),k22(I),k12(I),k0(I),k1(I),kf(I),0,0,0;
                rp(I)=yarn_rotation::evaluate(x,X(I),p,pu(I),pv(I),pq(I),aq(I),enabled(I)!=0,-1,Float(1),Float(0)).parameter_gradient;
                ra(I)=yarn_rotation::evaluate(x,X(I),p,pu(I),pv(I),pq(I),aq(I),enabled(I)!=0,-1,Float(0),Float(1)).parameter_gradient;
            });
        ParallelFor().file_line(__FILE__,__LINE__).apply(count,
            [pairs=pairs.cviewer(),B=jacobians.cviewer(),rp=plastic_force.cviewer(),ra=anchor_force.cviewer(),
             ids=model->tri_idxs.cviewer(),fixed=model->fem->is_fixed().cviewer(),
             measures=model->history_measures.cviewer(),
             H=info.H().viewer(),row_offset=info.dof_offset(info.frame())/3,dt=info.dt()]
            __device__(int I) mutable
            {
                const auto pair=pairs(I);const auto idx=ids(pair[0]);
                const Float scale=measures(pair[0])*dt*dt;
                for(int r=0;r<3;++r)for(int c=0;c<3;++c)
                {
                    Matrix3x3 block=Matrix3x3::Zero();
                    if(!fixed(idx(r)) && !fixed(idx(c)))
                        block=scale*(rp(pair[0]).segment<3>(3*r)*B(I).row(0).segment<3>(3*c)
                                    +ra(pair[0]).segment<3>(3*r)*B(I).row(2).segment<3>(3*c));
                    H(I*9+r*3+c).write(row_offset+idx(r),pair[1]+idx(c),block);
                }
            });
    }
};
REGISTER_SIM_SYSTEM(YarnRotationShearHistoryDiffDofReporter);




class YarnRotationStVKReplayer final : public AdjointMethodReplayer
{
  public:
    using AdjointMethodReplayer::AdjointMethodReplayer;
    struct State
    {
        Float plastic,anchor,hardening,yield;
        Vector9 dplastic,danchor,dhardening;
    };
    YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D* model=nullptr;
    muda::DeviceBuffer<State> states;
    void do_build(BuildInfo&) override
    {
        model=&require<YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D>();
    }
    void do_record(RecordInfo& info) override
    {
        using namespace muda;
        const auto N=model->plastic_uvs.size();
        if(!N)return;
        const auto offset=(info.frame()-1)*N;
        UIPC_ASSERT(states.size()==offset,"Yarn rotation replay frame mismatch");
        states.resize(offset+N);
        ParallelFor().file_line(__FILE__,__LINE__).apply(N,
            [out=states.viewer(),p=model->plastic_uvs.cviewer(),a=model->anchor_uvs.cviewer(),
             h=model->hardening_plastic_uvs.cviewer(),y=model->yield_uvs.cviewer(),
             dp=model->plastic_parameter_jacobians.cviewer(),da=model->anchor_parameter_jacobians.cviewer(),
             dh=model->hardening_parameter_jacobians.cviewer(),offset]
            __device__(int I) mutable {out(offset+I)=State{p(I),a(I),h(I),y(I),dp(I),da(I),dh(I)};});
    }
    void do_replay(ReplayInfo& info) override
    {
        using namespace muda;
        const auto N=model->plastic_uvs.size();
        if(!N)return;
        const auto offset=(info.frame()-1)*N;
        UIPC_ASSERT(offset+N<=states.size(),"Missing yarn rotation replay frame");
        ParallelFor().file_line(__FILE__,__LINE__).apply(N,
            [in=states.cviewer(),p=model->plastic_uvs.viewer(),a=model->anchor_uvs.viewer(),
             h=model->hardening_plastic_uvs.viewer(),y=model->yield_uvs.viewer(),
             dp=model->plastic_parameter_jacobians.viewer(),da=model->anchor_parameter_jacobians.viewer(),
             dh=model->hardening_parameter_jacobians.viewer(),offset]
            __device__(int I) mutable
            {
                const auto v=in(offset+I);p(I)=v.plastic;a(I)=v.anchor;h(I)=v.hardening;y(I)=v.yield;
                dp(I)=v.dplastic;da(I)=v.danchor;dh(I)=v.dhardening;
            });
    }
    void do_clear() override {states.clear();}
};
REGISTER_SIM_SYSTEM(YarnRotationStVKReplayer);

class DiffYarnRotationNonlinearFrictionHardeningPlasticStVKShell2D : public Codim2DConstitutionDiffParmReporter
{
  public:
    using Base = Codim2DConstitutionDiffParmReporter;
    using Base::Base;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D& constitution() noexcept
    {
        return static_cast<YarnRotationNonlinearFrictionHardeningPlasticStVKShell2D&>(Base::constitution());
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
        using namespace muda;

        {
            const auto N=k11_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=k11_parm_indices.cviewer(),ti=k11_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=SizeT(0)] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         0,dp(f)(0),da(f)(0));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=k22_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=k22_parm_indices.cviewer(),ti=k22_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         1,dp(f)(1),da(f)(1));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=k12_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=k12_parm_indices.cviewer(),ti=k12_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count + k22_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         2,dp(f)(2),da(f)(2));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=k33_0_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=k33_0_parm_indices.cviewer(),ti=k33_0_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         3,dp(f)(3),da(f)(3));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=k33_1_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=k33_1_parm_indices.cviewer(),ti=k33_1_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count + k33_0_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         4,dp(f)(4),da(f)(4));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=friction_k33_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=friction_k33_parm_indices.cviewer(),ti=friction_k33_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         5,dp(f)(5),da(f)(5));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=thres_uv_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=thres_uv_parm_indices.cviewer(),ti=thres_uv_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count + friction_k33_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         6,dp(f)(6),da(f)(6));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=yield_uv_ori_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=yield_uv_ori_parm_indices.cviewer(),ti=yield_uv_ori_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count + friction_k33_pGpP_triplet_count + thres_uv_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         7,dp(f)(7),da(f)(7));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }

        {
            const auto N=hardening_k33_parm_indices.size();
            ParallelFor().file_line(__FILE__,__LINE__).apply(N,
                [pi=hardening_k33_parm_indices.cviewer(),ti=hardening_k33_tri_indices.cviewer(),out=info.pGpP().viewer(),
                 k11=constitution().k11s.cviewer(),k22=constitution().k22s.cviewer(),k12=constitution().k12s.cviewer(),
                 k0=constitution().k33_0s.cviewer(),k1=constitution().k33_1s.cviewer(),
                 kf=constitution().friction_k33s.cviewer(),threshold=constitution().thres_uvs.cviewer(),
                 yield=constitution().yield_ori_uvs.cviewer(),hardening=constitution().hardening_k33s.cviewer(),
                 pu=constitution().plastic_uus.cviewer(),pv=constitution().plastic_vvs.cviewer(),
                 pq=constitution().plastic_uvs.cviewer(),aq=constitution().anchor_uvs.cviewer(),
                 dp=constitution().plastic_parameter_jacobians.cviewer(),da=constitution().anchor_parameter_jacobians.cviewer(),
                 enabled=constitution().rotation_enabled.cviewer(),X=constitution().rest_xs.cviewer(),
                 indices=info.indices().viewer(),xs=info.xs().viewer(),areas=info.rest_areas().viewer(),
                 thickness=info.thicknesses().viewer(),fixed=info.is_fixed().viewer(),dt=info.dt(),
                 offset=k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count + k33_0_pGpP_triplet_count + k33_1_pGpP_triplet_count + friction_k33_pGpP_triplet_count + thres_uv_pGpP_triplet_count + yield_uv_ori_pGpP_triplet_count] __device__(int I) mutable
                {
                    const auto f=ti(I);const auto ids=indices(f);
                    Vector9 x,p;
                    for(int j=0;j<3;++j)x.segment<3>(j*3)=xs(ids(j));
                    p<<k11(f),k22(f),k12(f),k0(f),k1(f),kf(f),threshold(f),yield(f),hardening(f);
                    const auto e=yarn_rotation::evaluate(x,X(f),p,pu(f),pv(f),pq(f),aq(f),enabled(f)!=0,
                                                         8,dp(f)(8),da(f)(8));
                    const Float scale=areas(f)*triangle_thickness(thickness(ids(0)),thickness(ids(1)),thickness(ids(2)))*dt*dt;
                    Vector9 mixed=e.parameter_gradient*scale;
                    Vector3i ignore={fixed(ids(0)),fixed(ids(1)),fixed(ids(2))};
                    zero_out(mixed,ignore);
                    for(int j=0;j<3;++j)out(offset+I*3+j).write(ids(j),pi(I),mixed.segment<3>(j*3));
                });
        }
    }
};
REGISTER_SIM_SYSTEM(DiffYarnRotationNonlinearFrictionHardeningPlasticStVKShell2D);
} // namespace uipc::backend::cuda
