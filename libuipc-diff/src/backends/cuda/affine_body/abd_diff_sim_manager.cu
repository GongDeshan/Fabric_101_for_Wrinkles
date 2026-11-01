#include <affine_body/abd_diff_sim_manager.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/common/range.h>
#include <uipc/common/enumerate.h>
#include <affine_body/constitutions/diff_mass_function.h>
#include <affine_body/constitutions/diff_volume_function.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/constitutions/abd_force_function.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDDiffSimManager);

void ABDDiffSimManager::do_build()
{
    m_impl.affine_body_dynamics    = &require<AffineBodyDynamics>();
    m_impl.global_diff_sim_manager = &require<GlobalDiffSimManager>();

    m_impl.global_diff_sim_manager->on_parm_update(this,
                                                   [&](GlobalDiffSimManager::DiffParmUpdateInfo& info)
                                                   { m_impl.update_parms(info); });
}

void ABDDiffSimManager::Impl::init(backend::WorldVisitor& world)
{
    auto    v_size          = affine_body_dynamics->Js().size();
    auto    b_size          = affine_body_dynamics->body_is_fixed().size();
    Vector3 default_gravity = world.scene().info()["gravity"].get<Vector3>();

    h_diff_rest_positions.resize(v_size);
    std::ranges::fill(h_diff_rest_positions, -Vector3i::Ones());  // -1 means not diff parm (default)

    h_body_rho.resize(b_size);
    h_instance_gravity.resize(b_size);

    vector<IndexT> body_need_diff(b_size, 0);

    auto rest_geo_slots = world.scene().rest_geometries();

    list<IndexT>   diff_vertices_buffer;
    list<IndexT>   bodies_buffer;
    list<Vector3i> diff_triangles_buffer;
    list<Vector3i> triangles_buffer;
    list<Vector4i> diff_tetrahedra_buffer;

    // find in the rest shape
    affine_body_dynamics->for_each(
        rest_geo_slots,
        [&](const AffineBodyDynamics::ForEachInfo& info, geometry::SimplicialComplex& sc)
        {
            auto diff_pos = sc.vertices().find<Vector3i>("diff/position");

            if(!diff_pos)  // no diff position, skip
                return;

            auto&        geo_info = info.geo_info();
            auto         b_count  = geo_info.body_count;
            const IndexT v_offset = geo_info.vertex_offset;
            auto         v_count  = sc.vertices().size();

            UIPC_ASSERT(geo_info.vertex_count == geo_info.body_count * v_count,
                        "vertex count mismatch, body vertex count:{}, body count:{}, geo vertex count:{}",
                        v_count,
                        b_count,
                        geo_info.vertex_count);

            // 1. Label the diff positions and body need diff
            auto geo_diff_pos =
                span{h_diff_rest_positions}.subspan(v_offset, v_count * b_count);
            {
                auto diff_pos_view = diff_pos->view();
                auto offset        = v_offset;
                for(auto bI : range(b_count))
                {
                    auto dst = geo_diff_pos.subspan(offset, v_count);
                    std::ranges::copy(diff_pos_view, dst.begin());
                    offset += v_count;
                }

                auto& h_v2b = affine_body_dynamics->m_impl.h_vertex_id_to_body_id;
                for(auto&& [I, v] : enumerate(geo_diff_pos))
                {
                    auto v_diff = geo_diff_pos[I];
                    if(v_diff.maxCoeff() >= 0)
                    {
                        body_need_diff[h_v2b[I]] = 1;
                    }
                }
            }


            // 2. Record the vertices and bodies
            for(auto&& [I, v] : enumerate(geo_diff_pos))
            {
                auto v_diff = geo_diff_pos[I];
                if(v_diff.maxCoeff() >= 0)
                {
                    diff_vertices_buffer.push_back(I + v_offset);
                }
            }

            for(auto bI : range(b_count))
            {
                if(body_need_diff[bI] == 1)
                {
                    bodies_buffer.push_back(bI);
                }
            }


            // 3. Record the triangles or tetrahedra
            UIPC_ASSERT(sc.dim() == 2, "Only 2D simplicial complex is supported now!");
            if(sc.dim() == 2)  // record the triangles
            {
                auto           Fs = sc.triangles().topo().view();
                list<Vector3i> body_local_diff_Fs;
                for(auto&& [i, f] : enumerate(Fs))
                {
                    auto v0 = f[0];
                    auto v1 = f[1];
                    auto v2 = f[2];

                    const Vector3i& v0_diff = geo_diff_pos[v0];
                    const Vector3i& v1_diff = geo_diff_pos[v1];
                    const Vector3i& v2_diff = geo_diff_pos[v2];

                    if(v0_diff.maxCoeff() >= 0 || v1_diff.maxCoeff() >= 0
                       || v2_diff.maxCoeff() >= 0)
                    {
                        body_local_diff_Fs.push_back(Vector3i{v0, v1, v2});
                    }
                }

                for(auto bI : range(b_count))
                {
                    auto offset = v_offset;
                    for(const Vector3i& diffF : body_local_diff_Fs)
                    {
                        diff_triangles_buffer.push_back(diffF.array() + offset);
                    }
                    offset += v_count;
                }

                for(auto bI : range(b_count))
                {
                    auto offset = v_offset;
                    for(const Vector3i& F : Fs)
                    {
                        triangles_buffer.push_back(F.array() + offset);
                    }
                    offset += v_count;
                }
            }
            else if(sc.dim() == 3)
            {
                // NOTE: just leave the code for completeness, but we haven't finish the tet based optimization

                auto           Ts = sc.tetrahedra().topo().view();
                list<Vector4i> body_local_diff_Ts;
                for(auto&& [i, t] : enumerate(Ts))
                {
                    auto v0 = t[0];
                    auto v1 = t[1];
                    auto v2 = t[2];
                    auto v3 = t[3];

                    const Vector3i& v0_diff = geo_diff_pos[v0];
                    const Vector3i& v1_diff = geo_diff_pos[v1];
                    const Vector3i& v2_diff = geo_diff_pos[v2];
                    const Vector3i& v3_diff = geo_diff_pos[v3];

                    if(v0_diff.maxCoeff() >= 0 || v1_diff.maxCoeff() >= 0
                       || v2_diff.maxCoeff() >= 0 || v3_diff.maxCoeff() >= 0)
                    {
                        body_local_diff_Ts.push_back(t);
                    }
                }

                for(auto bI : range(b_count))
                {
                    auto offset = v_offset;
                    for(const Vector4i& diffT : body_local_diff_Ts)
                    {
                        diff_tetrahedra_buffer.push_back(diffT.array() + offset);
                    }
                    offset += v_count;
                }
            }

            // 4. Record body rho
            auto rho = sc.meta().find<Float>(builtin::mass_density);
            UIPC_ASSERT(rho, "Mass density is required for diff optimization!");
            auto geo_rhos = span{h_body_rho}.subspan(geo_info.body_offset, b_count);
            std::ranges::fill(geo_rhos, rho->view()[0]);

            // 5. Record the gravity
            auto gravity = sc.instances().find<Vector3>(builtin::gravity);
            if(gravity)
            {
                auto geo_gravity =
                    span{h_instance_gravity}.subspan(geo_info.body_offset, b_count);
                std::ranges::copy(gravity->view(), geo_gravity.begin());
            }
            else
            {
                auto geo_gravity =
                    span{h_instance_gravity}.subspan(geo_info.body_offset, b_count);
                std::ranges::fill(geo_gravity, default_gravity);
            }
        });

    // 4. Copy to device
    //h_triangles.resize(triangles_buffer.size());
    //std::ranges::copy(triangles_buffer, h_triangles.begin());
    //triangles.copy_from(h_triangles);

    h_diff_vertices.resize(diff_vertices_buffer.size());
    std::ranges::copy(diff_vertices_buffer, h_diff_vertices.begin());
    diff_vertices.copy_from(h_diff_vertices);

    h_diff_bodies.resize(bodies_buffer.size());
    std::ranges::copy(bodies_buffer, h_diff_bodies.begin());
    diff_bodies.copy_from(h_diff_bodies);

    h_diff_triangles.resize(diff_triangles_buffer.size());
    std::ranges::copy(diff_triangles_buffer, h_diff_triangles.begin());
    diff_triangles.copy_from(h_diff_triangles);

    h_diff_tetrahedra.resize(diff_tetrahedra_buffer.size());
    std::ranges::copy(diff_tetrahedra_buffer, h_diff_tetrahedra.begin());
    diff_tetrahedra.copy_from(h_diff_tetrahedra);

    diff_rest_positions.copy_from(h_diff_rest_positions);
    body_rho.copy_from(h_body_rho);

    instance_gravity.copy_from(h_instance_gravity);
}

void ABDDiffSimManager::Impl::update_parms(GlobalDiffSimManager::DiffParmUpdateInfo& info)
{
    using namespace muda;

    auto Js       = affine_body_dynamics->m_impl.vertex_id_to_J.view();
    auto volumes  = affine_body_dynamics->m_impl.body_id_to_volume.view();
    auto abd_mass = affine_body_dynamics->m_impl.body_id_to_abd_mass.view();
    auto abd_mass_inv = affine_body_dynamics->m_impl.body_id_to_abd_mass_inv.view();
    auto abd_gravity = affine_body_dynamics->m_impl.body_id_to_abd_gravity.view();
    auto v2b = affine_body_dynamics->v2b();

    // 1. Update the diff rest positions
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(h_diff_vertices.size(),
               [parms         = info.parameters().viewer().name("parms"),
                diff_vertices = diff_vertices.viewer().name("diff_vertices"),
                diff_rest_positions = diff_rest_positions.viewer().name("diff_rest_positions"),
                Js = Js.viewer().name("Js")] __device__(int I)
               {
                   auto     vI            = diff_vertices(I);
                   Vector3  x_bar         = Js(vI).x_bar();
                   Vector3i diff_rest_pos = diff_rest_positions(vI);
                   for(int i = 0; i < 3; ++i)
                   {
                       if(diff_rest_pos[i] >= 0)
                       {
                           // update the diff rest position from the parms
                           x_bar[i] = parms(diff_rest_pos[i]);
                       }
                   }
                   Js(vI) = ABDJacobi{x_bar};
               });

    // 2. Update the mass volume and gravity
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diff_bodies.size(),
               [diff_bodies = diff_bodies.viewer().name("diff_bodies"),
                abd_gravity = abd_gravity.viewer().name("abd_gravity"),
                abd_mass    = abd_mass.viewer().name("abd_mass"),
                volumes = volumes.viewer().name("volumes")] __device__(int I)
               {
                   // Clear the diff body gravity/mass/volume
                   IndexT bI       = diff_bodies(I);
                   abd_gravity(bI) = Vector12::Zero();
                   abd_mass(bI)    = ABDJacobiDyadicMass::zero();
                   volumes(bI)     = 0.0f;
                   // Remain the other bodies no-change
               });

    // Now just atomic add the delta mass, later we can use a more efficient way like reduction
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diff_triangles.size(),
               [diff_triangles = diff_triangles.viewer().name("diff_triangles"),
                body_rho       = body_rho.viewer().name("body_rho"),
                v2b            = v2b.viewer().name("v2b"),
                Js             = Js.viewer().name("Js"),
                abd_mass       = abd_mass.viewer().name("abd_mass"),
                volumes        = volumes.viewer().name("volumes"),
                abd_gravity    = abd_gravity.viewer().name("abd_gravity"),
                instance_gravity = instance_gravity.viewer().name("instance_gravity"),
                parms = info.parameters().viewer().name("parms")] __device__(int I)
               {
                   Vector3i F = diff_triangles(I);

                   MUDA_ASSERT(v2b(F[0]) == v2b(F[1]) && v2b(F[0]) == v2b(F[2]),
                               "Triangle (%d,%d,%d) is not in the same body, which is in body (%d,%d,%d), why can it happen?",
                               F[0],
                               F[1],
                               F[2],
                               v2b(F[0]),
                               v2b(F[1]),
                               v2b(F[2]));

                   Vector9 X0;
                   X0.segment<3>(0) = Js(F[0]).x_bar();
                   X0.segment<3>(3) = Js(F[1]).x_bar();
                   X0.segment<3>(6) = Js(F[2]).x_bar();

                   IndexT bI  = v2b(F[0]);
                   Float  rho = body_rho(bI);

                   Float m;
                   sym::diff_m::m(m, X0, rho);

                   Vector3 La;
                   sym::diff_m::La(La, X0, rho);

                   Matrix3x3 Lab;
                   sym::diff_m::Lab(Lab, X0, rho);
                   auto local_mass = ABDJacobiDyadicMass::from_dyadic_mass(m, La, Lab);
                   ABDJacobiDyadicMass::atomic_add(abd_mass(bI), local_mass);

                   Float V;
                   sym::diff_v::V(V, X0);
                   atomic_add(&volumes(bI), V);

                   Vector12 abd_gravity_force;
                   Vector3  force_density = rho * instance_gravity(bI);

                   sym::force::F(abd_gravity_force, X0, force_density);
                   muda::eigen::atomic_add(abd_gravity(bI), abd_gravity_force);
               });

    // update the inverse mass and abd gravity
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diff_bodies.size(),
               [diff_bodies  = diff_bodies.viewer().name("diff_bodies"),
                abd_mass     = abd_mass.viewer().name("abd_mass"),
                abd_mass_inv = abd_mass_inv.viewer().name("abd_mass_inv"),
                abd_gravity = abd_gravity.viewer().name("abd_gravity")] __device__(int I)
               {
                   IndexT bI = diff_bodies(I);
                   abd_mass_inv(bI) = muda::eigen::inverse(abd_mass(bI).to_mat());
                   abd_gravity(bI) = abd_mass_inv(bI) * abd_gravity(bI);
               });
}

void ABDDiffSimManager::init()
{
    m_impl.init(world());
}
}  // namespace uipc::backend::cuda
