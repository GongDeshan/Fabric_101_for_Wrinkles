#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("6_affine_body_bump", "[abd]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    std::string trimesh_dir{AssetDir::trimesh_path()};
    auto        this_output_path = AssetDir::output_path(__FILE__);

    Engine engine{"cuda", this_output_path};
    World  world{engine};

    auto config = Scene::default_config();

    config["gravity"]                       = Vector3{0, -10, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["d_hat"]              = 0.01;
    config["contact"]["friction"]["enable"] = true;
    config["line_search"]["max_iter"]       = 8;
    config["linear_system"]["tol_rate"]     = 1e-3;

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    Transform pre_transform = Transform::Identity();
    //pre_transform.scale(0.23);
    pre_transform.scale(0.3);
    SimplicialComplexIO io{pre_transform};

    Scene scene{config};
    scene.contact_tabular().default_model(0.4, 1.0_GPa);
    auto self0 = scene.contact_tabular().create("self0");
    auto self1 = scene.contact_tabular().create("self1");
    scene.contact_tabular().insert(self0, self0, 0, 0, false);
    scene.contact_tabular().insert(self1, self1, 0, 0, false);

    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    parms.resize(2);  // 2 parameters for init velocity (x,z)

    // create constitution and contact model
    AffineBodyConstitution abd;

    // create object
    auto object = scene.objects().create("bunny");

    auto mesh = io.read(fmt::format("{}cube.obj", trimesh_dir));
    // auto mesh = io.read(fmt::format("{}bunny0.msh", tetmesh_dir));

    label_surface(mesh);
    if(mesh.dim() == 3)
        label_triangle_orient(mesh);
    Float rho = 1000.0;
    abd.apply_to(mesh, 100.0_MPa, rho);

    auto attr_volume = mesh.instances().find<Float>(builtin::volume);
    auto volume      = attr_volume->view()[0];
    auto mass        = volume * rho;

    {
        SimplicialComplex mesh0 = mesh;
        self0.apply_to(mesh0);  // ignore self contact
        // init velocity
        auto       vel = mesh0.instances().create<Matrix4x4>(builtin::velocity);
        Matrix4x4i diff_vel_index = -Matrix4x4i::Ones();  // mask out
        auto diff_vel = mesh0.instances().create<Matrix4x4i>("diff/velocity", diff_vel_index);
        // control the x and z velocity
        diff_vel_index(0, 3) = 0;  // map to parms[0]
        diff_vel_index(2, 3) = 1;  // map to parms[1]
        view(*diff_vel)[0]   = diff_vel_index;
        object->geometries().create(mesh0);
    }


    SimplicialComplex mesh1 = mesh;
    self1.apply_to(mesh1);  // ignore self contact
    Transform t = Transform::Identity();
    t.translate(Vector3{1, 0, 0});
    view(mesh1.transforms())[0]    = t.matrix();
    auto [geo_slot, rest_geo_slot] = object->geometries().create(mesh1);

    auto ground_obj = scene.objects().create("ground");
    auto g          = ground(-0.2);
    ground_obj->geometries().create(g);

    world.init(scene);
    REQUIRE(world.is_valid());

    spdlog::set_level(spdlog::level::info);

    IndexT  EndFrame = 20;
    Vector3 aim_pos{2, 0, 0};
    Float   K = 10 * mass;

    using test_case::DiffSimSolver;

    DiffSimSolver solver{world, this_output_path};
    solver.total_frame_count(EndFrame);
    solver.save_surf(true);

    auto& U = solver.U();
    U.resize(2);
    U.segment<2>(0) = Vector2{8, 0};

    solver.SelectDofs(
        [&](DiffSimSolver::SelectDofInfo& info)
        {
            if(info.frame() < EndFrame)
                return;

            auto& geo = geo_slot->geometry();

            auto offset   = geo.meta().find<IndexT>(builtin::dof_offset);
            auto offset_v = offset->view()[0];

            auto& sdi = info.sdi();

            sdi.resize(2);
            sdi(0) = offset_v + 0;
            sdi(1) = offset_v + 2;
        });

    solver.L(
        [&](DiffSimSolver::ComputeLossInfo& info) -> Float
        {
            auto& X = info.X();
            return 0.5 * K * (X - aim_pos).squaredNorm();
        });

    solver.dLdX(
        [&](DiffSimSolver::ComputedLdXInfo& info)
        {
            auto& X    = info.X();
            auto& dLdX = info.dLdX();
            dLdX       = K * (X - aim_pos);
        });

    solver.ExitCondition(
        [&](DiffSimSolver::ExitConditionInfo& info) -> bool
        {
            auto& dLdU = info.dLdU();

            auto norm = dLdU.lpNorm<Eigen::Infinity>();
            std::cout << "dLdU Norm: " << norm << std::endl;
            auto rel_loss = info.rel_loss();
            std::cout << "rel loss: " << rel_loss << std::endl;
            std::cout << "X:" << info.X().transpose() << std::endl;
            return norm < 1e-4 || rel_loss < 1e-4;
        });

    // should converge in 15 iterations
    REQUIRE(solver.gradient_descent(1));
}