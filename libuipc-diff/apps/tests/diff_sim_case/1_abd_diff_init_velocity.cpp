#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("1_abd_diff_init_velocity", "[abd]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    std::string trimesh_dir{AssetDir::trimesh_path()};
    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto        this_output_path = AssetDir::output_path(__FILE__);


    Engine engine{"cuda", this_output_path};
    World  world{engine};

    auto config = Scene::default_config();

    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = false;
    config["line_search"]["max_iter"]       = 8;
    config["linear_system"]["tol_rate"]     = 1e-3;

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    Transform pre_transform = Transform::Identity();
    pre_transform.scale(0.23);
    SimplicialComplexIO io{pre_transform};

    Scene scene{config};
    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    diff_sim.parameters().resize(2);  // 3 parameters for linear velocity of affine body

    // create constitution and contact model
    AffineBodyConstitution abd;

    // create object
    auto object = scene.objects().create("cube");

    auto mesh = io.read(fmt::format("{}bunny0.msh", tetmesh_dir));
    label_surface(mesh);
    label_triangle_orient(mesh);

    //auto mesh = io.read(fmt::format("{}cube.obj", trimesh_dir));
    //label_surface(mesh);

    abd.apply_to(mesh, 100.0_MPa);
    {
        auto vel      = mesh.instances().create<Matrix4x4>(builtin::velocity);
        auto diff_vel = mesh.instances().create<Matrix4x4i>("diff/velocity");
        // label the translation part
        Matrix4x4i indices = -Matrix4x4i::Ones();
        indices(0, 3)      = 0;  // -> map to the 0th parameter
        indices(2, 3)      = 1;  // -> map to the 1st parameter
        view(*diff_vel)[0] = indices;
    }

    auto [geo_slot, rest_geo_slot] = object->geometries().create(mesh);

    auto ground_obj = scene.objects().create("ground");
    auto g          = ground(-0.18);
    ground_obj->geometries().create(g);

    SceneIO sio{scene};
    world.init(scene);

    world.dump();
    sio.write_surface(
        fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));

    spdlog::set_level(spdlog::level::warn);

    constexpr IndexT EndFrame = 10;
    Vector2          aim_pos{2, 1};  // x z
    constexpr Float  K = 50.0;

    using test_case::DiffSimSolver;

    DiffSimSolver solver{world, this_output_path};
    solver.save_surf(true);
    solver.total_frame_count(EndFrame);

    auto& U = solver.U();
    U.resize(2);  // 2 parameters for linear velocity of affine body
    U.segment<2>(0) = Vector2::Zero();  // init parm

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
    REQUIRE(solver.gradient_descent(15));
}