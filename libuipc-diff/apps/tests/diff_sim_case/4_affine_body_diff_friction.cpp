#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("4_affine_body_diff_friction", "[abd]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
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
    pre_transform.scale(0.23);
    SimplicialComplexIO io{pre_transform};

    Scene scene{config};
    scene.contact_tabular().default_model(0.5, 0.1_GPa);
    auto self = scene.contact_tabular().create("self");
    scene.contact_tabular().insert(self, self, 0, 0, false);

    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    parms.resize(2);  // 2 parameters for init velocity (x,z) of particle

    // create constitution and contact model
    AffineBodyConstitution abd;

    // create object
    auto object = scene.objects().create("bunny");

    vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};
    vector<Vector3>  Vs = {Vector3{0, 1, 0},
                           Vector3{0, 0, 1},
                           Vector3{-std::sqrt(3) / 2, 0, -0.5},
                           Vector3{std::sqrt(3) / 2, 0, -0.5}};

    //auto mesh = tetmesh(Vs, Ts);
    auto mesh = io.read(fmt::format("{}bunny0.msh", tetmesh_dir));
    label_surface(mesh);
    label_triangle_orient(mesh);
    Float rho = 1000.0;
    abd.apply_to(mesh, 100.0_MPa, rho);
    self.apply_to(mesh);  // ignore self contact

    auto attr_volume = mesh.instances().find<Float>(builtin::volume);
    auto volume      = attr_volume->view()[0];
    auto mass        = volume * rho;

    // init velocity
    auto vel = mesh.instances().create<Matrix4x4>(builtin::velocity);

    auto diff_vel = mesh.instances().create<Matrix4x4i>("diff/velocity");
    // control the x and z velocity
    Matrix4x4i diff_vel_index = -Matrix4x4i::Ones();  // mask out
    diff_vel_index(0, 3)      = 0;                    // map to parms[0]
    diff_vel_index(2, 3)      = 1;                    // map to parms[1]
    view(*diff_vel)[0]        = diff_vel_index;

    auto [geo_slot, rest_geo_slot] = object->geometries().create(mesh);

    auto ground_obj = scene.objects().create("ground");
    auto g          = ground(-0.2);
    ground_obj->geometries().create(g);

    world.init(scene);
    REQUIRE(world.is_valid());

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

    spdlog::set_level(spdlog::level::warn);

    IndexT  EndFrame = 20;
    Vector2 aim_pos{3, 0};  // x z
    Float   K = mass;

    using test_case::DiffSimSolver;
    DiffSimSolver solver{world, this_output_path};
    solver.total_frame_count(EndFrame);
    solver.save_surf(true);

    auto& U = solver.U();
    U.resize(2);  // 2 parameters for init velocity (x,z) of particle
    U.segment<2>(0) = Vector2{1, 1};  // 2 parameters for init velocity (x,z) of particle


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

    bool  is_first = true;
    Float loss0    = 0;

    solver.L(
        [&](DiffSimSolver::ComputeLossInfo& info) -> Float
        {
            auto& X    = info.X();
            auto  loss = 0.5 * K * (X - aim_pos).squaredNorm();
            if(is_first)
            {
                loss0    = loss;
                is_first = false;
            }

            return loss;
        });

    solver.dLdX(
        [&](DiffSimSolver::ComputedLdXInfo& info)
        {
            auto& X    = info.X();
            auto& dLdX = info.dLdX();
            dLdX       = K * (X - aim_pos);
        });

    bool converged = false;

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