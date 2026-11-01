#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/particle.h>
#include <filesystem>
#include <fstream>

TEST_CASE("37_particle_init_velocity", "[fem]")
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

    config["gravity"]                   = Vector3{0, -9.8, 0};
    config["contact"]["enable"]         = false;  // disable contact
    config["line_search"]["max_iter"]   = 8;
    config["linear_system"]["tol_rate"] = 1e-3;

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    SimplicialComplexIO io;

    Scene scene{config};
    {
        // create constitution and contact model
        Particle pt;

        // create object
        auto object = scene.objects().create("point");

        vector<Vector3> Vs = {Vector3{0, 0, 0}};

        auto mesh = pointcloud(Vs);

        label_surface(mesh);
        pt.apply_to(mesh, 1e3);

        auto vel = mesh.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
        auto speed = Vector3{5, 5, 0};

        std::ranges::fill(view(*vel), speed);  // 45 degree

        object->geometries().create(mesh);
    }

    world.init(scene);
    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

    while(world.frame() < 100)
    {
        world.advance();
        world.retrieve();
        sio.write_surface(
            fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));
    }
}