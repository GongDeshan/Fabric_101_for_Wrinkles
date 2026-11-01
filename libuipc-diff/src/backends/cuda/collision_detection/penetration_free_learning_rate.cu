#include <diff_sim/learning_rate_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <collision_detection/global_trajectory_filter.h>
namespace uipc::backend::cuda
{
class PenetrationFreeLearningRate final : public LearningRateReporter
{
  public:
    using LearningRateReporter::LearningRateReporter;

    GlobalVertexManager*        vertex_manager    = nullptr;
    GlobalTrajectoryFilter*     trajectory_filter = nullptr;
    muda::DeviceBuffer<Vector3> prev_positions;
    muda::DeviceBuffer<Vector3> displacements;


    virtual void do_build(BuildInfo& info) override
    {
        vertex_manager    = &require<GlobalVertexManager>();
        trajectory_filter = &require<GlobalTrajectoryFilter>();
    }

    virtual void record_safe_state(RecordSafeStateInfo& info) override
    {
        auto positions = vertex_manager->positions();
        prev_positions.resize(positions.size());
        prev_positions.view().copy_from(positions);
    }

    virtual void report_learning_rate(LearningRateInfo& info) override
    {
        using namespace muda;

        // 1. Compute Displacements From Previous Positions to Current Positions
        {
            auto positions = vertex_manager->positions();  // new positions
            displacements.resize(positions.size());

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(positions.size(),
                       [positions      = positions.viewer().name("positions"),
                        disp           = displacements.viewer().name("disp"),
                        prev_positions = prev_positions.cviewer().name(
                            "prev_positions")] __device__(int I)
                       {
                           //
                           disp(I) = positions(I) - prev_positions(I);
                       });
        }

        // 2. Set Displacements to the Vertex Manager
        vertex_manager->update(prev_positions, displacements);  // set the displacements manually

        // 3. Filter the TOI
        trajectory_filter->detect(1.0);
        Float alpha = trajectory_filter->filter_toi(1.0);  // filter the toi

        info.learning_rate(alpha);
    }
};

REGISTER_SIM_SYSTEM(PenetrationFreeLearningRate);
}  // namespace uipc::backend::cuda
