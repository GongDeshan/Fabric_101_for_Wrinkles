#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/simplicial_surface_reporter.h>
#include <uipc/common/zip.h>
#include <uipc/common/enumerate.h>
#include <muda/cub/device/device_select.h>
#include <muda/atomic.h>
#include <muda/cub/device/device_scan.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalSimpicialSurfaceManager);

void GlobalSimpicialSurfaceManager::add_reporter(SimplicialSurfaceReporter* reporter) noexcept
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    m_impl.reporters.register_subsystem(*reporter);
}

muda::CBufferView<IndexT> GlobalSimpicialSurfaceManager::codim_vertices() const noexcept
{
    return m_impl.codim_vertices;
}

muda::CBufferView<IndexT> GlobalSimpicialSurfaceManager::surf_vertices() const noexcept
{
    return m_impl.surf_vertices;
}

muda::CBufferView<Vector2i> GlobalSimpicialSurfaceManager::surf_edges() const noexcept
{
    return m_impl.surf_edges;
}

muda::CBufferView<Vector3i> GlobalSimpicialSurfaceManager::surf_triangles() const noexcept
{
    return m_impl.surf_triangles;
}

muda::CBufferView<IndexT> GlobalSimpicialSurfaceManager::v2s() const noexcept
{
    return m_impl.v2s;
}

muda::CBufferView<IndexT> GlobalSimpicialSurfaceManager::surf_verts_triangles() const noexcept
{
    return m_impl.surf_verts_triangles;
}

muda::CBufferView<IndexT> GlobalSimpicialSurfaceManager::surf_verts_triangle_counts() const noexcept
{
    return m_impl.surf_verts_triangle_counts;
}

muda::CBufferView<IndexT> GlobalSimpicialSurfaceManager::surf_verts_triangle_offsets() const noexcept
{
    return m_impl.surf_verts_triangle_offsets;
}

void GlobalSimpicialSurfaceManager::do_build()
{
    m_impl.global_vertex_manager = find<GlobalVertexManager>();
}

void GlobalSimpicialSurfaceManager::Impl::init()
{
    // 1) build the core invariant data structure: reporter_infos
    auto reporter_view = reporters.view();
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;

    reporter_infos.resize(reporter_view.size());
    vector<SizeT> vertex_offsets(reporter_view.size(), 0);
    vector<SizeT> edge_offsets(reporter_view.size(), 0);
    vector<SizeT> triangle_offsets(reporter_view.size(), 0);
    for(auto&& [R, Rinfo] : zip(reporter_view, reporter_infos))
    {
        SurfaceCountInfo info;
        R->report_count(info);
        auto V = info.m_surf_vertex_count;
        auto E = info.m_surf_edge_count;
        auto F = info.m_surf_triangle_count;

        Rinfo.surf_vertex_count   = V;
        Rinfo.surf_edge_count     = E;
        Rinfo.surf_triangle_count = F;

        vertex_offsets[R->m_index]   = V;
        edge_offsets[R->m_index]     = E;
        triangle_offsets[R->m_index] = F;
    }

    std::exclusive_scan(
        vertex_offsets.begin(), vertex_offsets.end(), vertex_offsets.begin(), 0);
    std::exclusive_scan(edge_offsets.begin(), edge_offsets.end(), edge_offsets.begin(), 0);
    std::exclusive_scan(
        triangle_offsets.begin(), triangle_offsets.end(), triangle_offsets.begin(), 0);

    for(auto&& [i, Rinfo] : enumerate(reporter_infos))
    {
        Rinfo.surf_vertex_offset   = vertex_offsets[i];
        Rinfo.surf_edge_offset     = edge_offsets[i];
        Rinfo.surf_triangle_offset = triangle_offsets[i];
    }

    // set related data
    SizeT global_vert_count         = global_vertex_manager->positions().size();
    SizeT total_surf_vertex_count   = 0;
    SizeT total_surf_edge_count     = 0;
    SizeT total_surf_triangle_count = 0;

    if(!reporter_infos.empty())
    {
        auto& back_info = reporter_infos.back();
        total_surf_vertex_count = back_info.surf_vertex_offset + back_info.surf_vertex_count;
        total_surf_edge_count = back_info.surf_edge_offset + back_info.surf_edge_count;
        total_surf_triangle_count = back_info.surf_triangle_offset + back_info.surf_triangle_count;
    }

    // 2) resize the device buffer
    codim_vertices.resize(total_surf_vertex_count);
    surf_vertices.resize(total_surf_vertex_count);
    surf_edges.resize(total_surf_edge_count);
    surf_triangles.resize(total_surf_triangle_count);
    v2s.resize(global_vert_count);


    // 3) collect surface attributes
    for(auto&& [R, Rinfo] : zip(reporter_view, reporter_infos))
    {
        SurfaceAttributeInfo info{this, R->m_index};
        R->report_attributes(info);
    }

    // 4) collect Codim0D vertices
    _collect_codim_vertices();

    // 5) setup vertex to surface vertex mapping
    _setup_vertex_to_surface_vertex_mapping();

    // 6) setup verts triangles
    _setup_verts_triangles();
}

void GlobalSimpicialSurfaceManager::Impl::_collect_codim_vertices()
{
    using namespace muda;
    auto dim = global_vertex_manager->dimensions();

    codim_vertex_flags.resize(surf_vertices.size());

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(surf_vertices.size(),
               [dim           = dim.cviewer().name("dim"),
                surf_vertices = surf_vertices.viewer().name("surf_vertices"),
                flags = codim_vertex_flags.viewer().name("flags")] __device__(int I) mutable
               {
                   auto vI = surf_vertices(I);
                   flags(I) = dim(vI) <= 1 ? 1 : 0;  // codim 0D vert and vert from codim 1D edge
               });

    DeviceSelect().Flagged(surf_vertices.data(),
                           codim_vertex_flags.data(),
                           codim_vertices.data(),
                           selected_codim_0d_count.data(),
                           surf_vertices.size());

    IndexT count = selected_codim_0d_count;
    codim_vertices.resize(count);
}

void GlobalSimpicialSurfaceManager::Impl::_setup_vertex_to_surface_vertex_mapping()
{
    using namespace muda;

    v2s.fill(-1);  // fill with -1 for invalid value

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(surf_vertices.size(),
               [v2s = v2s.viewer().name("v2s"),
                surf_vertices = surf_vertices.cviewer().name("surf_vertices")] __device__(int I) mutable
               {
                   auto vI = surf_vertices(I);
                   v2s(vI) = I;
               });
}

void GlobalSimpicialSurfaceManager::Impl::_setup_verts_triangles()
{
    using namespace muda;

    auto N = surf_vertices.size() + 1;
    surf_verts_triangle_offsets.resize(N);
    surf_verts_triangle_offsets.fill(0);
    surf_verts_triangle_counts.resize(N);
    surf_verts_triangle_counts.fill(0);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(surf_triangles.size(),
               [surf_triangles = surf_triangles.viewer().name("surf_triangles"),
                surf_verts_triangle_offsets =
                    surf_verts_triangle_offsets.viewer().name("surf_verts_triangle_offsets"),
                surf_verts_triangle_counts =
                    surf_verts_triangle_counts.viewer().name("surf_verts_triangle_counts"),
                v2s = v2s.viewer().name("v2s")] __device__(int I) mutable
               {
                   Vector3i F = surf_triangles(I);

                   for(int i = 0; i < 3; ++i)
                   {
                       auto vI = F(i);
                       auto S  = v2s(vI);
                       atomic_add(&surf_verts_triangle_counts(S), 1);
                   }
               });

    DeviceScan().ExclusiveSum(surf_verts_triangle_counts.data(),
                              surf_verts_triangle_offsets.data(),
                              N);

    IndexT total;
    surf_verts_triangle_offsets.view(N - 1).copy_to(&total);  // N-1 is the last element
    surf_verts_triangles.resize(total);

    surf_verts_triangle_counts.fill(0);  // reset the counts for the next step

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(surf_triangles.size(),
               [surf_triangles = surf_triangles.viewer().name("surf_triangles"),
                surf_verts_triangles = surf_verts_triangles.viewer().name("surf_verts_triangles"),
                surf_verts_triangle_offsets =
                    surf_verts_triangle_offsets.viewer().name("surf_verts_triangle_offsets"),
                surf_verts_triangle_counts =
                    surf_verts_triangle_counts.viewer().name("surf_verts_triangle_counts"),
                v2s = v2s.viewer().name("v2s")] __device__(int I) mutable
               {
                   Vector3i F = surf_triangles(I);

                   for(int i = 0; i < 3; ++i)
                   {
                       auto vI = F(i);
                       auto S  = v2s(vI);
                       auto O  = surf_verts_triangle_offsets(S);
                       auto C  = atomic_add(&surf_verts_triangle_counts(S), 1);
                       surf_verts_triangles(O + C) = I;
                   }
               });
}

void GlobalSimpicialSurfaceManager::init()
{
    m_impl.init();
}

void GlobalSimpicialSurfaceManager::rebuild()
{
    UIPC_ASSERT(false, "Not implemented yet");
}

muda::BufferView<IndexT> GlobalSimpicialSurfaceManager::SurfaceAttributeInfo::surf_vertices() noexcept
{
    const auto& info = reporter_info();
    return m_impl->surf_vertices.view(info.surf_vertex_offset, info.surf_vertex_count);
}

muda::BufferView<Vector2i> GlobalSimpicialSurfaceManager::SurfaceAttributeInfo::surf_edges() noexcept
{
    const auto& info = reporter_info();
    return m_impl->surf_edges.view(info.surf_edge_offset, info.surf_edge_count);
}

muda::BufferView<Vector3i> GlobalSimpicialSurfaceManager::SurfaceAttributeInfo::surf_triangles() noexcept
{
    const auto& info = reporter_info();
    return m_impl->surf_triangles.view(info.surf_triangle_offset, info.surf_triangle_count);
}

auto GlobalSimpicialSurfaceManager::SurfaceAttributeInfo::reporter_info() const noexcept
    -> const ReporterInfo&
{
    return m_impl->reporter_infos[m_index];
}

void GlobalSimpicialSurfaceManager::SurfaceCountInfo::surf_vertex_count(SizeT count) noexcept
{
    m_surf_vertex_count = count;
}

void GlobalSimpicialSurfaceManager::SurfaceCountInfo::surf_edge_count(SizeT count) noexcept
{
    m_surf_edge_count = count;
}

void GlobalSimpicialSurfaceManager::SurfaceCountInfo::surf_triangle_count(SizeT count) noexcept
{
    m_surf_triangle_count = count;
}

void GlobalSimpicialSurfaceManager::SurfaceCountInfo::changable(bool value) noexcept
{
    m_changable = value;
}
}  // namespace uipc::backend::cuda
