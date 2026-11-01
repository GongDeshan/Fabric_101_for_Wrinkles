#include <linear_system/global_linear_system.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/off_diag_linear_subsystem.h>
#include <uipc/common/range.h>
#include <linear_system/iterative_solver.h>
#include <linear_system/global_preconditioner.h>
#include <linear_system/local_preconditioner.h>
#include <fstream>
#include <sim_engine.h>
#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalLinearSystem);

void GlobalLinearSystem::dump_linear_system(std::string_view filename)
{
    {
        auto& A = m_impl.debug_A;
        m_impl.ctx.convert(m_impl.bcoo_A, A);
        Eigen::MatrixXd mat;
        A.copy_to(mat);

        auto A_file = fmt::format("{}.A.csv", filename);

        std::ofstream file(A_file);
        // dump as .csv file
        for(int i = 0; i < mat.rows(); ++i)
        {
            for(int j = 0; j < mat.cols(); ++j)
            {
                file << mat(i, j) << ",";
            }
            file << "\n";
        }
    }

    {
        Eigen::VectorXd b;
        m_impl.b.copy_to(b);

        auto b_file = fmt::format("{}.b.csv", filename);

        std::ofstream file(b_file);
        // dump as .csv file
        for(int i = 0; i < b.size(); ++i)
        {
            file << b(i) << "\n";
        }
    }

    {
        Eigen::VectorXd x;
        m_impl.x.copy_to(x);

        auto x_file = fmt::format("{}.x.csv", filename);

        std::ofstream file(x_file);
        // dump as .csv file
        for(int i = 0; i < x.size(); ++i)
        {
            file << x(i) << "\n";
        }
    }
}

SizeT GlobalLinearSystem::dof_count() const
{
    return m_impl.b.size();
}

void GlobalLinearSystem::do_build() {}

// 2026-10-09, opt-in buckling audit. Kept separate from build_linear_system():
// raw export must NOT build a preconditioner or enter the SPD-only PCG solver.
void GlobalLinearSystem::dump_diagnostic_hessian(std::string_view filename,
                                                bool raw_material)
{
    const bool old_adjoint = m_impl.adjoint_assembly;
    const bool old_raw = m_diagnostic_raw_material_hessian;
    m_impl.adjoint_assembly = true;
    m_diagnostic_raw_material_hessian = raw_material;
    try
    {
        m_impl.empty_system = !m_impl._update_subsystem_extent();
        if(m_impl.empty_system)
            throw std::runtime_error("Buckling audit: empty linear system");
        m_impl._assemble_linear_system();
        m_impl.converter.convert(m_impl.triplet_A, m_impl.bcoo_A);
        // No ge2sym: export the FULL assembled matrix, not half storage.
        auto& A = m_impl.debug_A;
        m_impl.ctx.convert(m_impl.bcoo_A, A);
        Eigen::MatrixXd mat;
        A.copy_to(mat);
        Eigen::VectorXd gradient;
        m_impl.b.copy_to(gradient);
        if(!mat.allFinite() || !gradient.allFinite())
            throw std::runtime_error("Buckling audit: non-finite matrix or residual");
        const std::string prefix{filename};
        const auto parent = std::filesystem::path(prefix).parent_path();
        if(!parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream matrix_file(prefix + ".A.mtx");
        std::ofstream gradient_file(prefix + ".gradient.csv");
        std::ofstream metadata_file(prefix + ".json");
        if(!matrix_file || !gradient_file || !metadata_file)
            throw std::runtime_error("Buckling audit: cannot open matrix output");
        matrix_file << std::setprecision(17);
        gradient_file << std::setprecision(17);
        Eigen::Index nnz = 0;
        for(Eigen::Index i = 0; i < mat.rows(); ++i)
            for(Eigen::Index j = 0; j < mat.cols(); ++j)
                if(mat(i, j) != 0.0) ++nnz;
        matrix_file << "%%MatrixMarket matrix coordinate real general\n"
                    << mat.rows() << " " << mat.cols() << " " << nnz << "\n";
        for(Eigen::Index i = 0; i < mat.rows(); ++i)
            for(Eigen::Index j = 0; j < mat.cols(); ++j)
                if(mat(i, j) != 0.0)
                    matrix_file << i+1 << " " << j+1 << " " << mat(i, j) << "\n";
        for(Eigen::Index i = 0; i < gradient.size(); ++i)
            gradient_file << gradient(i) << "\n";
        // Includes inertia, soft constraints, fixed-DOF treatment and contacts.
        // ONLY the two audited material projections are bypassed in raw mode.
        metadata_file << std::setprecision(17) << "{\"version\":\"buckling-single-step-v1\",\"frame\":"
                      << engine().frame() << ",\"rows\":" << mat.rows()
                      << ",\"raw_material\":" << (raw_material ? "true" : "false")
                      << ",\"gradient_norm\":" << gradient.norm()
                      << ",\"scope\":\"orthotropic bending and nonlinear StVK only\","
                      << "\"includes_inertia_and_constraints\":true,\"solver_called\":false}\n";
        matrix_file.flush(); gradient_file.flush(); metadata_file.flush();
        if(!matrix_file || !gradient_file || !metadata_file)
            throw std::runtime_error("Buckling audit: incomplete matrix output");
        spdlog::info("[buckling-single-step-v1] Exported {} material_raw={}, dofs={}",
                     prefix, raw_material, mat.rows());
    }
    catch(...)
    {
        m_impl.adjoint_assembly = old_adjoint;
        m_diagnostic_raw_material_hessian = old_raw;
        throw;
    }
    m_impl.adjoint_assembly = old_adjoint;
    m_diagnostic_raw_material_hessian = old_raw;
}

void GlobalLinearSystem::solve()
{
    UIPC_ASSERT(!m_diagnostic_raw_material_hessian,
                "Raw diagnostic Hessian must never enter Newton/PCG");
    m_impl.build_linear_system();
    // if the system is empty, skip the following steps
    if(m_impl.empty_system) [[unlikely]]
        return;
    m_impl.solve_linear_system();
    m_impl.distribute_solution();
}

void GlobalLinearSystem::solve_adjoint_variable(muda::DenseVectorView<Float> lambda_i,
                                                muda::CDenseVectorView<Float> rhs)
{
    m_impl.adjoint_assembly = true;
    UIPC_ASSERT(!m_diagnostic_raw_material_hessian,
                "Raw diagnostic Hessian must never enter adjoint PCG");
    try
    {
        m_impl.build_linear_system();
    }
    catch(...)
    {
        m_impl.adjoint_assembly = false;
        throw;
    }
    m_impl.adjoint_assembly = false;
    // if the system is empty, skip the following steps
    if(m_impl.empty_system) [[unlikely]]
        return;
    m_impl.solve_adjoint_system(lambda_i, rhs);
    //dump_linear_system(fmt::format("hessian_{}", engine().frame()));
}

bool GlobalLinearSystem::DiagInfo::is_adjoint() const
{
    return m_impl->adjoint_assembly;
}

void GlobalLinearSystem::prepare_hessian(PrepareHessianInfo& info)
{
    m_impl.empty_system = !m_impl._update_subsystem_extent();
    // if empty, skip the following steps
    if(m_impl.empty_system) [[unlikely]]
        return;

    m_impl._assemble_linear_system();
    m_impl.converter.convert(m_impl.triplet_A, m_impl.bcoo_A);
    if(info.half_storage)
        m_impl.converter.ge2sym(m_impl.bcoo_A);
}

void GlobalLinearSystem::collect_dofs(muda::BufferView<Float>   dofs,
                                      muda::CBufferView<IndexT> indices)
{
    m_impl.collect_dofs(dofs, indices);
}

void GlobalLinearSystem::Impl::init()
{
    auto diag_subsystem_view       = diag_subsystems.view();
    auto off_diag_subsystem_view   = off_diag_subsystems.view();
    auto local_preconditioner_view = local_preconditioners.view();

    // 1) Record Diag and OffDiag Subsystems
    auto total_count = diag_subsystem_view.size() + off_diag_subsystem_view.size();
    subsystem_infos.resize(total_count);
    // put the diag subsystems in the front
    auto diag_span = span{subsystem_infos}.subspan(0, diag_subsystem_view.size());
    // then the off diag subsystems
    auto off_diag_span = span{subsystem_infos}.subspan(diag_subsystem_view.size(),
                                                       off_diag_subsystem_view.size());
    {
        auto offset = 0;
        for(auto i : range(diag_span.size()))
        {
            auto& dst_diag                  = diag_span[i];
            dst_diag.is_diag                = true;
            dst_diag.local_index            = i;
            auto index                      = offset + i;
            dst_diag.index                  = index;
            diag_subsystem_view[i]->m_index = index;
        }

        offset += diag_subsystem_view.size();
        for(auto i : range(off_diag_span.size()))
        {
            auto& dst_off_diag       = off_diag_span[i];
            dst_off_diag.is_diag     = false;
            dst_off_diag.local_index = i;
            dst_off_diag.index       = offset + i;
        }
    }

    // 2) DoF Offsets/Counts
    accuracy_statisfied_flags.resize(diag_subsystem_view.size());
    {
        diag_dof_offsets_counts.resize(diag_subsystem_view.size());
        auto diag_dof_counts = diag_dof_offsets_counts.counts();
        for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
        {
            InitDofExtentInfo info;
            diag_subsystem->report_init_extent(info);
            diag_dof_counts[i] = info.m_dof_count;
        }
        diag_dof_offsets_counts.scan();
        auto diag_dof_offsets = diag_dof_offsets_counts.offsets();
        for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
        {
            InitDofInfo info;
            info.m_dof_offset = diag_dof_offsets[i];
            info.m_dof_count  = diag_dof_counts[i];
            diag_subsystem->receive_init_dof_info(info);
        }
    }

    // 3) Triplet Offsets/Counts
    subsystem_triplet_offsets_counts.resize(total_count);
    off_diag_lr_triplet_counts.resize(off_diag_subsystem_view.size());

    // 4) Preconditioner
    // find out diag systems that don't have preconditioner
    for(auto precond : local_preconditioner_view)
    {
        auto index = precond->m_subsystem->m_index;
        diag_span[index].has_local_preconditioner = true;
    }
    no_precond_diag_subsystem_indices.reserve(diag_span.size());
    for(auto&& [i, diag_info] : enumerate(diag_span))
    {
        if(!diag_info.has_local_preconditioner)
        {
            no_precond_diag_subsystem_indices.push_back(i);
        }
    }
}

void GlobalLinearSystem::Impl::build_linear_system()
{
    Timer timer{"Build Linear System"};
    empty_system = !_update_subsystem_extent();
    // if empty, skip the following steps
    if(empty_system) [[unlikely]]
        return;


    _assemble_linear_system();

    converter.convert(triplet_A, bcoo_A);
    converter.ge2sym(bcoo_A);

    _assemble_preconditioner();
}

bool GlobalLinearSystem::Impl::_update_subsystem_extent()
{
    bool dof_count_changed     = false;
    bool triplet_count_changed = false;

    auto diag_subsystem_view       = diag_subsystems.view();
    auto off_diag_subsystem_view   = off_diag_subsystems.view();
    auto diag_dof_counts           = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets          = diag_dof_offsets_counts.offsets();
    auto subsystem_triplet_counts  = subsystem_triplet_offsets_counts.counts();
    auto subsystem_triplet_offsets = subsystem_triplet_offsets_counts.offsets();

    for(const auto& subsystem_info : subsystem_infos)
    {
        if(subsystem_info.is_diag)
        {
            auto           dof_i          = subsystem_info.local_index;
            auto           triplet_i      = subsystem_info.index;
            auto&          diag_subsystem = diag_subsystem_view[dof_i];
            DiagExtentInfo info;
            info.m_storage_type = HessianStorageType::Full;
            diag_subsystem->report_extent(info);

            dof_count_changed |= diag_dof_counts[dof_i] != info.m_dof_count;
            diag_dof_counts[dof_i] = info.m_dof_count;


            triplet_count_changed |= subsystem_triplet_counts[triplet_i] != info.m_block_count;
            subsystem_triplet_counts[triplet_i] = info.m_block_count;
        }
        else
        {
            auto triplet_i = subsystem_info.index;
            auto& off_diag_subsystem = off_diag_subsystem_view[subsystem_info.local_index];
            OffDiagExtentInfo info;
            info.m_storage_type = HessianStorageType::Full;
            off_diag_subsystem->report_extent(info);

            auto total_block_count = info.m_lr_block_count + info.m_rl_block_count;

            triplet_count_changed |= subsystem_triplet_counts[triplet_i] != total_block_count;
            subsystem_triplet_counts[triplet_i] = total_block_count;
            off_diag_lr_triplet_counts[subsystem_info.local_index] =
                ulonglong2{info.m_lr_block_count, info.m_rl_block_count};
        }
    }

    SizeT total_dof     = 0;
    SizeT total_triplet = 0;

    if(dof_count_changed)
    {
        diag_dof_offsets_counts.scan();
    }
    total_dof = diag_dof_offsets_counts.total_count();
    if(x.capacity() < total_dof)
    {
        auto reserve_count = total_dof * reserve_ratio;
        x.reserve(reserve_count);
        b.reserve(reserve_count);
    }
    auto blocked_dof = total_dof / DoFBlockSize;
    triplet_A.reshape(blocked_dof, blocked_dof);
    x.resize(total_dof);
    b.resize(total_dof);

    if(triplet_count_changed) [[likely]]
    {
        subsystem_triplet_offsets_counts.scan();
    }
    total_triplet = subsystem_triplet_offsets_counts.total_count();

    if(triplet_A.triplet_capacity() < total_triplet)
    {
        auto reserve_count = total_triplet * reserve_ratio;
        triplet_A.reserve_triplets(reserve_count);
        bcoo_A.reserve_triplets(reserve_count);
    }
    triplet_A.resize_triplets(total_triplet);

    if(total_dof == 0 || total_triplet == 0) [[unlikely]]
    {
        spdlog::warn("The global linear system is empty, skip *assembling, *solving and *solution distributing phase.");
        return false;
    }

    return true;
}

void GlobalLinearSystem::Impl::_assemble_linear_system()
{
    auto HA = triplet_A.view();
    auto B  = b.view();

    auto diag_subsystem_view     = diag_subsystems.view();
    auto off_diag_subsystem_view = off_diag_subsystems.view();

    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    auto subsystem_triplet_counts  = subsystem_triplet_offsets_counts.counts();
    auto subsystem_triplet_offsets = subsystem_triplet_offsets_counts.offsets();

    for(const auto& subsystem_info : subsystem_infos)
    {
        if(subsystem_info.is_diag)
        {
            auto  dof_i          = subsystem_info.local_index;
            auto  triplet_i      = subsystem_info.index;
            auto& diag_subsystem = diag_subsystem_view[dof_i];

            int  dof_offset         = diag_dof_offsets[dof_i];
            int  dof_count          = diag_dof_counts[dof_i];
            int  blocked_dof_offset = dof_offset / DoFBlockSize;
            int  blocked_dof_count  = dof_count / DoFBlockSize;
            int2 ij_offset          = {blocked_dof_offset, blocked_dof_offset};
            int2 ij_count           = {blocked_dof_count, blocked_dof_count};

            DiagInfo info{this};

            info.m_index        = triplet_i;
            info.m_storage_type = HessianStorageType::Full;
            info.m_gradient     = B.subview(dof_offset, dof_count);
            info.m_hessian = HA.subview(subsystem_triplet_offsets[triplet_i],
                                        subsystem_triplet_counts[triplet_i])
                                 .submatrix(ij_offset, ij_count);

            diag_subsystem->assemble(info);
        }
        else
        {
            auto triplet_i   = subsystem_info.index;
            auto local_index = subsystem_info.local_index;
            auto& off_diag_subsystem = off_diag_subsystem_view[subsystem_info.local_index];
            auto& l_diag_index = off_diag_subsystem->m_l->m_index;
            auto& r_diag_index = off_diag_subsystem->m_r->m_index;


            int l_blocked_dof_offset = diag_dof_offsets[l_diag_index] / DoFBlockSize;
            int l_blocked_dof_count = diag_dof_counts[l_diag_index] / DoFBlockSize;

            int r_blocked_dof_offset = diag_dof_offsets[r_diag_index] / DoFBlockSize;
            int r_blocked_dof_count = diag_dof_counts[r_diag_index] / DoFBlockSize;

            auto lr_triplet_offset = subsystem_triplet_offsets[triplet_i];
            auto lr_triplet_count  = off_diag_lr_triplet_counts[local_index].x;
            auto rl_triplet_offset = lr_triplet_offset + lr_triplet_count;
            auto rl_triplet_count  = off_diag_lr_triplet_counts[local_index].y;

            OffDiagInfo info{this};
            info.m_index        = triplet_i;
            info.m_storage_type = HessianStorageType::Full;

            info.m_lr_hessian =
                HA.subview(lr_triplet_offset, lr_triplet_count)
                    .submatrix(int2{l_blocked_dof_offset, r_blocked_dof_offset},
                               int2{l_blocked_dof_count, r_blocked_dof_count});

            info.m_rl_hessian =
                HA.subview(rl_triplet_offset, rl_triplet_count)
                    .submatrix(int2{r_blocked_dof_offset, l_blocked_dof_offset},
                               int2{r_blocked_dof_count, l_blocked_dof_count});

            // spdlog::info("rl_offset: {}, lr_offset: {}", rl_triplet_offset, lr_triplet_offset);

            off_diag_subsystem->assemble(info);
        }
    }
}

void GlobalLinearSystem::Impl::_assemble_preconditioner()
{
    if(global_preconditioner)
    {
        GlobalPreconditionerAssemblyInfo info{this};
        global_preconditioner->assemble(info);
    }

    for(auto&& preconditioner : local_preconditioners.view())
    {
        LocalPreconditionerAssemblyInfo info{this, preconditioner->m_subsystem->m_index};
        preconditioner->assemble(info);
    }
}

void GlobalLinearSystem::Impl::solve_linear_system()
{
    UIPC_ASSERT(iterative_solver, "Now we only support iterative solver, but none is provided.");
    SolvingInfo info{this};
    info.m_b = b.cview();
    info.m_x = x.view();

    {
        Timer timer{"Solve Linear System"};
        iterative_solver->solve(info);
    }

    spdlog::info("Iterative linear solver iteration count: {}", info.m_iter_count);
}

void GlobalLinearSystem::Impl::solve_adjoint_system(muda::DenseVectorView<Float> lambda_i,
                                                    muda::CDenseVectorView<Float> rhs)
{
    UIPC_ASSERT(iterative_solver, "Now we only support iterative solver, but none is provided.");
    SolvingInfo info{this};
    info.m_b = rhs;
    info.m_x = lambda_i;

    {
        Timer timer{"Solve Adjoint System"};
        iterative_solver->solve(info);
    }

    spdlog::info("Iterative linear solver iteration count: {}", info.m_iter_count);
}

void GlobalLinearSystem::Impl::distribute_solution()
{
    auto diag_subsystem_view = diag_subsystems.view();
    auto diag_dof_counts     = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets    = diag_dof_offsets_counts.offsets();

    // distribute the solution to all diag subsystems
    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystems.view()))
    {
        SolutionInfo info{this};
        info.m_solution = x.view().subview(diag_dof_offsets[i], diag_dof_counts[i]);
        diag_subsystem->retrieve_solution(info);
    }
}

void GlobalLinearSystem::Impl::collect_dofs(muda::BufferView<Float> selected_dofs,
                                            muda::CBufferView<IndexT> indices)
{
    auto total_dof = x.size();
    this->dofs.resize(total_dof);
    muda::DenseVectorView<Float> dofs_view    = dofs.view();
    auto                         diag_systems = diag_subsystems.view();
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();
    // collect all dofs from diag systems
    for(auto&& [i, diag_system] : enumerate(diag_systems))
    {
        CurrentDofInfo info;
        info.m_dofs = dofs_view.subview(diag_dof_offsets[i], diag_dof_counts[i]);
        diag_system->report_dofs(info);
    }

    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(indices.size(),
               [dst     = selected_dofs.viewer().name("dst"),
                src     = dofs.viewer().name("src"),
                indices = indices.viewer().name("indices")] __device__(int i)
               {
                   auto index = indices(i);
                   dst(i)     = src(index);
               });
}

void GlobalLinearSystem::Impl::apply_preconditioner(muda::DenseVectorView<Float> z,
                                                    muda::CDenseVectorView<Float> r)
{
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    if(global_preconditioner)
    {
        ApplyPreconditionerInfo info{this};
        info.m_z = z;
        info.m_r = r;
        global_preconditioner->apply(info);
    }

    for(auto& preconditioner : local_preconditioners.view())
    {
        ApplyPreconditionerInfo info{this};
        auto                    index  = preconditioner->m_subsystem->m_index;
        auto                    offset = diag_dof_offsets[index];
        auto                    count  = diag_dof_counts[index];
        info.m_z                       = z.subview(offset, count);
        info.m_r                       = r.subview(offset, count);
        preconditioner->apply(info);
    }

    if(!global_preconditioner)
    {
        for(auto i : no_precond_diag_subsystem_indices)
        {
            auto offset = diag_dof_offsets[i];
            auto count  = diag_dof_counts[i];
            auto z_sub  = z.subview(offset, count);
            auto r_sub  = r.subview(offset, count);
            z_sub.buffer_view().copy_from(r_sub.buffer_view());
        }
    }
}

void GlobalLinearSystem::Impl::spmv(Float                         a,
                                    muda::CDenseVectorView<Float> x,
                                    Float                         b,
                                    muda::DenseVectorView<Float>  y)
{
    spmver.rbk_sym_spmv(a, bcoo_A.cview(), x, b, y);
}

bool GlobalLinearSystem::Impl::accuracy_statisfied(muda::DenseVectorView<Float> r)
{
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    for(auto&& [i, diag_subsystems] : enumerate(diag_subsystems.view()))
    {
        AccuracyInfo info{this};
        info.m_r = r.subview(diag_dof_offsets[i], diag_dof_counts[i]);
        diag_subsystems->accuracy_check(info);

        accuracy_statisfied_flags[i] = info.m_statisfied ? 1 : 0;
    }

    return std::ranges::all_of(accuracy_statisfied_flags,
                               [](bool flag) { return flag; });
}

void GlobalLinearSystem::DiagExtentInfo::extent(SizeT hessian_block_count, SizeT dof_count) noexcept
{
    m_block_count = hessian_block_count;
    UIPC_ASSERT(dof_count % DoFBlockSize == 0,
                "dof_count must be multiple of {}, yours {}.",
                DoFBlockSize,
                dof_count);
    m_dof_count = dof_count;
}

void GlobalLinearSystem::OffDiagExtentInfo::extent(SizeT lr_hessian_block_count,
                                                   SizeT rl_hassian_block_count) noexcept
{
    m_lr_block_count = lr_hessian_block_count;
    m_rl_block_count = rl_hassian_block_count;
}
auto GlobalLinearSystem::AssemblyInfo::A() const -> CBCOOMatrixView
{
    return m_impl->bcoo_A.cview();
}

auto GlobalLinearSystem::AssemblyInfo::storage_type() const -> HessianStorageType
{
    return HessianStorageType::Symmetric;
}
SizeT GlobalLinearSystem::LocalPreconditionerAssemblyInfo::dof_offset() const
{
    auto diag_dof_offsets = m_impl->diag_dof_offsets_counts.offsets();
    return diag_dof_offsets[m_index];
}
SizeT GlobalLinearSystem::LocalPreconditionerAssemblyInfo::dof_count() const
{
    auto diag_dof_counts = m_impl->diag_dof_offsets_counts.counts();
    return diag_dof_counts[m_index];
}
IndexT GlobalLinearSystem::DiagInfo::dof_offset()
{
    return m_gradient.offset();
}
IndexT GlobalLinearSystem::DiagInfo::dof_count()
{
    return m_gradient.size();
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
void GlobalLinearSystem::add_subsystem(DiagLinearSubsystem* subsystem)
{
    check_state(SimEngineState::BuildSystems, "add_subsystem()");
    UIPC_ASSERT(subsystem != nullptr, "The subsystem should not be nullptr.");
    m_impl.diag_subsystems.register_subsystem(*subsystem);
}

void GlobalLinearSystem::add_subsystem(OffDiagLinearSubsystem* subsystem)
{
    check_state(SimEngineState::BuildSystems, "add_subsystem()");
    m_impl.off_diag_subsystems.register_subsystem(*subsystem);
}

void GlobalLinearSystem::add_solver(IterativeSolver* solver)
{
    check_state(SimEngineState::BuildSystems, "add_solver()");
    UIPC_ASSERT(solver != nullptr, "The solver should not be nullptr.");
    m_impl.iterative_solver.register_subsystem(*solver);
}

void GlobalLinearSystem::add_preconditioner(LocalPreconditioner* preconditioner)
{
    check_state(SimEngineState::BuildSystems, "add_preconditioner()");
    UIPC_ASSERT(preconditioner != nullptr, "The preconditioner should not be nullptr.");
    m_impl.local_preconditioners.register_subsystem(*preconditioner);
}

void GlobalLinearSystem::add_preconditioner(GlobalPreconditioner* preconditioner)
{
    check_state(SimEngineState::BuildSystems, "add_preconditioner()");
    UIPC_ASSERT(preconditioner != nullptr, "The preconditioner should not be nullptr.");
    m_impl.global_preconditioner.register_subsystem(*preconditioner);
}

void GlobalLinearSystem::init()
{
    m_impl.init();
}
}  // namespace uipc::backend::cuda
