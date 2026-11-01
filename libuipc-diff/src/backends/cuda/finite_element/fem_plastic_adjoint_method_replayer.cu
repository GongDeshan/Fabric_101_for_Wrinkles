#include <finite_element/fem_plastic_adjoint_method_replayer.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMPlasticAdjointMethodReplayer);

void FEMPlasticAdjointMethodReplayer::do_build(BuildInfo& info)
{
    m_impl.finite_element_method = &require<FiniteElementMethod>();

    m_impl.edge_offset_stack.push(0);  // dummy frame 1
    m_impl.orthotropic_edge_offset_stack.push(0);  // dummy frame 1

    m_impl.face_offset_stack.push(0);  // dummy frame 1
}

static void append(muda::DeviceBuffer<Float>& buffer, muda::CBufferView<Float> data)
{
    auto offset = buffer.size();
    auto count  = data.size();
    auto total  = offset + count;
    buffer.resize(total);
    buffer.view(offset, count).copy_from(data);
}

void FEMPlasticAdjointMethodReplayer::do_record(RecordInfo& info)
{
    // Confirmed this function is called in backpropgration

    //std::cout << "Replayer Record" << std::endl;

    // Copy Edge Plastic Informations;
    auto& edge_stack = m_impl.edge_offset_stack;
    auto& ortho_edge_stack = m_impl.orthotropic_edge_offset_stack;

    auto F             = info.frame();  // 1,2,...,N
    auto RecordedFrame = edge_stack.size();
    UIPC_ASSERT(F == RecordedFrame,
                "Frame mismatch (Edge), Input Frame={}, Last Frame={}, why can it happen?",
                F,
                RecordedFrame);

    auto geo = world().scene().geometries().front()->geometry().as<geometry::SimplicialComplex>();

    //auto e_count = geo->edges().size();

    //edge_stack.push(edge_stack.top() + e_count);  // update the vertex offsets for the next frame

    auto iso_count = m_impl.finite_element_method->plastic_thetas.size();
    auto ortho_count = m_impl.finite_element_method->plastic_thetas_warp.size();

    UIPC_ASSERT(m_impl.finite_element_method->dihedral_thetas.size() == iso_count
                    && m_impl.finite_element_method->anchor_thetas.size() == iso_count
                    && m_impl.finite_element_method->d_anchor_theta_d_thres_theta.size()
                           == iso_count
                    && m_impl.finite_element_method->d_plastic_theta_d_yield_ori.size()
                           == iso_count
                    && m_impl.finite_element_method->d_plastic_theta_d_k.size()
                           == iso_count
                    && m_impl.finite_element_method->d_plastic_theta_d_hardening_k.size()
                           == iso_count,
                "Isotropic bending replay buffer size mismatch");

    UIPC_ASSERT(ortho_edge_stack.size() == F,
                "Frame mismatch (Orthotropic Edge), Input Frame={}, Last Frame={}, why can it happen?",
                F,
                ortho_edge_stack.size());
    UIPC_ASSERT(m_impl.finite_element_method->plastic_thetas_weft.size()
                        == ortho_count
                    && m_impl.finite_element_method->anchor_thetas_warp.size()
                           == ortho_count
                    && m_impl.finite_element_method->anchor_thetas_weft.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_anchor_theta_d_thres_theta_warp.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_anchor_theta_d_thres_theta_weft.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_plastic_theta_d_k_warp.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_plastic_theta_d_k_weft.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_plastic_theta_d_yield_ori_warp.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_plastic_theta_d_yield_ori_weft.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_plastic_theta_d_hardening_k_warp.size()
                           == ortho_count
                    && m_impl.finite_element_method->d_plastic_theta_d_hardening_k_weft.size()
                           == ortho_count,
                "Orthotropic bending replay buffer size mismatch");

    edge_stack.push(edge_stack.top() + iso_count);
    ortho_edge_stack.push(ortho_edge_stack.top() + ortho_count);

    append(m_impl.dihedral_thetas, m_impl.finite_element_method->dihedral_thetas);
    append(m_impl.plastic_thetas, m_impl.finite_element_method->plastic_thetas);
    append(m_impl.anchor_thetas, m_impl.finite_element_method->anchor_thetas);

    append(m_impl.d_anchor_theta_d_thres_theta,
           m_impl.finite_element_method->d_anchor_theta_d_thres_theta);
    append(m_impl.d_plastic_theta_d_yield_ori,
           m_impl.finite_element_method->d_plastic_theta_d_yield_ori);
    append(m_impl.d_plastic_theta_d_k, m_impl.finite_element_method->d_plastic_theta_d_k);
    append(m_impl.d_plastic_theta_d_hardening_k,
           m_impl.finite_element_method->d_plastic_theta_d_hardening_k);

    // Orthotropic Bending
    append(m_impl.anchor_thetas_warp, m_impl.finite_element_method->anchor_thetas_warp);
    append(m_impl.anchor_thetas_weft, m_impl.finite_element_method->anchor_thetas_weft);

    append(m_impl.plastic_thetas_warp, m_impl.finite_element_method->plastic_thetas_warp);
    append(m_impl.plastic_thetas_weft, m_impl.finite_element_method->plastic_thetas_weft);

    append(m_impl.d_anchor_theta_d_thres_theta_warp,
           m_impl.finite_element_method->d_anchor_theta_d_thres_theta_warp);
    append(m_impl.d_anchor_theta_d_thres_theta_weft,
           m_impl.finite_element_method->d_anchor_theta_d_thres_theta_weft);

    append(m_impl.d_plastic_theta_d_k_warp,
           m_impl.finite_element_method->d_plastic_theta_d_k_warp);
    append(m_impl.d_plastic_theta_d_k_weft,
           m_impl.finite_element_method->d_plastic_theta_d_k_weft);

    append(m_impl.d_plastic_theta_d_yield_ori_warp,
           m_impl.finite_element_method->d_plastic_theta_d_yield_ori_warp);
    append(m_impl.d_plastic_theta_d_yield_ori_weft,
           m_impl.finite_element_method->d_plastic_theta_d_yield_ori_weft);

    append(m_impl.d_plastic_theta_d_hardening_k_warp,
           m_impl.finite_element_method->d_plastic_theta_d_hardening_k_warp);
    append(m_impl.d_plastic_theta_d_hardening_k_weft,
           m_impl.finite_element_method->d_plastic_theta_d_hardening_k_weft);

    // Copy Face Plastic Informations;
    auto& face_stack = m_impl.face_offset_stack;

    RecordedFrame = face_stack.size();
    UIPC_ASSERT(F == RecordedFrame,
                "Frame mismatch (Face), Input Frame={}, Last Frame={}, why can it happen?",
                F,
                RecordedFrame);

    auto f_count = geo->triangles().size();

    face_stack.push(face_stack.top() + f_count);

    append(m_impl.strain_uus, m_impl.finite_element_method->strain_uus);
    append(m_impl.strain_vvs, m_impl.finite_element_method->strain_vvs);
    append(m_impl.strain_uvs, m_impl.finite_element_method->strain_uvs);

    append(m_impl.yield_strain_uus, m_impl.finite_element_method->yield_strain_uus);
    append(m_impl.yield_strain_vvs, m_impl.finite_element_method->yield_strain_vvs);
    append(m_impl.yield_strain_uvs, m_impl.finite_element_method->yield_strain_uvs);

    append(m_impl.hardening_plastic_strain_uus,
           m_impl.finite_element_method->hardening_plastic_strain_uus);
    append(m_impl.hardening_plastic_strain_vvs,
           m_impl.finite_element_method->hardening_plastic_strain_vvs);
    append(m_impl.hardening_plastic_strain_uvs,
           m_impl.finite_element_method->hardening_plastic_strain_uvs);

    append(m_impl.plastic_strain_uus, m_impl.finite_element_method->plastic_strain_uus);
    append(m_impl.plastic_strain_vvs, m_impl.finite_element_method->plastic_strain_vvs);
    append(m_impl.plastic_strain_uvs, m_impl.finite_element_method->plastic_strain_uvs);

    append(m_impl.anchor_strain_uus, m_impl.finite_element_method->anchor_strain_uus);
    append(m_impl.anchor_strain_vvs, m_impl.finite_element_method->anchor_strain_vvs);
    append(m_impl.anchor_strain_uvs, m_impl.finite_element_method->anchor_strain_uvs);

    append(m_impl.d_anchor_uv_d_thres_uv, m_impl.finite_element_method->d_anchor_uv_d_thres_uv);
    append(m_impl.d_plastic_uv_d_hardening_k33,
           m_impl.finite_element_method->d_plastic_uv_d_hardening_k33);
    append(m_impl.d_plastic_uv_d_k33, m_impl.finite_element_method->d_plastic_uv_d_k33);
    append(m_impl.d_plastic_uv_d_yield_ori,
           m_impl.finite_element_method->d_plastic_uv_d_yield_ori);
}

void FEMPlasticAdjointMethodReplayer::do_replay(ReplayInfo& info)
{
    // Confirmed this function is called in backpropgration
    //std::cout << "Plastic Replayer Replay" << std::endl;

    // replay (Edge) plastic information
    auto& edge_stack    = m_impl.edge_offset_stack;
    auto  next_e_offset = edge_stack.top();
    edge_stack.pop();  // pop the last frame

    auto F             = info.frame();
    auto StackTopFrame = edge_stack.size();

    UIPC_ASSERT(F == StackTopFrame,
                "Frame mismatch (Edge), Input Frame={}, Stack Top Frame={}, why can it happen?",
                F,
                StackTopFrame);

    auto e_offset = edge_stack.top();  // get offset

    auto geo = world().scene().geometries().front()->geometry().as<geometry::SimplicialComplex>();

    auto e_count = m_impl.finite_element_method->plastic_thetas.size();

    UIPC_ASSERT(e_offset + e_count == next_e_offset,
                "Edge offset mismatch, e_offset={}, e_count={}, next_e_offset={}, why can it happen?",
                e_offset,
                e_count,
                next_e_offset);

    // copy the recorded data to the FEM
    m_impl.finite_element_method->dihedral_thetas.view().copy_from(
        m_impl.dihedral_thetas.view(e_offset, e_count));
    m_impl.finite_element_method->plastic_thetas.view().copy_from(
        m_impl.plastic_thetas.view(e_offset, e_count));
    m_impl.finite_element_method->anchor_thetas.view().copy_from(
        m_impl.anchor_thetas.view(e_offset, e_count));

    // For bending friction thres grad
    m_impl.finite_element_method->d_anchor_theta_d_thres_theta.view().copy_from(
        m_impl.d_anchor_theta_d_thres_theta.view(e_offset, e_count));
    // For bending plastic gradient
    m_impl.finite_element_method->d_plastic_theta_d_yield_ori.view().copy_from(
        m_impl.d_plastic_theta_d_yield_ori.view(e_offset, e_count));
    m_impl.finite_element_method->d_plastic_theta_d_hardening_k.view().copy_from(
        m_impl.d_plastic_theta_d_hardening_k.view(e_offset, e_count));
    m_impl.finite_element_method->d_plastic_theta_d_k.view().copy_from(
        m_impl.d_plastic_theta_d_k.view(e_offset, e_count));

    // Orthotropic bending is indexed by valid bending stencils, independently
    // from the isotropic bending buffers above.
    auto& ortho_edge_stack = m_impl.orthotropic_edge_offset_stack;
    auto  next_ortho_e_offset = ortho_edge_stack.top();
    ortho_edge_stack.pop();

    StackTopFrame = ortho_edge_stack.size();
    UIPC_ASSERT(F == StackTopFrame,
                "Frame mismatch (Orthotropic Edge), Input Frame={}, Stack Top Frame={}, why can it happen?",
                F,
                StackTopFrame);

    auto ortho_e_offset = ortho_edge_stack.top();
    auto ortho_e_count = m_impl.finite_element_method->plastic_thetas_warp.size();
    UIPC_ASSERT(ortho_e_offset + ortho_e_count == next_ortho_e_offset,
                "Orthotropic edge offset mismatch, e_offset={}, e_count={}, next_e_offset={}, why can it happen?",
                ortho_e_offset,
                ortho_e_count,
                next_ortho_e_offset);

    m_impl.finite_element_method->plastic_thetas_warp.view().copy_from(
        m_impl.plastic_thetas_warp.view(ortho_e_offset, ortho_e_count));
    m_impl.finite_element_method->plastic_thetas_weft.view().copy_from(
        m_impl.plastic_thetas_weft.view(ortho_e_offset, ortho_e_count));

    m_impl.finite_element_method->anchor_thetas_warp.view().copy_from(
        m_impl.anchor_thetas_warp.view(ortho_e_offset, ortho_e_count));
    m_impl.finite_element_method->anchor_thetas_weft.view().copy_from(
        m_impl.anchor_thetas_weft.view(ortho_e_offset, ortho_e_count));

    m_impl.finite_element_method->d_anchor_theta_d_thres_theta_warp.view().copy_from(
        m_impl.d_anchor_theta_d_thres_theta_warp.view(ortho_e_offset, ortho_e_count));
    m_impl.finite_element_method->d_anchor_theta_d_thres_theta_weft.view().copy_from(
        m_impl.d_anchor_theta_d_thres_theta_weft.view(ortho_e_offset, ortho_e_count));

    m_impl.finite_element_method->d_plastic_theta_d_yield_ori_warp.view().copy_from(
        m_impl.d_plastic_theta_d_yield_ori_warp.view(ortho_e_offset, ortho_e_count));
    m_impl.finite_element_method->d_plastic_theta_d_yield_ori_weft.view().copy_from(
        m_impl.d_plastic_theta_d_yield_ori_weft.view(ortho_e_offset, ortho_e_count));

    m_impl.finite_element_method->d_plastic_theta_d_hardening_k_warp.view().copy_from(
        m_impl.d_plastic_theta_d_hardening_k_warp.view(ortho_e_offset, ortho_e_count));
    m_impl.finite_element_method->d_plastic_theta_d_hardening_k_weft.view().copy_from(
        m_impl.d_plastic_theta_d_hardening_k_weft.view(ortho_e_offset, ortho_e_count));

    m_impl.finite_element_method->d_plastic_theta_d_k_warp.view().copy_from(
        m_impl.d_plastic_theta_d_k_warp.view(ortho_e_offset, ortho_e_count));
    m_impl.finite_element_method->d_plastic_theta_d_k_weft.view().copy_from(
        m_impl.d_plastic_theta_d_k_weft.view(ortho_e_offset, ortho_e_count));

    // replay (Face) plastic information
    auto& face_stack    = m_impl.face_offset_stack;
    auto  next_f_offset = face_stack.top();
    face_stack.pop();  // pop the last frame

    StackTopFrame = face_stack.size();

    UIPC_ASSERT(F == StackTopFrame,
                "Frame mismatch (Face), Input Frame={}, Stack Top Frame={}, why can it happen?",
                F,
                StackTopFrame);

    auto f_offset = face_stack.top();  // get offset

    auto f_count = geo->triangles().size();

    UIPC_ASSERT(f_offset + f_count == next_f_offset,
                "Face offset mismatch, e_offset={}, e_count={}, next_e_offset={}, why can it happen?",
                f_offset,
                f_count,
                next_f_offset);

    // copy the recorded data to the FEM
    m_impl.finite_element_method->strain_uus.view().copy_from(
        m_impl.strain_uus.view(f_offset, f_count));
    m_impl.finite_element_method->strain_vvs.view().copy_from(
        m_impl.strain_vvs.view(f_offset, f_count));
    m_impl.finite_element_method->strain_uvs.view().copy_from(
        m_impl.strain_uvs.view(f_offset, f_count));

    m_impl.finite_element_method->yield_strain_uus.view().copy_from(
        m_impl.yield_strain_uus.view(f_offset, f_count));
    m_impl.finite_element_method->yield_strain_vvs.view().copy_from(
        m_impl.yield_strain_vvs.view(f_offset, f_count));
    m_impl.finite_element_method->yield_strain_uvs.view().copy_from(
        m_impl.yield_strain_uvs.view(f_offset, f_count));

    m_impl.finite_element_method->hardening_plastic_strain_uus.view().copy_from(
        m_impl.hardening_plastic_strain_uus.view(f_offset, f_count));
    m_impl.finite_element_method->hardening_plastic_strain_vvs.view().copy_from(
        m_impl.hardening_plastic_strain_vvs.view(f_offset, f_count));
    m_impl.finite_element_method->hardening_plastic_strain_uvs.view().copy_from(
        m_impl.hardening_plastic_strain_uvs.view(f_offset, f_count));

    m_impl.finite_element_method->plastic_strain_uus.view().copy_from(
        m_impl.plastic_strain_uus.view(f_offset, f_count));
    m_impl.finite_element_method->plastic_strain_vvs.view().copy_from(
        m_impl.plastic_strain_vvs.view(f_offset, f_count));
    m_impl.finite_element_method->plastic_strain_uvs.view().copy_from(
        m_impl.plastic_strain_uvs.view(f_offset, f_count));

    m_impl.finite_element_method->anchor_strain_uus.view().copy_from(
        m_impl.anchor_strain_uus.view(f_offset, f_count));
    m_impl.finite_element_method->anchor_strain_vvs.view().copy_from(
        m_impl.anchor_strain_vvs.view(f_offset, f_count));
    m_impl.finite_element_method->anchor_strain_uvs.view().copy_from(
        m_impl.anchor_strain_uvs.view(f_offset, f_count));
    // For friction thres gradient
    m_impl.finite_element_method->d_anchor_uv_d_thres_uv.view().copy_from(
        m_impl.d_anchor_uv_d_thres_uv.view(f_offset, f_count));
    // For plastic yield gradient
    m_impl.finite_element_method->d_plastic_uv_d_yield_ori.view().copy_from(
        m_impl.d_plastic_uv_d_yield_ori.view(f_offset, f_count));
    m_impl.finite_element_method->d_plastic_uv_d_hardening_k33.view().copy_from(
        m_impl.d_plastic_uv_d_hardening_k33.view(f_offset, f_count));
    m_impl.finite_element_method->d_plastic_uv_d_k33.view().copy_from(
        m_impl.d_plastic_uv_d_k33.view(f_offset, f_count));
}

void FEMPlasticAdjointMethodReplayer::do_clear()
{
    //std::cout << "Clear Replayer" << std::endl;

    auto& edge_stack = m_impl.edge_offset_stack;
    m_impl.plastic_thetas.clear();
    m_impl.dihedral_thetas.clear();
    m_impl.anchor_thetas.clear();

    m_impl.d_anchor_theta_d_thres_theta.clear();

    m_impl.d_plastic_theta_d_yield_ori.clear();
    m_impl.d_plastic_theta_d_hardening_k.clear();
    m_impl.d_plastic_theta_d_k.clear();

    m_impl.plastic_thetas_warp.clear();
    m_impl.plastic_thetas_weft.clear();

    m_impl.anchor_thetas_warp.clear();
    m_impl.anchor_thetas_weft.clear();

    m_impl.d_anchor_theta_d_thres_theta_warp.clear();
    m_impl.d_anchor_theta_d_thres_theta_weft.clear();

    m_impl.d_plastic_theta_d_yield_ori_warp.clear();
    m_impl.d_plastic_theta_d_yield_ori_weft.clear();

    m_impl.d_plastic_theta_d_hardening_k_warp.clear();
    m_impl.d_plastic_theta_d_hardening_k_weft.clear();

    m_impl.d_plastic_theta_d_k_warp.clear();
    m_impl.d_plastic_theta_d_k_weft.clear();

    while(!edge_stack.empty())
        edge_stack.pop();
    edge_stack.push(0);  // dummy frame 1

    auto& ortho_edge_stack = m_impl.orthotropic_edge_offset_stack;
    while(!ortho_edge_stack.empty())
        ortho_edge_stack.pop();
    ortho_edge_stack.push(0);  // dummy frame 1

    auto& face_stack = m_impl.face_offset_stack;

    m_impl.strain_uus.clear();
    m_impl.strain_vvs.clear();
    m_impl.strain_uvs.clear();

    m_impl.anchor_strain_uus.clear();
    m_impl.anchor_strain_vvs.clear();
    m_impl.anchor_strain_uvs.clear();

    m_impl.yield_strain_uus.clear();
    m_impl.yield_strain_vvs.clear();
    m_impl.yield_strain_uvs.clear();

    m_impl.hardening_plastic_strain_uus.clear();
    m_impl.hardening_plastic_strain_vvs.clear();
    m_impl.hardening_plastic_strain_uvs.clear();

    m_impl.plastic_strain_uus.clear();
    m_impl.plastic_strain_vvs.clear();
    m_impl.plastic_strain_uvs.clear();

    m_impl.d_anchor_uv_d_thres_uv.clear();

    m_impl.d_plastic_uv_d_yield_ori.clear();
    m_impl.d_plastic_uv_d_hardening_k33.clear();
    m_impl.d_plastic_uv_d_k33.clear();

    while(!face_stack.empty())
        face_stack.pop();
    face_stack.push(0);  // dummy frame 1
}
}  // namespace uipc::backend::cuda
