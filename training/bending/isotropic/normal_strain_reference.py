
def initial_normal_strains_copy():
    with torch.no_grad():
        initial_positions = torch.from_numpy(mesh.positions().view().copy().reshape(-1, 3))
        initial_flat = initial_positions.reshape(-1)
        initial_edge1 = torch.mv(cloth_face_indices_1, initial_flat)
        initial_edge2 = torch.mv(cloth_face_indices_2, initial_flat)
        initial_ds = torch.stack([initial_edge1, initial_edge2], 1).view(NUM_FACES, 3, 2)
        initial_F = torch.bmm(initial_ds, cloth_faces_invDm)
        initial_G = .5 * (torch.bmm(initial_F.transpose(1, 2), initial_F)
                          - torch.eye(2, dtype=torch.float64).unsqueeze(0))
        return initial_G[:,0,0].detach().clone(), initial_G[:,1,1].detach().clone()

rest_strain_uus_tensor, rest_strain_vvs_tensor = initial_normal_strains_copy()
