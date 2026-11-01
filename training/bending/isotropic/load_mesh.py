import numpy as np

def area_ratio(mesh_path):

    faces_vi = []

    nodes_uv = []

    nodes_x = []

    with open(mesh_path, "r") as f:

        for line in f:

            line_tokens = line.strip().split(" ")

            if line_tokens[0] == "f":

                faces_vi.append([eval(nv.split("/")[1]) for nv in line_tokens[1:]])

            if line_tokens[0] == "v":

                nodes_x.append(np.array([eval(line_tokens[1]), eval(line_tokens[2]), eval(line_tokens[3])]))

            if line_tokens[0] == "vt":

                uv = np.array(np.array([eval(line_tokens[1]), eval(line_tokens[2])]))

                nodes_uv.append(uv)

    f_n0_idx, f_n1_idx, f_n2_idx = faces_vi[0]

    face_area_w = np.linalg.norm(np.cross(nodes_x[f_n1_idx - 1] - nodes_x[f_n0_idx - 1], nodes_x[f_n2_idx - 1] - nodes_x[f_n0_idx - 1]))

    face_area_m = np.linalg.det(np.stack([nodes_uv[f_n1_idx - 1] - nodes_uv[f_n0_idx - 1], nodes_uv[f_n2_idx - 1] - nodes_uv[f_n0_idx - 1]], axis=1))

    return face_area_w/face_area_m


def load_mesh_uv(mesh_path):

    faces_vi = []
    faces_uvi = []

    nodes_uv = []

    node_pos_uv_index_dict = dict()

    with open(mesh_path, "r") as f:

        for line in f:

            line_tokens = line.strip().split(" ")

            if line_tokens[0] == "f":

                faces_vi.append([eval(nv.split("/")[0]) for nv in line_tokens[1:]])
                faces_uvi.append([eval(nv.split("/")[1]) for nv in line_tokens[1:]])

                this_face_uvi = faces_uvi[-1]
                this_face_ui = faces_vi[-1]

                for i_v, f_v in enumerate(this_face_ui):
                    if (f_v - 1) in node_pos_uv_index_dict.keys():
                        if (this_face_uvi[i_v]-1) not in node_pos_uv_index_dict[(f_v-1)]:
                            node_pos_uv_index_dict[(f_v-1)].append(this_face_uvi[i_v]-1)
                        else:
                            continue
                    else:
                        node_pos_uv_index_dict[(f_v-1)] = [this_face_uvi[i_v]-1]

            if line_tokens[0] == "vt":


                uv = np.array([eval(line_tokens[1]), eval(line_tokens[2])])

                nodes_uv.append(uv)


    face_uv = np.stack([np.concat([nodes_uv[faces_uvi[idx][0]-1], nodes_uv[faces_uvi[idx][1]-1], nodes_uv[faces_uvi[idx][2]-1]]) for idx in range(len(faces_vi))])



    return face_uv

def load_edge_uv(mesh_path):

    edges_vi = []
    edges_uvi = []

    nodes_uv = []

    nodes_pos = []

    node_pos_uv_index_dict = dict()

    with open(mesh_path, "r") as f:

        for line in f:

            line_tokens = line.strip().split(" ")

            f_idx = 0

            if line_tokens[0] == "f":

                f_idx += 0

                face_vs = [eval(nv.split("/")[0]) for nv in line_tokens[1:]]
                face_uvs = [eval(nv.split("/")[1]) for nv in line_tokens[1:]]

                for i_v, f_v in enumerate(face_vs):
                    if (f_v - 1) in node_pos_uv_index_dict.keys():
                        if (face_uvs[i_v]-1) not in node_pos_uv_index_dict[(f_v-1)]:
                            node_pos_uv_index_dict[(f_v-1)].append(face_uvs[i_v]-1)
                        else:
                            continue
                    else:
                        node_pos_uv_index_dict[(f_v-1)] = [face_uvs[i_v]-1]


                if {face_vs[0], face_vs[1]} not in edges_vi:
                    edges_vi.append({face_vs[0], face_vs[1]})
                    edges_uvi.append({face_uvs[0], face_uvs[1]})

                if {face_vs[1], face_vs[2]} not in edges_vi:
                    edges_vi.append({face_vs[1], face_vs[2]})
                    edges_uvi.append({face_uvs[1], face_uvs[2]})

                if {face_vs[2], face_vs[0]} not in edges_vi:
                    edges_vi.append({face_vs[2], face_vs[0]})
                    edges_uvi.append({face_uvs[2], face_uvs[0]})

            if line_tokens[0] == "vt":


                uv = np.array([eval(line_tokens[1]), eval(line_tokens[2])])

                nodes_uv.append(uv)

            if line_tokens[0] == "v":
                
                position = np.array([eval(line_tokens[1]), eval(line_tokens[2]), eval(line_tokens[3])])

                nodes_pos.append(position)


    edges_vi_list = [sorted(list(e_vi_set)) for e_vi_set in edges_vi]   

    edges_uvi_list = [sorted(list(e_uvi_set)) for e_uvi_set in edges_uvi]   

    sorted_edges_vi_list = sorted(edges_vi_list, key=lambda x: (x[0], x[1]))

    sorted_edges_uvi_list = sorted(edges_uvi_list, key=lambda x: (x[0], x[1]))

    edges_vi_topo = np.stack([np.stack([[e_vi_list[0]-1], [e_vi_list[1]-1]]) for e_vi_list in sorted_edges_vi_list])


    edge_uv = np.stack([np.stack([nodes_uv[node_pos_uv_index_dict[e_uvi[0]-1][0]], nodes_uv[node_pos_uv_index_dict[e_uvi[1]-1][0]]]) for e_uvi in sorted_edges_vi_list])

    return edges_vi_topo, edge_uv

def edge_bias_angle(mesh_path):

    edges_topo, edges_uv = load_edge_uv(mesh_path)

    angles_from_warp = []

    for e_uv in edges_uv:


        edge_vec = np.array(e_uv[1]) - np.array(e_uv[0])

        angle_from_x = np.atan2(abs(edge_vec[0]), abs(edge_vec[1]))
        angle_from_y = np.pi/2 - angle_from_x

        angles_from_warp.append(angle_from_y)
    
    return edges_topo, np.stack(angles_from_warp)

def bending_stiffness(k_warp, k_weft, bias_angles):
    
    bending_stiffnesses = []

    for b_ang in bias_angles:

        weight_k_warp = b_ang / 90
        weight_k_weft = 1.0 - weight_k_warp

        bending_stiffnesses.append(weight_k_warp * k_warp + weight_k_weft * k_weft)

    return bending_stiffnesses

if __name__ == "__main__":

    mesh_path = "D:\\Research_4\\libipc\\libuipc-samples\\assets\\sim_data\\trimesh\\LUFHES_uv.obj"


    edges_topo, angles_from_warp = edge_bias_angle(mesh_path)

    print(angles_from_warp.shape)


