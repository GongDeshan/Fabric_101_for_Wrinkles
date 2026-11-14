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

    nodes_uv = []

    with open(mesh_path, "r") as f:

        for line in f:

            line_tokens = line.strip().split(" ")

            if line_tokens[0] == "f":

                faces_vi.append([eval(nv.split("/")[1]) for nv in line_tokens[1:]])

            if line_tokens[0] == "vt":

                # Keep the uv lines

                uv = np.array([eval(line_tokens[1]), eval(line_tokens[2])])

                nodes_uv.append(uv)


    face_uv = np.stack([np.concatenate([nodes_uv[f_vi[0]-1], nodes_uv[f_vi[1]-1], nodes_uv[f_vi[2]-1]]) for f_vi in faces_vi])

    return face_uv
