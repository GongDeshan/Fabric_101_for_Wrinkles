import numpy as np


def load_mesh_uv(mesh_path):

    faces_vi = []

    nodes_uv = []

    with open(mesh_path, "r") as f:

        for line in f:

            line_tokens = line.split()
            if not line_tokens:
                continue

            if line_tokens[0] == "f":

                faces_vi.append([int(nv.split("/")[1]) for nv in line_tokens[1:]])

            if line_tokens[0] == "vt":

                # Keep the uv lines

                uv = np.array([float(line_tokens[1]), float(line_tokens[2])])

                nodes_uv.append(uv)


    face_uv = np.stack([np.concatenate([nodes_uv[f_vi[0]-1], nodes_uv[f_vi[1]-1], nodes_uv[f_vi[2]-1]]) for f_vi in faces_vi])

    return face_uv
