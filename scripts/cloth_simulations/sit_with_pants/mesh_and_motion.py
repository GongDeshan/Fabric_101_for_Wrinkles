"""Linear-time OBJ/UV reader and frame-based motion for the shirt copy."""
from pathlib import Path
import numpy as np


def body_offset(frame, distance, start=50, sit_frames=50, hold_frames=20):
    bottom = start + sit_frames
    rise = bottom + hold_frames
    if frame <= start:
        return 0.0
    if frame <= bottom:
        return -distance * (frame-start) / sit_frames
    if frame <= rise:
        return -distance
    return -distance * max(0.0, 1.0-(frame-rise)/sit_frames)


def lift_fraction(frame, start=170, duration=50):
    return float(np.clip((frame-start)/duration, 0.0, 1.0))


def read_obj_uv(path):
    vertices, uvs, faces, face_uv = [], [], [], []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == 'vt':
                uvs.append([float(v) for v in parts[1:3]])
            elif parts[0] == 'f':
                if len(parts) != 4:
                    raise ValueError('Cloth OBJ must contain only triangular faces')
                vi, ti = [], []
                for token in parts[1:]:
                    indices = token.split('/')
                    if len(indices) < 2 or not indices[1]:
                        raise ValueError('Each cloth corner needs an explicit UV index')
                    v, t = int(indices[0]), int(indices[1])
                    vi.append(v-1 if v > 0 else len(vertices)+v)
                    ti.append(t-1 if t > 0 else len(uvs)+t)
                faces.append(vi)
                face_uv.append(ti)
    x, uv = np.asarray(vertices, dtype=float), np.asarray(uvs, dtype=float)
    f, ft = np.asarray(faces, dtype=np.int64), np.asarray(face_uv, dtype=np.int64)
    if not len(f) or not len(uv) or not np.isfinite(x).all() or not np.isfinite(uv).all():
        raise ValueError('Missing or nonfinite cloth geometry/UV')
    if f.min() < 0 or f.max() >= len(x) or ft.min() < 0 or ft.max() >= len(uv):
        raise ValueError('OBJ index out of bounds')
    corners = uv[ft]
    dm = np.stack((corners[:,1]-corners[:,0], corners[:,2]-corners[:,0]), axis=-1)
    if np.any(np.abs(np.linalg.det(dm)) < 1e-14):
        raise ValueError('Degenerate UV triangle')
    ds = np.stack((x[f[:,1]]-x[f[:,0]], x[f[:,2]]-x[f[:,0]]), axis=-1)
    deformation = ds @ np.linalg.inv(dm)
    green = .5*(np.swapaxes(deformation,1,2) @ deformation-np.eye(2))
    edges, inconsistent = {}, set()
    for face, triangle in zip(f, corners):
        for a,b in ((0,1),(1,2),(2,0)):
            key = tuple(sorted((int(face[a]), int(face[b]))))
            d = triangle[b]-triangle[a]
            angle = float(np.arctan2(abs(d[1]), abs(d[0])))
            if key in edges and abs(edges[key]-angle) > 1e-5:
                inconsistent.add(key)
            edges.setdefault(key, angle)
    audit = dict(vertices=len(x), faces=len(f), edges=len(edges),
                 uv_bounds=[uv.min(axis=0).tolist(),uv.max(axis=0).tolist()],
                 initial_green_abs_quantiles=np.quantile(np.abs(green),[.5,.95,1]).tolist(),
                 shared_edges_with_uv_direction_disagreement=len(inconsistent))
    return x,f,corners,edges,audit


def align_face_uv(obj_faces, obj_corners, mesh_faces):
    mesh_faces = np.asarray(mesh_faces).reshape(-1,3)
    if np.array_equal(obj_faces, mesh_faces):
        return obj_corners
    lookup = {tuple(sorted(map(int,f))):i for i,f in enumerate(obj_faces)}
    if len(lookup) != len(obj_faces):
        raise ValueError('Duplicate triangle topology prevents unambiguous UV mapping')
    result = []
    for f in mesh_faces:
        i = lookup[tuple(sorted(map(int,f)))]
        positions = {int(v):j for j,v in enumerate(obj_faces[i])}
        result.append(obj_corners[i,[positions[int(v)] for v in f]])
    return np.asarray(result)
