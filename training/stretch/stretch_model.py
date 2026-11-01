import uipc
import numpy as np
import polyscope as ps
from polyscope import imgui
import torch
from torch import optim

from uipc import view
from uipc import Logger, Timer
from uipc import Transform
from uipc import builtin
from uipc.core import Animation, Engine, Scene, World
from uipc.geometry import AttributeSlotI32, GeometrySlot, SimplicialComplex, SimplicialComplexIO, label_surface
from uipc.constitution import SoftPositionConstraint
from uipc.constitution import DiscreteShellFrictionHardeningPlasticBending, StVKNonlinearFrictionHardeningPlasticShell
from uipc.unit import kPa
from uipc.gui import GUIInfo, SceneGUI
from diff_sim_friction_plastic_gated_module import DiffSimLR, DiffSimModule, DiffSimParameter

from load_mesh import load_mesh_uv

from asset_dir import AssetDir

if __name__ == '__main__' and not globals().get('_stretch_embedded_scene', False):
    from stretch_data_controls import run_scene_window
    run_scene_window(__file__)
    raise SystemExit(0)

run_type = "init"

tested_fabric_type = "59_Cotton"

fabric_type = tested_fabric_type.split("_")[-1].lower()

sample_vertial_direction = "warp"
sample_vertial_direction = "weft"

warp_stiffness = 2000.0 * kPa
weft_stiffness = 2000.0 * kPa

K11 = weft_stiffness  if sample_vertial_direction == "warp" else warp_stiffness 
K22 = warp_stiffness  if sample_vertial_direction == "warp" else weft_stiffness 

K12 = 0.0
K33_0 = 100.0 * kPa
K33_1 = 100.0 * kPa

Hardening_K33 = 1.0
Fric_K33 = 100.0 * kPa

Thres_uv = 1.0
Yield_UV = 1.0

MASS_DENSITY = 430
THICKNESS = 0.0004

BENDING_STIFFNESS = 400.0
Hardening_K = 100.0
YEILD_THETA = 3.14
PLASTIC_THETA = 0.0

BENDING_FRICTION = 400.0
ANCHOR_THETA = 0.0
THRES_THETA = 3.14

EndFrame = 30

MaxPos = 0.01

PARAM_LR = 100.0
NUM_EPOCH = 500

MESH_PATH = str(__import__('pathlib').Path(__file__).resolve().with_name('mesh.obj'))

GT_PATH = str(__import__('pathlib').Path(__file__).resolve().parent/'data'/f'{tested_fabric_type}_{sample_vertial_direction}_friction.npy')

with open(GT_PATH, "rb") as f:
    real_load_pos_forces = np.load(f)
    MaxPos = np.max(real_load_pos_forces[0])/1000
    forces_gt = torch.from_numpy(real_load_pos_forces[1])

pass  

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
this_folder = AssetDir.folder(__file__)
tetmesh_path = AssetDir.tetmesh_path()
trimesh_path = AssetDir.trimesh_path()

engine = Engine('cuda', workspace)
world = World(engine)

dt = 0.01
config = Scene.default_config()
config["dt"] = dt
config['contact']['d_hat'] = THICKNESS
config['gravity'] = [[0.0], [0.0], [0.0]]
config['contact']['enable'] = False

pass  
scene = Scene(config)

cloth_obj = scene.objects().create('cloth')
nlstckfhps = StVKNonlinearFrictionHardeningPlasticShell()
dsfhpb = DiscreteShellFrictionHardeningPlasticBending()

t = Transform.Identity()
t.rotate(uipc.AngleAxis(1.57, [1.0, 0.0, 0.0]))
t.scale(1.0)
io = SimplicialComplexIO(t)
mesh = io.read(MESH_PATH)
label_surface(mesh)

cloth_mesh_vertices = mesh.vertices()
cloth_mesh_pos_v = cloth_mesh_vertices.find(builtin.position)
pos_v_const_view = cloth_mesh_pos_v.view()
cloth_mesh_center = pos_v_const_view.mean(axis=0)

cloth_mesh_center[0] = 0.0  
cloth_mesh_center[1] = -cloth_mesh_center[1] * 2.0

NUM_VERTICES = mesh.vertices().size()
NUM_EDGES = mesh.edges().size()
NUM_FACES = mesh.triangles().size()

view(mesh.positions())[:] += cloth_mesh_center

nlstckfhps.apply_to(mesh, K11, K12, K22, K33_0, K33_1, hardening_k33=Hardening_K33, friction_k33=Fric_K33,
                  thres_uv=Thres_uv, anchor_strain_uv=0.0, yield_uv=Yield_UV, plastic_strain_uv=0.0,
                  mass_density=MASS_DENSITY, thickness=THICKNESS)

dsfhpb.apply_to(mesh, bending_stiffness=BENDING_STIFFNESS, hardening_k=Hardening_K, yield_theta=YEILD_THETA, 
                plastic_theta=PLASTIC_THETA, friction_k=BENDING_FRICTION, thres_theta=THRES_THETA, anchor_theta=ANCHOR_THETA)

spc = SoftPositionConstraint()
spc.apply_to(mesh, 2000.0) # constraint strength ratio

is_fixed = mesh.vertices().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
pos_view = mesh.positions().view()
is_fixed_view[:] = np.where(pos_view[:, 1].reshape(-1) < 0.02, 1, 0)

rest_yield_ori_uvs = mesh.triangles().find("yield_uv").view().copy()
rest_yield_ori_tensor = torch.from_numpy(rest_yield_ori_uvs)

rest_yield_theta_ori = mesh.edges().find("yield_theta").view().copy()
rest_yield_theta_ori_tensor = torch.from_numpy(rest_yield_theta_ori)

diff_k22: AttributeSlotI32 = mesh.triangles().create('diff/k22', -1)
view(diff_k22)[:] = 0  

pass  
geo_slot, _ = cloth_obj.geometries().create(mesh)

comp_d = 0.0
compress_dis = MaxPos / 30

def LUFHES_compress_cloth(info:Animation.UpdateInfo):
    global comp_d

    if info.frame() == 1:
        comp_d = 0.0

    geo_slots:list[GeometrySlot] = info.geo_slots()
    geo:SimplicialComplex = geo_slots[0].geometry()
    rest_geo_slots:list[GeometrySlot] = info.rest_geo_slots()
    rest_geo:SimplicialComplex = rest_geo_slots[0].geometry()

    is_constrained = geo.vertices().find(builtin.is_constrained)
    is_constrained_view = view(is_constrained)
    aim_position = geo.vertices().find(builtin.aim_position)
    aim_position_view = view(aim_position)

    rest_position_view = rest_geo.positions().view()

    for i in range(len(is_constrained_view)):
        if rest_position_view[i][1] > 0.07:
            is_constrained_view[i] = 1

    comp_d += compress_dis

    for i in range(aim_position_view.shape[0]):
        if rest_position_view[i][1] > 0.07:
            aim_position_view[i] = rest_position_view[i] + np.array([0.0, comp_d, 0.0])[:, np.newaxis]

scene.animator().insert(cloth_obj, LUFHES_compress_cloth)


edge_topos = geo_slot.geometry().edges().find(builtin.topo).view()
face_topos = geo_slot.geometry().triangles().find(builtin.topo).view()

x_bar = load_mesh_uv(MESH_PATH)[:,:, np.newaxis]
view(geo_slot.geometry().triangles().find("rest_x"))[:] = x_bar

class MyDiffSimParameter(DiffSimParameter):
    def forward(self) -> torch.Tensor:
        return self.U()**2

parm = MyDiffSimParameter(scene=scene, size=1)

world.init(scene)

rest_plastic_thetas = geo_slot.geometry().edges().find("plastic_theta").view().copy()
rest_anchor_thetas = geo_slot.geometry().edges().find("anchor_theta").view().copy()

rest_anchor_uvs = geo_slot.geometry().triangles().find("anchor_strain_uv").view().copy()
rest_plastic_uvs = geo_slot.geometry().triangles().find("plastic_strain_uv").view().copy()

node_force_indices = []

egde_rest_lengths = geo_slot.geometry().edges().find("L0").view()
V_bars = geo_slot.geometry().edges().find("V_bar").view()
h_bars = geo_slot.geometry().edges().find("h_bar").view()
bending_edge_stencils = geo_slot.geometry().edges().find("stencil").view()

is_edge_cal_force = np.zeros(mesh.edges().size(), dtype=np.bool)
is_face_cal_force = np.zeros(mesh.triangles().size(), dtype=np.bool)

x_init = mesh.vertices().find(builtin.position).view().copy()

for i, node_x in enumerate(x_init):
    if node_x[1] < 0.02:
        node_force_indices.append(i)
        is_edge_cal_force = is_edge_cal_force | np.any(
            np.isin(bending_edge_stencils, i), axis=1).squeeze()       
        is_face_cal_force = is_face_cal_force | np.any(
            np.isin(face_topos, i), axis=1).squeeze()

rest_areas = []

tri_values = torch.tensor([1, 1, 1, -1, -1, -1], dtype=torch.int8)

tris_indices_1 = []
tris_indices_2 = []

faces_invDms = []

Dus = []
Dvs = []

stretch_f_tensors = []
stretch_J_tensors = []

for f_i, f_vs in enumerate(face_topos):

    face_v_indices = f_vs.squeeze()

    n0_idx = face_v_indices[0]
    n1_idx = face_v_indices[1]
    n2_idx = face_v_indices[2]

    tri_indices_row = torch.arange(3, dtype=torch.long).repeat(2)
    tri_1_indices_col = torch.tensor([n1_idx * 3, n0_idx * 3], dtype=torch.long).repeat_interleave(3) + torch.arange(3, dtype=torch.long).repeat(2)
    tri_2_indices_col = torch.tensor([n2_idx * 3, n0_idx * 3], dtype=torch.long).repeat_interleave(3) + torch.arange(3, dtype=torch.long).repeat(2)

    tri_1_spa_indices = torch.stack([tri_indices_row, tri_1_indices_col], dim=0)
    tri_2_spa_indices = torch.stack([tri_indices_row, tri_2_indices_col], dim=0)

    tri_1_indices = torch.sparse_coo_tensor(tri_1_spa_indices, tri_values, (3, NUM_VERTICES * 3), dtype=torch.int8)
    tri_2_indices = torch.sparse_coo_tensor(tri_2_spa_indices, tri_values, (3, NUM_VERTICES * 3), dtype=torch.int8)

    tris_indices_1.append(tri_1_indices)
    tris_indices_2.append(tri_2_indices)

    xs_tri_rest = x_bar[f_i].squeeze()

    face_dm_m = np.stack([xs_tri_rest[2:4] - xs_tri_rest[:2], xs_tri_rest[4:6] - xs_tri_rest[:2]], axis=1)

    dm_inv = np.linalg.inv(face_dm_m)

    faces_invDms.append(dm_inv)

    d = dm_inv.flatten()

    Du = torch.cat([(-d[0] - d[2]) * torch.eye(3, dtype=torch.double), d[0] * torch.eye(3, dtype=torch.double), d[2] * torch.eye(3, dtype=torch.double)], dim=1)
    Dv = torch.cat([(-d[1] - d[3]) * torch.eye(3, dtype=torch.double), d[1] * torch.eye(3, dtype=torch.double), d[3] * torch.eye(3, dtype=torch.double)], dim=1)

    Dus.append(Du)
    Dvs.append(Dv)

    stretch_f_tensor = torch.zeros((NUM_VERTICES * 3, 9), dtype=torch.int8)

    stretch_f_tensor[n0_idx * 3:n0_idx * 3 + 3, 0:3] = torch.eye(3, dtype=torch.int8)
    stretch_f_tensor[n1_idx * 3:n1_idx * 3 + 3, 3:6] = torch.eye(3, dtype=torch.int8)
    stretch_f_tensor[n2_idx * 3:n2_idx * 3 + 3, 6:9] = torch.eye(3, dtype=torch.int8)

    d_n0_x_d_n0 = n0_idx * (NUM_VERTICES * 3 * 3) + n0_idx * 3
    d_n0_x_d_n1 = n0_idx * (NUM_VERTICES * 3 * 3) + n1_idx * 3
    d_n0_x_d_n2 = n0_idx * (NUM_VERTICES * 3 * 3) + n2_idx * 3

    d_n0_y_d_n0 = n0_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n0_idx * 3
    d_n0_y_d_n1 = n0_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n1_idx * 3
    d_n0_y_d_n2 = n0_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n2_idx * 3

    d_n0_z_d_n0 = n0_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n0_idx * 3
    d_n0_z_d_n1 = n0_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n1_idx * 3
    d_n0_z_d_n2 = n0_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n2_idx * 3

    d_n1_x_d_n0 = n1_idx * (NUM_VERTICES * 3 * 3) + n0_idx * 3
    d_n1_x_d_n1 = n1_idx * (NUM_VERTICES * 3 * 3) + n1_idx * 3
    d_n1_x_d_n2 = n1_idx * (NUM_VERTICES * 3 * 3) + n2_idx * 3

    d_n1_y_d_n0 = n1_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n0_idx * 3
    d_n1_y_d_n1 = n1_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n1_idx * 3
    d_n1_y_d_n2 = n1_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n2_idx * 3

    d_n1_z_d_n0 = n1_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n0_idx * 3
    d_n1_z_d_n1 = n1_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n1_idx * 3
    d_n1_z_d_n2 = n1_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n2_idx * 3

    d_n2_x_d_n0 = n2_idx * (NUM_VERTICES * 3 * 3) + n0_idx * 3
    d_n2_x_d_n1 = n2_idx * (NUM_VERTICES * 3 * 3) + n1_idx * 3
    d_n2_x_d_n2 = n2_idx * (NUM_VERTICES * 3 * 3) + n2_idx * 3

    d_n2_y_d_n0 = n2_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n0_idx * 3
    d_n2_y_d_n1 = n2_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n1_idx * 3
    d_n2_y_d_n2 = n2_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) + n2_idx * 3

    d_n2_z_d_n0 = n2_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n0_idx * 3
    d_n2_z_d_n1 = n2_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n1_idx * 3
    d_n2_z_d_n2 = n2_idx * (NUM_VERTICES * 3 * 3) + (NUM_VERTICES * 3) * 2 + n2_idx * 3

    row_index = torch.tensor([d_n0_x_d_n0, d_n0_x_d_n1, d_n0_x_d_n2, d_n0_y_d_n0, d_n0_y_d_n1, d_n0_y_d_n2, d_n0_z_d_n0, d_n0_z_d_n1, d_n0_z_d_n2,
                              d_n1_x_d_n0, d_n1_x_d_n1, d_n1_x_d_n2, d_n1_y_d_n0, d_n1_y_d_n1, d_n1_y_d_n2, d_n1_z_d_n0, d_n1_z_d_n1, d_n1_z_d_n2,
                              d_n2_x_d_n0, d_n2_x_d_n1, d_n2_x_d_n2, d_n2_y_d_n0, d_n2_y_d_n1, d_n2_y_d_n2, d_n2_z_d_n0, d_n2_z_d_n1, d_n2_z_d_n2], dtype=torch.long)

    row_index = row_index.repeat_interleave(3) + torch.tensor([0, 1, 2], dtype=torch.int).repeat(27)

    col_index = torch.arange(81, dtype=torch.long)

    indices = torch.stack([row_index, col_index], dim=0)

    values = torch.ones(81, dtype=torch.int8)

    stretch_J_tensor = torch.sparse_coo_tensor(indices, values, (NUM_VERTICES * 3 * NUM_VERTICES * 3, 3 * 3 * 3 * 3), dtype=torch.int8)

    stretch_f_tensors.append(stretch_f_tensor.to_sparse())
    stretch_J_tensors.append(stretch_J_tensor)

    x0_init, x1_init, x2_init = [x_init[i].squeeze() for i in f_vs.squeeze()]

    rest_areas.append(0.5 * np.linalg.norm(np.cross(x1_init - x0_init, x2_init - x0_init)))

cloth_face_indices_1 = torch.cat(tris_indices_1, dim=0).to(dtype=torch.double)
cloth_face_indices_2 = torch.cat(tris_indices_2, dim=0).to(dtype=torch.double)

cloth_faces_invDm = torch.from_numpy(np.stack(faces_invDms, axis=0))

cloth_faces_m = torch.from_numpy(np.stack(rest_areas))

cloth_Du = torch.stack(Dus)
cloth_Dv = torch.stack(Dvs)

cloth_SF_T = torch.cat(stretch_f_tensors, dim=1).to(dtype=torch.double)
cloth_SJ_T = torch.cat(stretch_J_tensors, dim=1).to(dtype=torch.double)

edge_cal_force_indices = np.argwhere(is_edge_cal_force)
face_cal_force_indices = np.argwhere(is_face_cal_force)

edges_node_indices = []

e0_indices = []
e1_indices = []
e2_indices = []
e3_indices = []
e4_indices = []

e0_n_indices = []
e2_n_indices = []

valid_edge_index = []
BF_index = []

for e_idx, l0 in enumerate(egde_rest_lengths):

    if l0.item() != 0.0 and np.any(np.isin(edge_cal_force_indices, e_idx)):

        valid_idx_tensor = torch.zeros(egde_rest_lengths.shape, dtype=torch.double)

        one_BF_index_tensor = torch.zeros((NUM_VERTICES * 3, 12), dtype=torch.double)

        valid_idx_tensor[e_idx] = 1.0

        valid_edge_index.append(valid_idx_tensor)

        one_idx_tensor = torch.zeros((4, NUM_VERTICES), dtype=torch.double)

        for i in range(4):
            one_idx_tensor[i][bending_edge_stencils[e_idx][i]] = 1.0

            for j in range(3):
                one_BF_index_tensor[bending_edge_stencils[e_idx][i] * 3 + j, i * 3 + j] = 1.0

        BF_index.append(one_BF_index_tensor.to_sparse())

        edges_node_indices.append(one_idx_tensor)

        one_e0_idx_tensor = torch.zeros(
            (1, NUM_VERTICES), dtype=torch.double)
        one_e1_idx_tensor = torch.zeros(
            (1, NUM_VERTICES), dtype=torch.double)
        one_e2_idx_tensor = torch.zeros(
            (1, NUM_VERTICES), dtype=torch.double)
        one_e3_idx_tensor = torch.zeros(
            (1, NUM_VERTICES), dtype=torch.double)
        one_e4_idx_tensor = torch.zeros(
            (1, NUM_VERTICES), dtype=torch.double)

        one_e0_idx_tensor[0][bending_edge_stencils[e_idx][2]] = 1.0
        one_e0_idx_tensor[0][bending_edge_stencils[e_idx][1]] = -1.0

        one_e1_idx_tensor[0][bending_edge_stencils[e_idx][0]] = 1.0
        one_e1_idx_tensor[0][bending_edge_stencils[e_idx][1]] = -1.0

        one_e2_idx_tensor[0][bending_edge_stencils[e_idx][3]] = 1.0
        one_e2_idx_tensor[0][bending_edge_stencils[e_idx][1]] = -1.0

        one_e3_idx_tensor[0][bending_edge_stencils[e_idx][0]] = 1.0
        one_e3_idx_tensor[0][bending_edge_stencils[e_idx][2]] = -1.0

        one_e4_idx_tensor[0][bending_edge_stencils[e_idx][3]] = 1.0
        one_e4_idx_tensor[0][bending_edge_stencils[e_idx][2]] = -1.0

        one_e0_n_idx_tensor = torch.zeros(
            (1, NUM_VERTICES), dtype=torch.double)
        one_e2_n_idx_tensor = torch.zeros(
            (1, NUM_VERTICES), dtype=torch.double)

        one_e0_n_idx_tensor[0][bending_edge_stencils[e_idx][2]] = 1.0
        one_e0_n_idx_tensor[0][bending_edge_stencils[e_idx][3]] = -1.0

        one_e2_n_idx_tensor[0][bending_edge_stencils[e_idx][1]] = 1.0
        one_e2_n_idx_tensor[0][bending_edge_stencils[e_idx][3]] = -1.0

        e0_indices.append(one_e0_idx_tensor)
        e1_indices.append(one_e1_idx_tensor)
        e2_indices.append(one_e2_idx_tensor)
        e3_indices.append(one_e3_idx_tensor)
        e4_indices.append(one_e4_idx_tensor)

        e0_n_indices.append(one_e0_n_idx_tensor)
        e2_n_indices.append(one_e2_n_idx_tensor)

cloth_edges_node_indices = torch.stack(edges_node_indices)

cloth_e0_indices = torch.stack(e0_indices)
cloth_e1_indices = torch.stack(e1_indices)
cloth_e2_indices = torch.stack(e2_indices)
cloth_e3_indices = torch.stack(e3_indices)
cloth_e4_indices = torch.stack(e4_indices)

cloth_e0_n_indices = torch.stack(e0_n_indices)
cloth_e2_n_indices = torch.stack(e2_n_indices)

cloth_valid_indices = torch.stack(valid_edge_index)
cloth_BF_index = torch.stack(BF_index)

cloth_l0s = torch.mv(cloth_valid_indices, torch.from_numpy(egde_rest_lengths.copy()))
cloth_v_bars = torch.mv(cloth_valid_indices, torch.from_numpy(V_bars.copy()))
cloth_h_bars = torch.mv(cloth_valid_indices, torch.from_numpy(h_bars.copy()))

cloth_select_force_indices = torch.zeros((len(node_force_indices) * 3, NUM_VERTICES * 3), dtype=torch.double)

for i, node_f_idx in enumerate(node_force_indices):
    for j in range(3):
        if j == 1:
            cloth_select_force_indices[i*3+j][node_f_idx*3 + j] = 1.0

def hinge_theta_gradient(x0, x1, x2, x3):
    e0, e1, e2 = x2-x1, x0-x1, x3-x1
    e3, e4 = x0-x2, x3-x2
    n1 = torch.linalg.cross(e0, e1, dim=-1)
    n2 = torch.linalg.cross(e2, e0, dim=-1)
    length = torch.linalg.vector_norm(e0, dim=-1)
    sq1, sq2 = n1.square().sum(-1), n2.square().sum(-1)
    if bool(((length == 0) | (sq1 == 0) | (sq2 == 0)).any()):
        raise ValueError('Degenerate bending hinge')
    e5, e6 = x2-x3, x1-x3
    n1s_theta = torch.linalg.cross(e0, e1, dim=-1)
    n2s_theta = torch.linalg.cross(e5, e6, dim=-1)
    n1_dot_n2 = (n1s_theta * n2s_theta).sum(dim=-1)
    edge_hat = -e0 / length.unsqueeze(-1)
    sin_term = (torch.linalg.cross(n2s_theta, n1s_theta, dim=-1) * edge_hat).sum(dim=-1)
    theta = torch.atan2(sin_term, n1_dot_n2)
    dot = lambda a,b: (a*b).sum(-1)
    g0 = (-length/sq1).unsqueeze(-1)*n1
    g1 = (-dot(e0,e3)/(length*sq1)).unsqueeze(-1)*n1 - (dot(e0,e4)/(length*sq2)).unsqueeze(-1)*n2
    g2 = (dot(e0,e1)/(length*sq1)).unsqueeze(-1)*n1 + (dot(e0,e2)/(length*sq2)).unsqueeze(-1)*n2
    g3 = (-length/sq2).unsqueeze(-1)*n2
    return theta, torch.cat([g0,g1,g2,g3],dim=-1)

def dihedral_angle(x0, x1, x2, x3):
    return hinge_theta_gradient(x0,x1,x2,x3)

def stretch_strain_init(x_init_np):
    x=torch.from_numpy(x_init_np.copy()).reshape(-1)
    ds=torch.stack([torch.mv(cloth_face_indices_1,x),torch.mv(cloth_face_indices_2,x)],1).view(NUM_FACES,3,2)
    F=torch.bmm(ds,cloth_faces_invDm)
    G=.5*(torch.bmm(F.transpose(1,2),F)-torch.eye(2,dtype=torch.float64).unsqueeze(0))
    return G[:,0,0],G[:,1,1],G[:,0,1]

strain_uus_init, strain_vvs_init, strain_uvs_init = stretch_strain_init(x_init)

def net_force(xs_torch, bending_k, friction_k, hardening_k, yield_theta_ori, yield_thetas, hardening_plastic_thetas, plastic_thetas, thres_thetas, anchor_thetas,
              K11, K22, K12, K33_0, K33_1, Friction_K33, thres_uvs, anchor_strain_uvs, hardening_k33, yield_uv_ori, yield_uvs, hardening_plastic_uvs, plastic_strain_uvs):
    hinge=torch.matmul(cloth_edges_node_indices,xs_torch)
    theta,gradient=hinge_theta_gradient(hinge[:,0],hinge[:,1],hinge[:,2],hinge[:,3])
    plastic=torch.mv(cloth_valid_indices,plastic_thetas)
    anchor=torch.mv(cloth_valid_indices,anchor_thetas)
    coefficient=-2*cloth_l0s*cloth_v_bars/cloth_h_bars
    hinge_force=(coefficient*(bending_k*(theta-plastic)+friction_k*(theta-anchor))).unsqueeze(1)*gradient
    bending_force=torch.bmm(cloth_BF_index,hinge_force.unsqueeze(2)).sum(0).reshape(-1)
    flat=xs_torch.reshape(-1)
    ds=torch.stack([torch.mv(cloth_face_indices_1,flat),torch.mv(cloth_face_indices_2,flat)],1).view(NUM_FACES,3,2)
    F=torch.bmm(ds,cloth_faces_invDm)
    G=.5*(torch.bmm(F.transpose(1,2),F)-torch.eye(2,dtype=torch.float64).unsqueeze(0))
    xu,xv=F[:,:,0].unsqueeze(2),F[:,:,1].unsqueeze(2)
    Du,Dv=cloth_Du.transpose(1,2),cloth_Dv.transpose(1,2)
    du=torch.bmm(Du,xu).squeeze(-1)
    dv=torch.bmm(Dv,xv).squeeze(-1)
    dq=.5*(torch.bmm(Du,xv)+torch.bmm(Dv,xu)).squeeze(-1)
    eu,ev=G[:,0,0]-strain_uus_init,G[:,1,1]-strain_vvs_init
    q=G[:,0,1]
    z=q-plastic_strain_uvs
    tau=(K33_0+K33_1*torch.abs(q))*z+.5*K33_1*torch.sign(q)*z.square()
    stress=(K11*eu+K12*ev).unsqueeze(1)*du+(K22*ev+K12*eu).unsqueeze(1)*dv
    stress=stress+(tau+Friction_K33*(q-anchor_strain_uvs)).unsqueeze(1)*dq
    stretching_force=THICKNESS*torch.mv(cloth_SF_T,(-cloth_faces_m.unsqueeze(1)*stress).reshape(-1))
    return torch.mv(cloth_select_force_indices,bending_force+stretching_force).sum()

def dof_select(info: DiffSimModule.DofSelectInfo):
    geo = geo_slot.geometry()
    offset = geo.meta().find(builtin.dof_offset).view()[0]
    verts_idx = offset + np.arange(NUM_VERTICES * 3, dtype=np.int32)
    sdi = torch.from_numpy(verts_idx)
    return sdi

with torch.no_grad():
    u = np.sqrt(K22)
    parm.U().fill_(u)
    parm.sync()

diff_module = DiffSimModule(world, geo_slot=geo_slot, end_frame=EndFrame, parm=parm, dof_select=dof_select)
optimizer = optim.Adam(diff_module.parameters(), lr=PARAM_LR)
lr = DiffSimLR(diff_module, optimizer)

losses_record = []

def loss_fn() -> torch.Tensor:

    optimizer.zero_grad()

    X = diff_module()

    plastic_thetas = torch.from_numpy(np.stack(diff_module.get_plastic_thetas()))
    anchor_thetas = torch.from_numpy(np.stack(diff_module.get_anchor_thetas()))
    yield_thetas = torch.from_numpy(np.stack(diff_module.get_yield_thetas()))
    hardening_thetas = torch.from_numpy(np.stack(diff_module.get_hardening_plastic_thetas()))

    thres_thetas_tensor = torch.from_numpy(geo_slot.geometry().edges().find("thres_theta").view().copy())

    plastic_uvs = torch.from_numpy(np.stack(diff_module.get_plastic_strains()))
    yield_uvs = torch.from_numpy(np.stack(diff_module.get_yield_strains()))
    hardening_uvs = torch.from_numpy(np.stack(diff_module.get_hardening_strains()))
    anchor_uvs = torch.from_numpy(np.stack(diff_module.get_anchor_strains()))

    thres_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("thres_uv").view().copy())

    parm_k22_tensor = optimizer.param_groups[0]["params"][0]**2

    hanlde_forces_strength = [torch.tensor(0.0)]

    for f in range(EndFrame):
        X_pred = X[f * NUM_VERTICES * 3: (f+1) * NUM_VERTICES * 3].reshape(-1, 3)

        plastic_thetas_a_frame = plastic_thetas[f]
        yield_theta_a_frame = yield_thetas[f]
        hardening_theta_a_frame = hardening_thetas[f]
        anchor_theta_a_frame = anchor_thetas[f]

        plastic_uvs_a_frame = plastic_uvs[f]
        yield_uvs_a_frame = yield_uvs[f]
        hardening_uvs_a_frame = hardening_uvs[f]
        anchor_uvs_a_frame = anchor_uvs[f]

        force = net_force(X_pred, BENDING_STIFFNESS, BENDING_FRICTION, Hardening_K, rest_yield_theta_ori_tensor, yield_theta_a_frame, hardening_theta_a_frame, plastic_thetas_a_frame, 
                          thres_thetas_tensor, anchor_theta_a_frame, K11, parm_k22_tensor, K12, K33_0, K33_1, Fric_K33, thres_uvs_tensor, anchor_uvs_a_frame, Hardening_K33, 
                          rest_yield_ori_tensor, yield_uvs_a_frame, hardening_uvs_a_frame, plastic_uvs_a_frame)
        
        hanlde_forces_strength.append(force.sum())

    forces_pred = torch.stack(hanlde_forces_strength)
    globals()['_stretch_last_prediction'] = forces_pred.detach().cpu().numpy().copy()

    pass  
    pass  

    loss = (forces_pred - forces_gt).square().mean()

    pass  

    losses_record.append(loss.clone().cpu().detach().item())

    loss.backward()

    pass  

    return loss

def optimize_once():
    for e in range(NUM_EPOCH):
        pass  
        optimizer.step(loss_fn)

    pass  

handle_force_scalars = []
comp_ds = []

def recover():
    global handle_force_scalars, comp_ds
    handle_force_scalars.clear();comp_ds.clear()
    handle_force_scalars.append(0.0);comp_ds.append(0.0)
    if not world.recover(0):
        raise RuntimeError('Frame-0 recovery failed')
    view(geo_slot.geometry().edges().find('reset_anchor_plastic'))[:]=1
    view(geo_slot.geometry().edges().find('plastic_theta'))[:]=rest_plastic_thetas
    view(geo_slot.geometry().edges().find('anchor_theta'))[:]=rest_anchor_thetas
    view(geo_slot.geometry().edges().find('yield_theta'))[:]=rest_yield_theta_ori
    view(geo_slot.geometry().triangles().find('reset_anchor_plastic'))[:]=1
    view(geo_slot.geometry().triangles().find('anchor_strain_uv'))[:]=rest_anchor_uvs
    view(geo_slot.geometry().triangles().find('plastic_strain_uv'))[:]=rest_plastic_uvs
    view(geo_slot.geometry().triangles().find('yield_uv'))[:]=rest_yield_ori_uvs
    parm.sync()

world.dump()

def save_mesh_sequence(frame: int):

    io = SimplicialComplexIO()

    io.write(
        f"{output_dir}/cloth_{frame:04}.obj",
        geo_slot.geometry(),
    )

if GUIInfo.enabled():
    if not globals().get('_stretch_polyscope_initialized', False):
        ps.init()
    ps.set_ground_plane_height_mode('manual')
    ps.set_ground_plane_height(-0.3)
    sgui = SceneGUI(scene)
    sgui.register()
    run = False
    from stretch_data_controls import StretchDataControls
    training_data_controls = StretchDataControls(globals())
    from stretch_training_progress import StretchTrainingProgress
    training_progress = StretchTrainingProgress(globals())

    def update():
        global comp_d

        global run
        if training_data_controls.draw():
            return
        training_progress.draw()
        imgui.Text(f'frame: {world.frame()}')

        if (imgui.Button('Run & Stop')):
            training_progress.running = False
            training_progress.message = 'Forward simulation'
            run = not run
            handle_force_scalars.clear()
            if run:
                recover()

        if (imgui.Button('optim')):
            training_progress.start()

        if training_progress.running:
            training_progress.step()
            return

        if run:
            if (world.frame() < EndFrame):

                pass  

                world.advance()
                world.retrieve()


                if world.frame() == 1:
                    view(geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 0
                    view(geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 0

                xs_numpy = np.stack([xs_np.squeeze() for xs_np in geo_slot.geometry().vertices().find(builtin.position).view()])
                xs_tensor = torch.from_numpy(xs_numpy)

                plastic_thetas_tensor = torch.from_numpy(geo_slot.geometry().edges().find("plastic_theta").view().copy())
                yield_thetas_tensor = torch.from_numpy(geo_slot.geometry().edges().find("yield_theta").view().copy())
                hardening_plasti_thetas_tensor = torch.from_numpy(geo_slot.geometry().edges().find("hardening_plastic_theta").view().copy())

                anchor_thetas_tensor = torch.from_numpy(geo_slot.geometry().edges().find("anchor_theta").view().copy())
                thres_thetas_tensor = torch.from_numpy(geo_slot.geometry().edges().find("thres_theta").view().copy())

                anchor_strain_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("anchor_strain_uv").view().copy())
                thres_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("thres_uv").view().copy())

                plastic_strain_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("plastic_strain_uv").view().copy())
                yield_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("yield_uv").view().copy())
                hardening_plastic_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("hardening_plastic_uv").view().copy())

                stretch_force = net_force(xs_tensor, BENDING_STIFFNESS, BENDING_FRICTION, Hardening_K, rest_yield_theta_ori_tensor, yield_thetas_tensor, hardening_plasti_thetas_tensor, plastic_thetas_tensor, 
                                        thres_thetas_tensor, anchor_thetas_tensor, K11, parm.forward()[0], K12, K33_0, K33_1, Fric_K33, thres_uvs_tensor, anchor_strain_uvs_tensor, 
                                        Hardening_K33, rest_yield_ori_tensor, yield_uvs_tensor, hardening_plastic_uvs_tensor, plastic_strain_uvs_tensor)

                handle_force_scalars.append(stretch_force.sum().item())

                comp_ds.append(comp_d)

                sgui.update()
                pass  
            else:
                run = False

    ps.set_user_callback(update)
    ps.show()


