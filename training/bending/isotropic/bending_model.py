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
from uipc.constitution import StVKNonlinearFrictionHardeningPlasticShell, DiscreteShellFrictionHardeningPlasticBending
from uipc.unit import kPa
from uipc.gui import GUIInfo, SceneGUI
from diff_sim_friction_plastic_gated_module import DiffSimLR, DiffSimModule, DiffSimParameter

from load_mesh import load_mesh_uv

from asset_dir import AssetDir



if __name__ == '__main__':
    import argparse
    _parser = argparse.ArgumentParser(description='Three-stage isotropic bending training')
    _parser.add_argument('--fabric',default='59_Cotton',choices=['59_Cotton','60_Linen','61_Wool','62_pet','63_Silk'])
    _parser.add_argument('--epochs',type=int,default=100,help='Maximum epochs per stage')
    _parser.add_argument('--headless-train',action='store_true')
    _parser.add_argument('--no-plastic',action='store_true',help='Disable bending plasticity for all isotropic stages and the later joint model')
    _args = _parser.parse_args()
    from bending_staged_gui import run
    run(__file__,_args)
    raise SystemExit(0)

tested_fabric_type = "59_Cotton"


fabric_type = tested_fabric_type.split("_")[-1].lower()

run_type = "syn"

Warp_Stiffness = 5753754.3771
Weft_Stiffness = 1229799.9644

Up_Direction = "warp"

K11 = Weft_Stiffness if Up_Direction == "warp" else Warp_Stiffness 
K22 = Warp_Stiffness if Up_Direction == "warp" else Weft_Stiffness 

K12 = 0.0
K33_0 = 334029.5011
K33_1 = 1259237.5213

Hardening_K33 = 722631.7525
Fric_K33 = 1695927.8589

Thres_uv = 100.0
Yield_UV = 100.0

MASS_DENSITY = 430.0
THICKNESS = 0.0004

BENDING_STIFFNESS = 600.0
Hardening_K = 50.0
YEILD_THETA = 1.0
PLASTIC_THETA = 0.0

BENDING_FRICTION = 400.0
ANCHOR_THETA = 0.0
THRES_THETA = 0.1


COMP_DIS = 0.016

LOOP_STEPS = 100
ONE_DIR_ROT_STEPS = LOOP_STEPS / 2
NUM_LOOPS = 2
EndFrame = 70

EndFrame = 70


PARAM_LR = 0.3 
NUM_EPOCH = 500

TRAIN_START_FRAME = 29
TRAIN_END_FRAME = 70


MESH_PATH = str(__import__('pathlib').Path(__file__).resolve().with_name('mesh.obj'))




with open(__import__('pathlib').Path(__file__).resolve().parent/'data/59_Cotton_warp_compression.npy','rb') as f:
    gt_comp_force=torch.from_numpy(np.load(f)[1][TRAIN_START_FRAME+1:TRAIN_END_FRAME+1])

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
config['contact']['enable'] = False

print(config)
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

spc.apply_to(mesh, 200.0)  # constraint strength ratio


is_fixed = mesh.vertices().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
pos_view = mesh.positions().view()
is_fixed_view[:] = np.where(pos_view[:, 1].reshape(-1) < 0.02, 1, 0)

rest_yield_ori_uvs = mesh.triangles().find("yield_uv").view().copy()
rest_yield_ori_tensor = torch.from_numpy(rest_yield_ori_uvs)

rest_yield_theta_ori = mesh.edges().find("yield_theta").view().copy()
rest_yield_theta_ori_tensor = torch.from_numpy(rest_yield_theta_ori)







diff_friction_k: AttributeSlotI32 = mesh.edges().create('diff/friction_stiffness', -1)
view(diff_friction_k)[:] = 0  




print('mesh:', mesh)
geo_slot, _ = cloth_obj.geometries().create(mesh)

face_topos = geo_slot.geometry().triangles().find(builtin.topo).view()

x_bar = load_mesh_uv(MESH_PATH)[:, :, np.newaxis]
view(geo_slot.geometry().triangles().find("rest_x"))[:] = x_bar

comp_d = 0.0
compress_dis = COMP_DIS / ONE_DIR_ROT_STEPS

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
        else:
            is_constrained_view[i] = 0

    if info.frame() <= ONE_DIR_ROT_STEPS or 2*ONE_DIR_ROT_STEPS < info.frame() <= 3*ONE_DIR_ROT_STEPS:
        comp_d -= compress_dis
    elif ONE_DIR_ROT_STEPS < info.frame() <= 2*ONE_DIR_ROT_STEPS or 3*ONE_DIR_ROT_STEPS < info.frame() :
        comp_d += compress_dis

    for i in range(aim_position_view.shape[0]):
        if rest_position_view[i][1] > 0.07:
            aim_position_view[i] = rest_position_view[i] + np.array([0.0, comp_d, 0.0])[:, np.newaxis]

scene.animator().insert(cloth_obj, LUFHES_compress_cloth)



class MyDiffSimParameter(DiffSimParameter):
    def forward(self) -> torch.Tensor:
        return self.U()**2

parm = MyDiffSimParameter(scene=scene, size=1)

world.init(scene)

rest_plastic_thetas = geo_slot.geometry().edges().find("plastic_theta").view().copy()
rest_anchor_thetas = geo_slot.geometry().edges().find("anchor_theta").view().copy()

rest_anchor_uvs = geo_slot.geometry().triangles().find("anchor_strain_uv").view().copy()
rest_plastic_uvs = geo_slot.geometry().triangles().find("plastic_strain_uv").view().copy()

rest_strain_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("strain_uv").view().copy())
rest_strain_uus_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("strain_uu").view().copy())
rest_strain_vvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("strain_vv").view().copy())

node_force_indices = []

egde_rest_lengths = geo_slot.geometry().edges().find("L0").view()
V_bars = geo_slot.geometry().edges().find("V_bar").view()
h_bars = geo_slot.geometry().edges().find("h_bar").view()
bending_edge_stencils = geo_slot.geometry().edges().find("stencil").view()

is_edge_cal_force = np.zeros(mesh.edges().size(), dtype=np.bool)
is_face_cal_force = np.zeros(mesh.triangles().size(), dtype=np.bool)

x_init = mesh.vertices().find(builtin.position).view().copy()

for i, node_x in enumerate(x_init):
    if node_x[1] > 0.07:
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
    tri_1_indices_col = torch.tensor(
        [n1_idx * 3, n0_idx * 3], dtype=torch.long).repeat_interleave(3) + torch.arange(3, dtype=torch.long).repeat(2)
    tri_2_indices_col = torch.tensor(
        [n2_idx * 3, n0_idx * 3], dtype=torch.long).repeat_interleave(3) + torch.arange(3, dtype=torch.long).repeat(2)

    tri_1_spa_indices = torch.stack(
        [tri_indices_row, tri_1_indices_col], dim=0)
    tri_2_spa_indices = torch.stack(
        [tri_indices_row, tri_2_indices_col], dim=0)

    tri_1_indices = torch.sparse_coo_tensor(
        tri_1_spa_indices, tri_values, (3, NUM_VERTICES * 3), dtype=torch.int8)
    tri_2_indices = torch.sparse_coo_tensor(
        tri_2_spa_indices, tri_values, (3, NUM_VERTICES * 3), dtype=torch.int8)

    tris_indices_1.append(tri_1_indices)
    tris_indices_2.append(tri_2_indices)

    xs_tri_rest = x_bar[f_i].squeeze()

    face_dm_m = np.stack([xs_tri_rest[2:4] - xs_tri_rest[:2],
                         xs_tri_rest[4:6] - xs_tri_rest[:2]], axis=1)

    dm_inv = np.linalg.inv(face_dm_m)

    faces_invDms.append(dm_inv)

    d = dm_inv.flatten()

    Du = torch.cat([(-d[0] - d[2]) * torch.eye(3, dtype=torch.double), d[0] *
                   torch.eye(3, dtype=torch.double), d[2] * torch.eye(3, dtype=torch.double)], dim=1)
    Dv = torch.cat([(-d[1] - d[3]) * torch.eye(3, dtype=torch.double), d[1] *
                   torch.eye(3, dtype=torch.double), d[3] * torch.eye(3, dtype=torch.double)], dim=1)

    Dus.append(Du)
    Dvs.append(Dv)

    stretch_f_tensor = torch.zeros((NUM_VERTICES * 3, 9), dtype=torch.int8)

    stretch_f_tensor[n0_idx * 3:n0_idx * 3 + 3,
                     0:3] = torch.eye(3, dtype=torch.int8)
    stretch_f_tensor[n1_idx * 3:n1_idx * 3 + 3,
                     3:6] = torch.eye(3, dtype=torch.int8)
    stretch_f_tensor[n2_idx * 3:n2_idx * 3 + 3,
                     6:9] = torch.eye(3, dtype=torch.int8)

    d_n0_x_d_n0 = n0_idx * (NUM_VERTICES * 3 * 3) + n0_idx * 3
    d_n0_x_d_n1 = n0_idx * (NUM_VERTICES * 3 * 3) + n1_idx * 3
    d_n0_x_d_n2 = n0_idx * (NUM_VERTICES * 3 * 3) + n2_idx * 3

    d_n0_y_d_n0 = n0_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n0_idx * 3
    d_n0_y_d_n1 = n0_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n1_idx * 3
    d_n0_y_d_n2 = n0_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n2_idx * 3

    d_n0_z_d_n0 = n0_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n0_idx * 3
    d_n0_z_d_n1 = n0_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n1_idx * 3
    d_n0_z_d_n2 = n0_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n2_idx * 3

    d_n1_x_d_n0 = n1_idx * (NUM_VERTICES * 3 * 3) + n0_idx * 3
    d_n1_x_d_n1 = n1_idx * (NUM_VERTICES * 3 * 3) + n1_idx * 3
    d_n1_x_d_n2 = n1_idx * (NUM_VERTICES * 3 * 3) + n2_idx * 3

    d_n1_y_d_n0 = n1_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n0_idx * 3
    d_n1_y_d_n1 = n1_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n1_idx * 3
    d_n1_y_d_n2 = n1_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n2_idx * 3

    d_n1_z_d_n0 = n1_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n0_idx * 3
    d_n1_z_d_n1 = n1_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n1_idx * 3
    d_n1_z_d_n2 = n1_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n2_idx * 3

    d_n2_x_d_n0 = n2_idx * (NUM_VERTICES * 3 * 3) + n0_idx * 3
    d_n2_x_d_n1 = n2_idx * (NUM_VERTICES * 3 * 3) + n1_idx * 3
    d_n2_x_d_n2 = n2_idx * (NUM_VERTICES * 3 * 3) + n2_idx * 3

    d_n2_y_d_n0 = n2_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n0_idx * 3
    d_n2_y_d_n1 = n2_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n1_idx * 3
    d_n2_y_d_n2 = n2_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) + n2_idx * 3

    d_n2_z_d_n0 = n2_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n0_idx * 3
    d_n2_z_d_n1 = n2_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n1_idx * 3
    d_n2_z_d_n2 = n2_idx * (NUM_VERTICES * 3 * 3) + \
        (NUM_VERTICES * 3) * 2 + n2_idx * 3

    row_index = torch.tensor([d_n0_x_d_n0, d_n0_x_d_n1, d_n0_x_d_n2, d_n0_y_d_n0, d_n0_y_d_n1, d_n0_y_d_n2, d_n0_z_d_n0, d_n0_z_d_n1, d_n0_z_d_n2,
                              d_n1_x_d_n0, d_n1_x_d_n1, d_n1_x_d_n2, d_n1_y_d_n0, d_n1_y_d_n1, d_n1_y_d_n2, d_n1_z_d_n0, d_n1_z_d_n1, d_n1_z_d_n2,
                              d_n2_x_d_n0, d_n2_x_d_n1, d_n2_x_d_n2, d_n2_y_d_n0, d_n2_y_d_n1, d_n2_y_d_n2, d_n2_z_d_n0, d_n2_z_d_n1, d_n2_z_d_n2], dtype=torch.long)

    row_index = row_index.repeat_interleave(
        3) + torch.tensor([0, 1, 2], dtype=torch.int).repeat(27)

    col_index = torch.arange(81, dtype=torch.long)

    indices = torch.stack([row_index, col_index], dim=0)

    values = torch.ones(81, dtype=torch.int8)

    stretch_J_tensor = torch.sparse_coo_tensor(
        indices, values, (NUM_VERTICES * 3 * NUM_VERTICES * 3, 3 * 3 * 3 * 3), dtype=torch.int8)

    stretch_f_tensors.append(stretch_f_tensor.to_sparse())
    stretch_J_tensors.append(stretch_J_tensor)

    x0_init, x1_init, x2_init = [x_init[i].squeeze() for i in f_vs.squeeze()]

    rest_areas.append(
        0.5 * np.linalg.norm(np.cross(x1_init - x0_init, x2_init - x0_init)))

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

        valid_idx_tensor = torch.zeros(
            egde_rest_lengths.shape, dtype=torch.double)

        one_BF_index_tensor = torch.zeros(
            (NUM_VERTICES * 3, 12), dtype=torch.double)

        valid_idx_tensor[e_idx] = 1.0

        valid_edge_index.append(valid_idx_tensor)

        one_idx_tensor = torch.zeros((4, NUM_VERTICES), dtype=torch.double)

        for i in range(4):
            one_idx_tensor[i][bending_edge_stencils[e_idx][i]] = 1.0

            for j in range(3):
                one_BF_index_tensor[bending_edge_stencils[e_idx]
                                    [i] * 3 + j, i * 3 + j] = 1.0

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

cloth_l0s = torch.mv(cloth_valid_indices,
                     torch.from_numpy(egde_rest_lengths.copy()))
cloth_v_bars = torch.mv(cloth_valid_indices, torch.from_numpy(V_bars.copy()))
cloth_h_bars = torch.mv(cloth_valid_indices, torch.from_numpy(h_bars.copy()))

cloth_select_force_indices = torch.zeros((NUM_VERTICES, 3), dtype=torch.double)

for node_f_idx in node_force_indices:
    cloth_select_force_indices[node_f_idx] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.double)

def dihedral_angle(x0, x1, x2, x3):

    e0 = x1 - x0
    e1 = x2 - x0
    e2 = x3 - x0
    e3 = x2 - x1
    e4 = x3 - x1

    n1 = torch.linalg.cross(e0, e1)
    n2 = torch.linalg.cross(e2, e0)

    n1SqNorm = n1.dot(n1)
    n2SqNorm = n2.dot(n2)
    e0norm = torch.norm(e0)

    n1 = torch.linalg.cross(x1 - x0, x2 - x0)
    n2 = torch.linalg.cross(x2 - x3, x1 - x3)

    n1SqNorm_theta = n1.dot(n1)
    n2SqNorm_theta = n2.dot(n2)

    grad = torch.cat([
        -e0norm / n1SqNorm * n1,
        -e0.dot(e3) / (e0norm * n1SqNorm) * n1 -
        e0.dot(e4) / (e0norm * n2SqNorm) * n2,
        e0.dot(e1) / (e0norm * n1SqNorm) * n1 +
        e0.dot(e2) / (e0norm * n2SqNorm) * n2,
        -e0norm / n2SqNorm * n2])

    dA = torch.acos(torch.max(-torch.ones(1), torch.min(torch.ones(1),
                    n1.dot(n2) / torch.sqrt(n1SqNorm_theta * n2SqNorm_theta))))

    if (torch.linalg.cross(n2, n1).dot(x1 - x2) < 0):
        dA = -dA

    return dA, grad

def net_force(xs_torch, bending_k, friction_k, hardening_k, yield_theta_ori, yield_thetas, hardening_plastic_thetas, plastic_thetas, thres_thetas, anchor_thetas,
              K11, K22, K12, K33_0, K33_1, Friction_K33, thres_uvs, anchor_strain_uvs, hardening_k33, yield_uv_ori, yield_uvs, hardening_plastic_uvs, plastic_strain_uvs):

    e0s = torch.matmul(cloth_e0_indices, xs_torch).squeeze()
    e1s = torch.matmul(cloth_e1_indices, xs_torch).squeeze()
    e2s = torch.matmul(cloth_e2_indices, xs_torch).squeeze()
    e3s = torch.matmul(cloth_e3_indices, xs_torch).squeeze()
    e4s = torch.matmul(cloth_e4_indices, xs_torch).squeeze()

    e5s = torch.matmul(cloth_e0_n_indices, xs_torch).squeeze()
    e6s = torch.matmul(cloth_e2_n_indices, xs_torch).squeeze()

    n1s = torch.cross(e0s, e1s, dim=1)
    n2s = torch.cross(e2s, e0s, dim=1)

    n1s_norm = torch.linalg.norm(n1s, dim=1)
    n2s_norm = torch.linalg.norm(n2s, dim=1)

    e0s_norm = torch.linalg.norm(e0s, dim=1)

    n1s_sqnorm = n1s_norm**2
    n2s_sqnorm = n2s_norm**2

    n1s_theta = torch.cross(e0s, e1s, dim=1)
    n2s_theta = torch.cross(e5s, e6s, dim=1)

    n1_dot_n2 = torch.bmm(n1s_theta.unsqueeze(1), n2s_theta.unsqueeze(2)).squeeze()

    edge_hat = -e0s / e0s_norm.unsqueeze(1)
    sin_term = (torch.cross(n2s_theta, n1s_theta, dim=1) * edge_hat).sum(dim=1)
    thetas = torch.atan2(sin_term, n1_dot_n2)
    
    grad_part_1 = (-e0s_norm / n1s_sqnorm).unsqueeze(1) * n1s
    grad_part_2 = -(torch.bmm(e0s.unsqueeze(1), e3s.unsqueeze(2)).squeeze() / (e0s_norm * n1s_sqnorm)).unsqueeze(1) * n1s - (
        torch.bmm(e0s.unsqueeze(1), e4s.unsqueeze(2)).squeeze() / (e0s_norm * n2s_sqnorm)).unsqueeze(1) * n2s
    grad_part_3 = (torch.bmm(e0s.unsqueeze(1), e1s.unsqueeze(2)).squeeze() / (e0s_norm * n1s_sqnorm)).unsqueeze(1) * n1s + (
        torch.bmm(e0s.unsqueeze(1), e2s.unsqueeze(2)).squeeze() / (e0s_norm * n2s_sqnorm)).unsqueeze(1) * n2s
    grad_part_4 = (-e0s_norm / n2s_sqnorm).unsqueeze(1) * n2s

    grads = torch.concat(
        [grad_part_1, grad_part_2, grad_part_3, grad_part_4], dim=1)

    elastic_theta = thetas - torch.mv(cloth_valid_indices, plastic_thetas)
    dis_to_yield = torch.abs(elastic_theta) - torch.mv(cloth_valid_indices, yield_thetas)
    hardening_plastic_thetas_valid = torch.mv(cloth_valid_indices, hardening_plastic_thetas)
    hardening_plastic_thetas_valid = hardening_plastic_thetas_valid + torch.relu(dis_to_yield) * (bending_k/(hardening_k + bending_k))
    new_yield_thetas = torch.mv(cloth_valid_indices, yield_theta_ori) + hardening_plastic_thetas_valid * (hardening_k/bending_k)

    elastic_theta_current = thetas - (torch.mv(cloth_valid_indices, plastic_thetas) + torch.sign(elastic_theta) * torch.relu(torch.abs(elastic_theta) - new_yield_thetas))

    BF_batch = (-cloth_l0s * bending_k * 2.0 * elastic_theta_current / cloth_h_bars * cloth_v_bars).unsqueeze(1) * grads 

    var_theta = thetas - torch.mv(cloth_valid_indices, anchor_thetas)
    dis_to_slide = torch.abs(var_theta) - torch.mv(cloth_valid_indices, thres_thetas)

    var_theta_current = thetas - (torch.mv(cloth_valid_indices, anchor_thetas) + torch.sign(var_theta) * torch.relu(dis_to_slide))

    BFric_batch = (-cloth_l0s * friction_k * 2.0 * var_theta_current / cloth_h_bars * cloth_v_bars).unsqueeze(1) * grads

    Bending_Force = torch.bmm(cloth_BF_index, (BF_batch + BFric_batch).unsqueeze(2)).sum(dim=0).squeeze().reshape(-1, 3)


    xs_torch_stretching = xs_torch.flatten()

    Dm_w_1 = torch.mv(cloth_face_indices_1, xs_torch_stretching)
    Dm_w_2 = torch.mv(cloth_face_indices_2, xs_torch_stretching)

    Dm_w_l = torch.stack([Dm_w_1, Dm_w_2], 1)

    Dm_w_t = Dm_w_l.view(NUM_FACES, 3, 2)

    F_s = torch.bmm(Dm_w_t, cloth_faces_invDm)

    G = 0.5 * (torch.bmm(F_s.permute(0, 2, 1), F_s) - torch.eye(2, dtype=torch.double).unsqueeze(0)).reshape(-1, 4)

    G = G.T

    xu = F_s[:, :, 0].unsqueeze(2)
    xv = F_s[:, :, 1].unsqueeze(2)

    Dut = cloth_Du.permute(0, 2, 1)
    Dvt = cloth_Dv.permute(0, 2, 1)

    fuu = torch.bmm(Dut, xu).squeeze()
    fvv = torch.bmm(Dvt, xv).squeeze()
    fuv = 0.5 * (torch.bmm(Dut, xv).squeeze() + torch.bmm(Dvt, xu).squeeze())

    strain_uus = G[0] - rest_strain_uus_tensor
    strain_vvs = G[3] - rest_strain_vvs_tensor
    strain_uvs = G[1]

    K33 = K33_0 + K33_1 * torch.abs(strain_uvs)

    elastic_strain_uvs = strain_uvs - plastic_strain_uvs
    dis_to_yield_uv = torch.abs(elastic_strain_uvs) - yield_uvs
    hardening_plastic_uvs = hardening_plastic_uvs + torch.relu(dis_to_yield_uv) * (K33/(hardening_k33 + K33))
    new_yield_uvs = yield_uv_ori + hardening_plastic_uvs * (hardening_k33/K33)
    
    current_elastic_strain_uvs = strain_uvs - (plastic_strain_uvs + torch.sign(elastic_strain_uvs) * torch.relu(torch.abs(elastic_strain_uvs) - new_yield_uvs))

    shear_stress = (K33 * current_elastic_strain_uvs
                    + 0.5 * K33_1 * torch.sign(strain_uvs)
                    * current_elastic_strain_uvs.square())
    grad_e = K11 * strain_uus * fuu.T + K22 * strain_vvs * fvv.T + K12 * (strain_uus * fvv.T + strain_vvs * fuu.T) + shear_stress * fuv.T

    strain_uvs_var = strain_uvs - anchor_strain_uvs
    dis_to_slide_uv = torch.abs(strain_uvs_var) - thres_uvs
    current_strain_uvs_var = strain_uvs - (anchor_strain_uvs + torch.sign(strain_uvs_var) * torch.relu(dis_to_slide_uv))


    grad_fric = Friction_K33 * current_strain_uvs_var * fuv.T

    Stretching_Force = (THICKNESS * torch.mv(cloth_SF_T, (-cloth_faces_m * (grad_e + grad_fric)).T.flatten())).reshape(-1, 3)

    compress_force = torch.bmm((Bending_Force + Stretching_Force).unsqueeze(1), cloth_select_force_indices.unsqueeze(2)).squeeze()

    return compress_force.sum()

def dof_select(info: DiffSimModule.DofSelectInfo):
    geo = geo_slot.geometry()
    offset = geo.meta().find(builtin.dof_offset).view()[0]
    verts_idx = offset + np.arange(NUM_VERTICES * 3, dtype=np.int32)
    sdi = torch.from_numpy(verts_idx)
    return sdi

diff_module = DiffSimModule(world, geo_slot=geo_slot, end_frame=EndFrame, parm=parm, dof_select=dof_select)

optimizer = optim.Adam(diff_module.parameters(), lr=PARAM_LR)
lr = DiffSimLR(diff_module, optimizer)

losses_record = []

with torch.no_grad():

    u = np.sqrt(BENDING_FRICTION)   
    parm.U().fill_(u)

    parm.sync()

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








    parm_bending_friction_k_tensor = optimizer.param_groups[0]["params"][0]**2




    compress_force_strength = []

    for f in range(TRAIN_START_FRAME, TRAIN_END_FRAME):

        X_pred = X[f * NUM_VERTICES * 3: (f+1) * NUM_VERTICES * 3].reshape(-1, 3)


        plastic_thetas_a_frame = plastic_thetas[f]
        yield_theta_a_frame = yield_thetas[f]
        hardening_theta_a_frame = hardening_thetas[f]
        anchor_theta_a_frame = anchor_thetas[f]

        plastic_uvs_a_frame = plastic_uvs[f]
        yield_uvs_a_frame = yield_uvs[f]
        hardening_uvs_a_frame = hardening_uvs[f]
        anchor_uvs_a_frame = anchor_uvs[f]

        force = net_force(X_pred, BENDING_STIFFNESS, parm_bending_friction_k_tensor, Hardening_K, rest_yield_theta_ori_tensor, yield_theta_a_frame, hardening_theta_a_frame, plastic_thetas_a_frame, 
                          thres_thetas_tensor, anchor_theta_a_frame, K11, K22, K12, K33_0, K33_1, Fric_K33, thres_uvs_tensor, anchor_uvs_a_frame, Hardening_K33, 
                          rest_yield_ori_tensor, yield_uvs_a_frame, hardening_uvs_a_frame, plastic_uvs_a_frame)

        compress_force_strength.append(force)

    forces_pred = torch.stack(compress_force_strength)

    print("predict force: ", forces_pred)
    print("GT force: ", gt_comp_force)

    loss = (forces_pred - gt_comp_force).square().mean()

    print('loss_fn:', loss, 'parm:', parm)

    losses_record.append(loss.clone().cpu().detach().item())

    loss.backward()

    print("para gradient: ", optimizer.param_groups[0]["params"][0].grad)

    return loss

def optimize_once():
    for e in range(NUM_EPOCH):
        print("Epoch :", e)
        optimizer.step(loss_fn)

    Timer.report()

tested_loads = [0.0]
compress_dists = [0.0]

def recover():

    global tested_torques, twist_angles

    tested_loads.clear()
    compress_dists.clear()

    tested_loads.append(0.0)
    compress_dists.append(0.0)

    world.recover(0)

    view(geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 1

    view(geo_slot.geometry().edges().find("dihedral_theta"))[:] = 0.0
    view(geo_slot.geometry().edges().find("plastic_theta"))[:] = rest_plastic_thetas    
    view(geo_slot.geometry().edges().find("anchor_theta"))[:] = rest_anchor_thetas      
    view(geo_slot.geometry().edges().find("yield_theta"))[:] = rest_yield_theta_ori_tensor      
    
    view(geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 1

    view(geo_slot.geometry().triangles().find("anchor_strain_uv"))[:] = rest_anchor_uvs
    view(geo_slot.geometry().triangles().find("plastic_strain_uv"))[:] = rest_plastic_uvs
    view(geo_slot.geometry().triangles().find("yield_uv"))[:] = rest_yield_ori_uvs

    parm.sync()  


world.dump()

def save_mesh_sequence(frame: int):

    io = SimplicialComplexIO()

    io.write(
        f"{output_dir}/cloth_{frame:04}.obj",
        geo_slot.geometry(),
    )

if GUIInfo.enabled():
    ps.init()
    ps.set_ground_plane_height_mode('manual')
    ps.set_ground_plane_height(-0.3)
    sgui = SceneGUI(scene)
    sgui.register()
    run = False

    def update():
        global tested_torques, compress_dis, compress_dists
        global rest_strain_uvs_tensor, rest_strain_uus_tensor, rest_strain_vvs_tensor
        
        global run
        imgui.Text(f'frame: {world.frame()}')

        if (imgui.Button('Run & Stop')):
            run = not run
            if run:
                recover()

        if (imgui.Button('optim')):
            optimize_once()

        if (imgui.Button("print_load")):
            print(tested_loads)

        if run:
            if (world.frame() < EndFrame):

                print(f"Frame {world.frame()}")

                world.advance()
                world.retrieve()

                if world.frame() == 1:
                    view(geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 0
                    view(geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 0

                xs_numpy = np.stack([xs_np.squeeze() for xs_np in geo_slot.geometry(
                ).vertices().find(builtin.position).view()])
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

                compress_force = net_force(xs_tensor, BENDING_STIFFNESS, BENDING_FRICTION, Hardening_K, rest_yield_theta_ori_tensor, yield_thetas_tensor, hardening_plasti_thetas_tensor, plastic_thetas_tensor, 
                                           thres_thetas_tensor, anchor_thetas_tensor, K11, K22, K12, K33_0, K33_1, Fric_K33, thres_uvs_tensor, anchor_strain_uvs_tensor, 
                                        Hardening_K33, rest_yield_ori_tensor, yield_uvs_tensor, hardening_plastic_uvs_tensor, plastic_strain_uvs_tensor)

                tested_loads.append(compress_force)
                compress_dists.append(comp_d)




                sgui.update()
            else:
                run = False

    ps.set_user_callback(update)
    ps.show()

fabric_type = "syn"

force_log_path = f"D:/Research_4/libipc/diffuipc-demos/python/final_experiments/save_plot_data/{fabric_type}_compress_{Up_Direction}_{run_type}.npy"


