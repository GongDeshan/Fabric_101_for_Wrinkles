import uipc
import os
import sys
import datetime
import argparse
import json
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



run_type = "init"

tested_fabric_type = "59_Cotton"


_cli = argparse.ArgumentParser(add_help=True)
_cli.add_argument("--fabric", choices=["59_Cotton", "60_Linen", "61_Wool", "62_pet", "63_Silk"],
                  help="Fabric/data-set identifier. Defaults to the value set above.")
_cli.add_argument("--direction", choices=["warp", "weft"], default="warp",
                  help="Twist direction/data file. Training defaults to warp; forward validation supports both.")
_cli.add_argument("--epochs", type=int, help="Override NUM_EPOCH for this run.")
_cli.add_argument("--cotton-warm-start", action="store_true",
                  help="Start Cotton from its prior fitted parameter estimate while using the corrected pipeline.")
_cli.add_argument("--forward-checkpoint", type=str,
                  help="Best-checkpoint JSON to evaluate in a stateful, no-gradient forward rollout.")
_cli.add_argument("--forward-frames", type=int, default=200,
                  help="Number of measured samples to replay in --forward-checkpoint mode (default: 200).")
_cli.add_argument("--headless-train", action="store_true",
                  help="Run optimize_once() immediately instead of waiting for the GUI button.")
_cli_args, _unknown_cli_args = _cli.parse_known_args()
if _cli_args.fabric is not None:
    tested_fabric_type = _cli_args.fabric
if _cli_args.epochs is not None and _cli_args.epochs < 1:
    _cli.error("--epochs must be positive")
if _cli_args.cotton_warm_start and tested_fabric_type != "59_Cotton":
    _cli.error("--cotton-warm-start is only valid with --fabric 59_Cotton")
if _cli_args.forward_checkpoint and _cli_args.headless_train:
    _cli.error("--forward-checkpoint and --headless-train cannot be used together")
if _cli_args.forward_frames < 2:
    _cli.error("--forward-frames must be at least 2")
if _cli_args.cotton_warm_start:
    run_type = "warm_start"
FORWARD_VALIDATION = _cli_args.forward_checkpoint is not None
if FORWARD_VALIDATION:
    run_type = "forward_validation"

if __name__ == '__main__' and not FORWARD_VALIDATION:
    from shear_staged_gui import run as run_shear_staged_gui
    run_shear_staged_gui(__file__, _cli_args)
    raise SystemExit(0)

fabric_type = tested_fabric_type.split("_")[-1].lower()

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
        if "\n" in data:
            self.flush()
    def flush(self):
        for s in self._streams:
            s.flush()

_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(
    _log_dir,
    f"{fabric_type}_twist_{run_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, "a", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f"Logging this run to: {_log_path}")

FABRIC_INITIAL_GUESSES = {
    "59_Cotton": dict(warp=5753754.3771, weft=1229799.9644, k33_0=100.0 * kPa,
                      k33_1=100.0 * kPa, friction_k33=100.0 * kPa, thres_uv=0.001,
                      yield_uv=0.020, hardening_k33=70.0 * kPa, density=430, thickness=0.0004),
    "60_Linen": dict(warp=7268873.8665, weft=4894094.1377, k33_0=100.0 * kPa,
                     k33_1=100.0 * kPa, friction_k33=100.0 * kPa, thres_uv=0.001,
                     yield_uv=0.020, hardening_k33=70.0 * kPa, density=503, thickness=0.0003),
    "61_Wool": dict(warp=3368709.2254, weft=2448119.5590, k33_0=100.0 * kPa,
                    k33_1=100.0 * kPa, friction_k33=100.0 * kPa, thres_uv=0.001,
                    yield_uv=0.020, hardening_k33=70.0 * kPa, density=590, thickness=0.0003),
    "62_pet": dict(warp=7131122.6556, weft=11186885.7799, k33_0=100.0 * kPa,
                   k33_1=100.0 * kPa, friction_k33=100.0 * kPa, thres_uv=0.001,
                   yield_uv=0.020, hardening_k33=70.0 * kPa, density=1000, thickness=0.0001),
    "63_Silk": dict(warp=2333543.1573, weft=1794466.0650, k33_0=100.0 * kPa,
                     k33_1=100.0 * kPa, friction_k33=100.0 * kPa, thres_uv=0.001,
                     yield_uv=0.020, hardening_k33=70.0 * kPa, density=410, thickness=0.0003),
}
_material = FABRIC_INITIAL_GUESSES[tested_fabric_type]

COTTON_WARM_START = dict(k33_0=590783.7353, k33_1=95824.5059,
                         friction_k33=360019.4933, thres_uv=0.001,
                         yield_uv=0.0113, hardening_k33=10.0 * kPa)
if _cli_args.cotton_warm_start:
    _material = {**_material, **COTTON_WARM_START}
    print("Using corrected-pipeline Cotton warm start:", COTTON_WARM_START)

Warp_Stiffness = _material["warp"]
Weft_Stiffness = _material["weft"]

Up_Direction = _cli_args.direction

K11 = Weft_Stiffness if Up_Direction == "warp" else Warp_Stiffness 
K22 = Warp_Stiffness if Up_Direction == "warp" else Weft_Stiffness 

K12 = 0.0
K33_0 = _material["k33_0"]
K33_1 = _material["k33_1"]
Hardening_K33 = _material["hardening_k33"]
Fric_K33 = _material["friction_k33"]
Thres_uv = _material["thres_uv"]
Yield_UV = _material["yield_uv"]
MASS_DENSITY = _material["density"]
THICKNESS = _material["thickness"]




BENDING_STIFFNESS = 4000.0
Hardening_K = 50
YEILD_THETA = 3.14
PLASTIC_THETA = 0.0

BENDING_FRICTION = 1381.8486
ANCHOR_THETA = 0.0
THRES_THETA = 3.14

PARAM_LR = 0.02
NUM_EPOCH = 500
if _cli_args.epochs is not None:
    NUM_EPOCH = _cli_args.epochs

LOOP_STEPS = 40
ONE_DIR_ROT_STEPS = LOOP_STEPS / 2
NUM_LOOPS = 2
EndFrame = LOOP_STEPS * NUM_LOOPS
EndFrame = _cli_args.forward_frames - 1 if FORWARD_VALIDATION else 40

Rotate_rad = np.deg2rad(5)

MESH_PATH = str(__import__('pathlib').Path(__file__).resolve().with_name('mesh.obj'))


IS_ROTATION_STIFFNESS = True
IS_SAVE_LOAD_DEFORMATION: bool = True

training_data_path = (
    str(__import__('pathlib').Path(__file__).resolve().parent/'data'/f'{tested_fabric_type}_{Up_Direction}_twist.npy'))
print(f"Training data: {training_data_path}")
with open(training_data_path, "rb") as f:
    friction_plastic_bending_force = np.load(f)
    raw_twist_angles = np.asarray(friction_plastic_bending_force[0], dtype=np.float64).reshape(-1)
    raw_torque = np.asarray(friction_plastic_bending_force[1], dtype=np.float64).reshape(-1)

if raw_twist_angles.size != raw_torque.size:
    raise RuntimeError(
        f"Training-data angle/torque length mismatch: "
        f"{raw_twist_angles.size} != {raw_torque.size}")
if raw_twist_angles.size < EndFrame + 1:
    raise RuntimeError(
        f"Training data has only {raw_twist_angles.size} samples, but simulation needs {EndFrame + 1}.")

Rotate_rad = np.deg2rad(np.max(raw_twist_angles))
torque_gt = torch.from_numpy(raw_torque[:EndFrame + 1])
if FORWARD_VALIDATION:
    forward_angle_schedule = np.deg2rad(raw_twist_angles[:EndFrame + 1])
    print(f"Forward validation: replaying {EndFrame + 1} measured angle samples without state resets.")
else:
    forward_angle_schedule = None
    print(f"Training alignment: using first {EndFrame + 1} raw samples (one load/unload cycle).")

K33_0_tensor = torch.tensor(K33_0)
K33_1_tensor = torch.tensor(K33_1)

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
config['contact']['d_hat'] = THICKNESS * 0.001
config['gravity'] = [[0.0], [0.0], [0.0]]
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
spc.apply_to(mesh, 200.0)  

is_fixed = mesh.vertices().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
pos_view = mesh.positions().view()
is_fixed_view[:] = np.where(pos_view[:, 1].reshape(-1) < 0.02, 1, 0)

rest_yield_ori_uvs = mesh.triangles().find("yield_uv").view().copy()
rest_yield_ori_tensor = torch.from_numpy(rest_yield_ori_uvs)

rest_yield_theta_ori = mesh.edges().find("yield_theta").view().copy()
rest_yield_theta_ori_tensor = torch.from_numpy(rest_yield_theta_ori)

diff_33_0: AttributeSlotI32 = mesh.triangles().create('diff/k33_0', -1)
view(diff_33_0)[:] = 0  

diff_33_1: AttributeSlotI32 = mesh.triangles().create('diff/k33_1', -1)
view(diff_33_1)[:] = 1  

diff_friction_33: AttributeSlotI32 = mesh.triangles().create('diff/friction_k33', -1)
view(diff_friction_33)[:] = 2  

diff_thres_uv: AttributeSlotI32 = mesh.triangles().create('diff/thres_uv', -1)
view(diff_thres_uv)[:] = 3  

diff_yield_uv: AttributeSlotI32 = mesh.triangles().create('diff/yield_ori_uv', -1) 
view(diff_yield_uv)[:] = 4  

diff_hardening_k33: AttributeSlotI32 = mesh.triangles().create('diff/hardening_k33', -1)
view(diff_hardening_k33)[:] = 5  

print('mesh:', mesh)
geo_slot, _ = cloth_obj.geometries().create(mesh)

edge_topos = geo_slot.geometry().edges().find(builtin.topo).view()
face_topos = geo_slot.geometry().triangles().find(builtin.topo).view()

x_bar = load_mesh_uv(MESH_PATH)[:, :, np.newaxis]
view(geo_slot.geometry().triangles().find("rest_x"))[:] = x_bar

twist_angle = 0.0
d_theta = Rotate_rad / ONE_DIR_ROT_STEPS

def LUFHES_twist_cloth(info: Animation.UpdateInfo):

    global twist_angle

    if info.frame() == 1:
        twist_angle = 0.0

    geo_slots: list[GeometrySlot] = info.geo_slots()
    geo: SimplicialComplex = geo_slots[0].geometry()
    rest_geo_slots: list[GeometrySlot] = info.rest_geo_slots()
    rest_geo: SimplicialComplex = rest_geo_slots[0].geometry()

    is_constrained = geo.vertices().find(builtin.is_constrained)
    is_constrained_view = view(is_constrained)
    aim_position = geo.vertices().find(builtin.aim_position)
    aim_position_view = view(aim_position)

    rest_position_view = rest_geo.positions().view()

    for i in range(len(is_constrained_view)):
        if rest_position_view[i][1] > 0.07:
            is_constrained_view[i] = 1

    if FORWARD_VALIDATION:
        twist_angle = float(forward_angle_schedule[info.frame()])
    elif info.frame() <= (ONE_DIR_ROT_STEPS * 1) or (ONE_DIR_ROT_STEPS * 2) < info.frame() <= (ONE_DIR_ROT_STEPS * 3):
        twist_angle += d_theta
    elif (ONE_DIR_ROT_STEPS * 1) < info.frame() <= (ONE_DIR_ROT_STEPS * 2) or (ONE_DIR_ROT_STEPS * 3) < info.frame():
        twist_angle -= d_theta
    else:
        pass

    rot_z = np.array([
        [np.cos(twist_angle), 0.0, np.sin(twist_angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(twist_angle), 0.0, np.cos(twist_angle)]])

    for i in range(aim_position_view.shape[0]):
        if rest_position_view[i][1] > 0.07:
            aim_position_view[i] = rot_z @ rest_position_view[i]

scene.animator().insert(cloth_obj, LUFHES_twist_cloth)

if FORWARD_VALIDATION:
    rotate_thetas = forward_angle_schedule.copy()
else:
    rotate_thetas = [0.0]
    theta_record = 0.0
    for f in range(1, EndFrame+1):
        if f <= (ONE_DIR_ROT_STEPS * 1) or (ONE_DIR_ROT_STEPS * 2) < f <= (ONE_DIR_ROT_STEPS * 3):
            theta_record += d_theta
            rotate_thetas.append(theta_record)
        elif (ONE_DIR_ROT_STEPS * 1) < f <= (ONE_DIR_ROT_STEPS * 2) or (ONE_DIR_ROT_STEPS * 3) < f:
            theta_record -= d_theta
            rotate_thetas.append(theta_record)
    rotate_thetas = np.array(rotate_thetas)


MAX_SHEAR_STRAIN = 0.029
THRES_YIELD_SPLIT = 0.33 * MAX_SHEAR_STRAIN   

THRES_UV_LO, THRES_UV_HI = 1e-4, THRES_YIELD_SPLIT
YIELD_UV_LO, YIELD_UV_HI = THRES_YIELD_SPLIT, 0.98 * MAX_SHEAR_STRAIN

SHEAR_PARAMETER_BOUNDS = (
    (1.0e3, 1.5e6),  
    (1.0e3, 1.5e6),  
    (1.0e3, 2.0e6),  
    (THRES_UV_LO, THRES_UV_HI),
    (YIELD_UV_LO, YIELD_UV_HI),
    (1.0e3, 1.5e6),  
)

def _logit(frac: float) -> float:
    return float(np.log(frac / (1.0 - frac)))

class MyDiffSimParameter(DiffSimParameter):
    def __init__(self, scene, size, p_ref):
        self.p_ref = torch.as_tensor(p_ref, dtype=torch.double)
        super().__init__(scene, size)
        with torch.no_grad():
            for i, (lower, upper) in enumerate(SHEAR_PARAMETER_BOUNDS):
                initial = float(np.clip(self.p_ref[i].item(), lower + 1e-12, upper - 1e-12))
                self.U()[i] = _logit((initial - lower) / (upper - lower))

    def forward(self) -> torch.Tensor:
        U = self.U()
        lower = torch.tensor([bound[0] for bound in SHEAR_PARAMETER_BOUNDS],
                             dtype=torch.double, device=U.device)
        upper = torch.tensor([bound[1] for bound in SHEAR_PARAMETER_BOUNDS],
                             dtype=torch.double, device=U.device)
        return lower + (upper - lower) * torch.sigmoid(U)

parm = MyDiffSimParameter(scene=scene, size=6,
                           p_ref=[K33_0, K33_1, Fric_K33, Thres_uv, Yield_UV, Hardening_K33])

if FORWARD_VALIDATION:
    checkpoint_path = os.path.abspath(_cli_args.forward_checkpoint)
    with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    expected_parameter_order = ['k33_0', 'k33_1', 'friction_k33',
                                'thres_uv', 'yield_uv', 'hardening_k33']
    if checkpoint.get("parameter_order") != expected_parameter_order:
        raise RuntimeError("Forward checkpoint has an unexpected parameter order")
    checkpoint_parameters = checkpoint.get("physical_parameters")
    if not isinstance(checkpoint_parameters, list) or len(checkpoint_parameters) != 6:
        raise RuntimeError("Forward checkpoint must contain six physical_parameters")
    with torch.no_grad():
        for i, (lower, upper) in enumerate(SHEAR_PARAMETER_BOUNDS):
            physical_value = float(checkpoint_parameters[i])
            if not lower < physical_value < upper:
                raise RuntimeError(f"Checkpoint parameter {i}={physical_value} is outside ({lower}, {upper})")
            parm.U()[i] = _logit((physical_value - lower) / (upper - lower))
    print(f"Loaded forward-validation checkpoint: {checkpoint_path}")

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
    if node_x[1] > 0.07:
        node_force_indices.append(i)
        is_edge_cal_force = is_edge_cal_force | np.any(
            np.isin(edge_topos, i), axis=1).squeeze()
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

        one_e0_idx_tensor[0][bending_edge_stencils[e_idx][1]] = 1.0
        one_e0_idx_tensor[0][bending_edge_stencils[e_idx][0]] = -1.0

        one_e1_idx_tensor[0][bending_edge_stencils[e_idx][2]] = 1.0
        one_e1_idx_tensor[0][bending_edge_stencils[e_idx][0]] = -1.0

        one_e2_idx_tensor[0][bending_edge_stencils[e_idx][3]] = 1.0
        one_e2_idx_tensor[0][bending_edge_stencils[e_idx][0]] = -1.0

        one_e3_idx_tensor[0][bending_edge_stencils[e_idx][2]] = 1.0
        one_e3_idx_tensor[0][bending_edge_stencils[e_idx][1]] = -1.0

        one_e4_idx_tensor[0][bending_edge_stencils[e_idx][3]] = 1.0
        one_e4_idx_tensor[0][bending_edge_stencils[e_idx][1]] = -1.0

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

cloth_select_force_indices = torch.zeros(NUM_VERTICES, dtype=torch.double)

for node_f_idx in node_force_indices:
    cloth_select_force_indices[node_f_idx] = 1.0

zero_z_component_mat = torch.tensor([[1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]], dtype=torch.double)

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

    grad = torch.cat([
        -e0norm / n1SqNorm * n1,
        -e0.dot(e3) / (e0norm * n1SqNorm) * n1 -
        e0.dot(e4) / (e0norm * n2SqNorm) * n2,
        e0.dot(e1) / (e0norm * n1SqNorm) * n1 +
        e0.dot(e2) / (e0norm * n2SqNorm) * n2,
        -e0norm / n2SqNorm * n2])

    edge_hat = (x1 - x2) / torch.linalg.vector_norm(x1 - x2)
    sin_term = torch.linalg.cross(n2, n1).dot(edge_hat)
    dA = torch.atan2(sin_term, n1.dot(n2))

    return dA, grad

def stretch_strain_init(x_init_np):

    x_init_np[:, [1, 2]] = x_init_np[:, [2, 1]]   
    x_init = torch.from_numpy(x_init_np)

    xs_torch_stretching = x_init.flatten()

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
    fuv = torch.bmm(Dut, xv).squeeze() + torch.bmm(Dvt, xu).squeeze()

    strain_uus = G[0]
    strain_vvs = G[3]
    strain_uvs = G[1]

    return strain_uus, strain_vvs, strain_uvs

strain_uus_init, strain_vvs_init, strain_uvs_init = stretch_strain_init(x_init)

def principle_rotation(theta, K11, K22, K12, K33):

    stiffness_mat = torch.tensor([
        [K11, K12, 0.0],
        [K12, K22, 0.0],
        [0.0, 0.0, 0.0]], dtype=torch.double) + K33.view(-1, 1, 1) * torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]], dtype=torch.double).unsqueeze(0)

    T = torch.tensor(    
        [[np.cos(0.0)**2, np.sin(0.0)**2, np.sin(0.0) * np.cos(0.0)],
        [np.cos(theta)**2, np.sin(theta)**2, np.sin(theta) * np.cos(theta)],
        [2 * np.cos(0.0) * np.cos(theta), 2 * np.sin(0.0) * np.sin(theta), np.sin(theta)]])
    

    stiffness_principle = T.T @ stiffness_mat @ T

    K11_principle = stiffness_principle[:,0,0]
    K22_principle = stiffness_principle[:,1,1]
    K12_principle = stiffness_principle[:,0,1]
    K33_principle = stiffness_principle[:,2,2]

    return K11_principle, K22_principle, K12_principle, K33_principle

def net_force(xs_torch, bending_k, friction_k, hardening_k, yield_theta_ori, yield_thetas, hardening_plastic_thetas, plastic_thetas, thres_thetas, anchor_thetas,
              K11, K22, K12, K33_0, K33_1, Friction_K33, thres_uvs, anchor_strain_uvs, hardening_k33, yield_uv_ori, yield_uvs, hardening_plastic_uvs, plastic_strain_uvs,
              is_rotate_yarns=False, rotate_angle=0.0):

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
    n1_dot_n2 = (n1s_theta * n2s_theta).sum(dim=1)
    edge_hat = -e3s / torch.linalg.vector_norm(e3s, dim=1).unsqueeze(1)
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

    Bending_Force = THICKNESS * \
        torch.bmm(cloth_BF_index, (BF_batch + BFric_batch).unsqueeze(2)
                  ).sum(dim=0).squeeze()


    xs_torch_stretching = xs_torch.flatten()

    Dm_w_1 = torch.mv(cloth_face_indices_1, xs_torch_stretching)
    Dm_w_2 = torch.mv(cloth_face_indices_2, xs_torch_stretching)

    Dm_w_l = torch.stack([Dm_w_1, Dm_w_2], 1)

    Dm_w_t = Dm_w_l.view(NUM_FACES, 3, 2)

    F_s = torch.bmm(Dm_w_t, cloth_faces_invDm)

    G = 0.5 * (torch.bmm(F_s.permute(0, 2, 1), F_s) -
               torch.eye(2, dtype=torch.double).unsqueeze(0)).reshape(-1, 4)

    G = G.T

    xu = F_s[:, :, 0].unsqueeze(2)
    xv = F_s[:, :, 1].unsqueeze(2)

    Dut = cloth_Du.permute(0, 2, 1)
    Dvt = cloth_Dv.permute(0, 2, 1)

    fuu = torch.bmm(Dut, xu).squeeze()
    fvv = torch.bmm(Dvt, xv).squeeze()
    fuv = torch.bmm(Dut, xv).squeeze() + torch.bmm(Dvt, xu).squeeze()

    strain_uus = G[0] - strain_uus_init
    strain_vvs = G[3] - strain_vvs_init
    strain_uvs = G[1] - strain_uvs_init

    if is_rotate_yarns:
        K11, K22, K12, K33 = principle_rotation(0.5*np.pi-rotate_angle, K11, K22, K12, K33_0 + K33_1 * torch.abs(strain_uvs))
    else:
        K33 = K33_0 + K33_1 * torch.abs(strain_uvs)

    elastic_strain_uvs = strain_uvs - plastic_strain_uvs
    dis_to_yield_uv = torch.abs(elastic_strain_uvs) - yield_uvs
    hardening_plastic_uvs = hardening_plastic_uvs + torch.relu(dis_to_yield_uv) * (K33/(hardening_k33 + K33))
    new_yield_uvs = yield_uv_ori + hardening_plastic_uvs * (hardening_k33/K33)


    current_elastic_strain_uvs = strain_uvs - (plastic_strain_uvs + torch.sign(elastic_strain_uvs) * torch.relu(torch.abs(elastic_strain_uvs) - new_yield_uvs))


    grad_e = K11 * strain_uus * fuu.T + K22 * strain_vvs * fvv.T + K12 * (strain_uus * fvv.T + strain_vvs * fuu.T) + K33 * current_elastic_strain_uvs * fuv.T


    strain_uvs_var = strain_uvs - anchor_strain_uvs
    dis_to_slide_uv = torch.abs(strain_uvs_var) - thres_uvs

    current_strain_uvs_var = strain_uvs - (anchor_strain_uvs + torch.sign(strain_uvs_var) * torch.relu(dis_to_slide_uv))


    grad_fric = Friction_K33 * current_strain_uvs_var * fuv.T

    Stretching_Force = THICKNESS * torch.mv(cloth_SF_T, (-cloth_faces_m * (grad_e + grad_fric)).T.flatten())

    NetForce_xy = torch.mm((Bending_Force + Stretching_Force).reshape(-1, 3), zero_z_component_mat)

    Pos_xy = torch.mm(xs_torch, zero_z_component_mat)

    torques = torch.cross(Pos_xy, NetForce_xy, dim=1)

    torch_sum = torch.dot(torques[:, 2], cloth_select_force_indices)

    return torch_sum

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
    parm.sync()

PARAM_NAMES = ['k33_0', 'k33_1', 'friction_k33', 'thres_uv', 'yield_uv', 'hardening_k33']
DEAD_GRAD_EPS = 1e-9       
DEAD_GRAD_PATIENCE = 5     
dead_grad_streaks = [0] * len(PARAM_NAMES)
frozen_params = [False] * len(PARAM_NAMES)
_current_epoch = -1
_best_loss = float("inf")
_best_checkpoint_path = _log_path.removesuffix(".log") + ".best.json"


def save_best_checkpoint(loss: torch.Tensor, P: torch.Tensor, torques_pred: torch.Tensor):
    """Persist the pre-step physical parameters for the lowest observed loss."""
    global _best_loss

    loss_value = float(loss.detach().cpu().item())
    if loss_value >= _best_loss:
        return

    _best_loss = loss_value
    payload = {
        "fabric": tested_fabric_type,
        "epoch": _current_epoch,
        "loss_mse": loss_value,
        "parameter_order": PARAM_NAMES,
        "physical_parameters": [float(value) for value in P.detach().cpu().tolist()],
        "predicted_torque": [float(value) for value in torques_pred.detach().cpu().tolist()],
        "ground_truth_torque": [float(value) for value in torque_gt.detach().cpu().tolist()],
    }
    with open(_best_checkpoint_path, "w", encoding="utf-8") as checkpoint_file:
        json.dump(payload, checkpoint_file, indent=2)
    print(f"[best-checkpoint] epoch={_current_epoch}, loss={loss_value:.12g}: "
          f"{_best_checkpoint_path}")

def check_dead_gradients():
    grad = optimizer.param_groups[0]["params"][0].grad
    if grad is None:
        return

    for i, name in enumerate(PARAM_NAMES):
        if abs(grad[i].item()) < DEAD_GRAD_EPS:
            dead_grad_streaks[i] += 1
        else:
            dead_grad_streaks[i] = 0

        if dead_grad_streaks[i] >= DEAD_GRAD_PATIENCE:
            print(f"[dead-grad] '{name}' has |grad| < {DEAD_GRAD_EPS} for "
                  f"{DEAD_GRAD_PATIENCE} epochs (monitoring only; still trainable).")

def loss_fn() -> torch.Tensor:

    global rest_yield_ori_tensor

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

    P = parm.forward()

    parm_k33_0_tensor = P[0]
    parm_k33_1_tensor = P[1]
    parm_friction_k33_tensor = P[2]

    thres_uvs_tensor = P[3] * torch.ones_like(yield_uvs[0], dtype=torch.double)
    yield_ori_uv_tensor = P[4] * torch.ones_like(yield_uvs[0], dtype=torch.double)

    parm_hardening_k33_tensor = P[5]

    hanlde_torques_strength = [torch.tensor(0.0)]

    for f in range(EndFrame):

        X_pred = X[f * NUM_VERTICES * 3: (f+1) * NUM_VERTICES * 3].reshape(-1, 3)

        X_pred = X_pred[:, [0, 2, 1]]

        plastic_thetas_a_frame = plastic_thetas[f]
        yield_theta_a_frame = yield_thetas[f]
        hardening_theta_a_frame = hardening_thetas[f]
        anchor_theta_a_frame = anchor_thetas[f]

        plastic_uvs_a_frame = plastic_uvs[f]
        yield_uvs_a_frame = yield_uvs[f]
        hardening_uvs_a_frame = hardening_uvs[f]
        anchor_uvs_a_frame = anchor_uvs[f]

        
        force = net_force(X_pred, BENDING_STIFFNESS, BENDING_FRICTION, Hardening_K, rest_yield_theta_ori_tensor, yield_theta_a_frame, hardening_theta_a_frame, plastic_thetas_a_frame, 
                    thres_thetas_tensor, anchor_theta_a_frame, K11, K22, K12, parm_k33_0_tensor, parm_k33_1_tensor, parm_friction_k33_tensor, thres_uvs_tensor, anchor_uvs_a_frame, parm_hardening_k33_tensor,
                    yield_ori_uv_tensor, yield_uvs_a_frame, hardening_uvs_a_frame, plastic_uvs_a_frame, IS_ROTATION_STIFFNESS, rotate_thetas[f])

        hanlde_torques_strength.append(force.sum())

    torques_pred = torch.stack(hanlde_torques_strength)

    loss = (torques_pred - torque_gt).square().mean()

    print("Predict torque: ", torques_pred)
    print("GT torque: ", torque_gt)

    P_detached = P.detach()
    print('loss_fn:', loss.item(),
          '\n  k33_0:', P_detached[0].item(),
          '\n  k33_1:', P_detached[1].item(),
          '\n  friction_k33:', P_detached[2].item(),
          '\n  thres_uv:', P_detached[3].item(),
          '\n  yield_uv:', P_detached[4].item(),
          '\n  hardening_k33:', P_detached[5].item())

    losses_record.append(loss.clone().cpu().detach().item())
    save_best_checkpoint(loss, P, torques_pred)

    loss.backward()

    check_dead_gradients()

    print("para gradient: ", optimizer.param_groups[0]["params"][0].grad)

    return loss

def optimize_once():
    global _current_epoch
    for e in range(NUM_EPOCH):
        _current_epoch = e
        print("Epoch :", e)
        optimizer.step(loss_fn)

    Timer.report()

tested_torques = [0.0]
twist_angles = [0.0]

def recover():

    global tested_torques, twist_angles
    global rest_yield_ori_tensor

    tested_torques.clear()
    twist_angles.clear()

    tested_torques.append(0.0)
    twist_angles.append(0.0)

    world.recover(0)
    parm.sync()  

    view(geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 1

    view(geo_slot.geometry().edges().find("dihedral_theta"))[:] = 0.0
    view(geo_slot.geometry().edges().find("plastic_theta"))[:] = rest_plastic_thetas    
    view(geo_slot.geometry().edges().find("anchor_theta"))[:] = rest_anchor_thetas      
    view(geo_slot.geometry().edges().find("yield_theta"))[:] = rest_yield_theta_ori      

    view(geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 1

    view(geo_slot.geometry().triangles().find("anchor_strain_uv"))[:] = rest_anchor_uvs
    view(geo_slot.geometry().triangles().find("plastic_strain_uv"))[:] = rest_plastic_uvs
    view(geo_slot.geometry().triangles().find("yield_uv"))[:] = rest_yield_ori_uvs


    world.recover(0)
    parm.sync()  


def forward_validate():
    """Replay a measured multi-cycle angle schedule with fixed learned parameters."""
    recover()  
    predicted_torques = [0.0]

    with torch.no_grad():
        P = parm.forward().detach()
        for frame in range(1, EndFrame + 1):
            world.advance()
            world.retrieve()

            if frame == 1:
                view(geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 0
                view(geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 0

            xs_numpy = np.stack([
                xs_np.squeeze()
                for xs_np in geo_slot.geometry().vertices().find(builtin.position).view()
            ])
            xs_numpy[:, [1, 2]] = xs_numpy[:, [2, 1]]
            xs_tensor = torch.from_numpy(xs_numpy)

            geo = geo_slot.geometry()
            plastic_thetas = torch.from_numpy(geo.edges().find("plastic_theta").view().copy())
            yield_thetas = torch.from_numpy(geo.edges().find("yield_theta").view().copy())
            hardening_thetas = torch.from_numpy(
                geo.edges().find("hardening_plastic_theta").view().copy())
            anchor_thetas = torch.from_numpy(geo.edges().find("anchor_theta").view().copy())
            thres_thetas = torch.from_numpy(geo.edges().find("thres_theta").view().copy())

            anchor_uvs = torch.from_numpy(geo.triangles().find("anchor_strain_uv").view().copy())
            plastic_uvs = torch.from_numpy(geo.triangles().find("plastic_strain_uv").view().copy())
            yield_uvs = torch.from_numpy(geo.triangles().find("yield_uv").view().copy())
            hardening_uvs = torch.from_numpy(
                geo.triangles().find("hardening_plastic_uv").view().copy())

            thres_uvs = P[3] * torch.ones_like(yield_uvs, dtype=torch.double)
            yield_ori_uvs = P[4] * torch.ones_like(yield_uvs, dtype=torch.double)
            torque = net_force(
                xs_tensor, BENDING_STIFFNESS, BENDING_FRICTION, Hardening_K,
                rest_yield_theta_ori_tensor, yield_thetas, hardening_thetas, plastic_thetas,
                thres_thetas, anchor_thetas, K11, K22, K12,
                P[0], P[1], P[2], thres_uvs, anchor_uvs, P[5], yield_ori_uvs,
                yield_uvs, hardening_uvs, plastic_uvs, IS_ROTATION_STIFFNESS,
                float(forward_angle_schedule[frame]),
            )
            predicted_torques.append(float(torque.item()))

    predicted_torques = np.asarray(predicted_torques, dtype=np.float64)
    target_torques = raw_torque[:EndFrame + 1]
    angles_deg = raw_twist_angles[:EndFrame + 1]
    mse = float(np.mean((predicted_torques - target_torques) ** 2))

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_base = os.path.join(
        _log_dir, f"{fabric_type}_{Up_Direction}_forward_validation_{stamp}")
    np.savez(result_base + ".npz", angle_deg=angles_deg, torque_pred=predicted_torques,
             torque_gt=target_torques, physical_parameters=P.cpu().numpy(), mse=mse)
    np.savetxt(result_base + ".csv",
               np.column_stack((angles_deg, predicted_torques, target_torques)),
               delimiter=",", header="angle_deg,torque_pred,torque_gt", comments="")
    print(f"Forward validation complete: frames={EndFrame + 1}, MSE={mse:.12g}")
    print(f"Saved forward rollout: {result_base}.npz and {result_base}.csv")

world.dump()

if FORWARD_VALIDATION:
    forward_validate()
elif _cli_args.headless_train:
    print(f"Starting headless training: fabric={tested_fabric_type}, epochs={NUM_EPOCH}, "
          f"lr={PARAM_LR}, direction={Up_Direction}")
    optimize_once()
elif GUIInfo.enabled():
    ps.init()
    ps.set_ground_plane_height_mode('manual')
    ps.set_ground_plane_height(-0.3)
    sgui = SceneGUI(scene)
    sgui.register()
    run = False

    def update():
        global tested_torques, twist_angle, twist_angles

        global run
        imgui.Text(f'frame: {world.frame()}')

        if (imgui.Button('Run & Stop')):
            run = not run
            if run:
                recover()

        if (imgui.Button('optim')):
            optimize_once()

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
                xs_numpy[:, [1, 2]] = xs_numpy[:, [2, 1]]
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


                torque = net_force(xs_tensor, BENDING_STIFFNESS, BENDING_FRICTION, Hardening_K, rest_yield_theta_ori_tensor, yield_thetas_tensor, hardening_plasti_thetas_tensor, plastic_thetas_tensor, 
                                    thres_thetas_tensor, anchor_thetas_tensor, K11, K22, K12, K33_0_tensor, K33_1_tensor, Fric_K33, thres_uvs_tensor, anchor_strain_uvs_tensor, 
                                    Hardening_K33, rest_yield_ori_tensor, yield_uvs_tensor, hardening_plastic_uvs_tensor, plastic_strain_uvs_tensor, IS_ROTATION_STIFFNESS, twist_angle)

                tested_torques.append(torque)
                twist_angles.append(twist_angle)

                print("Maximum Plastic Strain UV: ", np.max(geo_slot.geometry().triangles().find("plastic_strain_uv").view()))
                print("Maximum Anchor Strain UV: ", np.max(geo_slot.geometry().triangles().find("anchor_strain_uv").view()))

                print("Strain UU: ", np.max(geo_slot.geometry().triangles().find("strain_uu").view()))
                print("Strain VV: ", np.max(geo_slot.geometry().triangles().find("strain_vv").view()))
                print("Strain UV: ", np.max(geo_slot.geometry().triangles().find("strain_uv").view()))

                print("Maximum Yield UV: ", np.max(np.abs(geo_slot.geometry().triangles().find("yield_uv").view())))
                print("Maximum Yield Ori UV: ", np.max(np.abs(geo_slot.geometry().triangles().find("yield_ori_uv").view())))

                sgui.update()
            else:
                run = False

    ps.set_user_callback(update)
    ps.show()

if IS_SAVE_LOAD_DEFORMATION and not FORWARD_VALIDATION:

    rotation_suffix = "_rot" if IS_ROTATION_STIFFNESS else ""
    save_file_for_plot = str(__import__('pathlib').Path(__file__).resolve().parent/'save_plot_data'/f'{fabric_type}_shear_{Up_Direction}_torque{rotation_suffix}_{run_type}.npy')

    with open(save_file_for_plot, "wb") as f:
        np.save(f, np.stack([np.array(twist_angles), np.array(tested_torques)]))


