import os
import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer
from uipc import Vector3, Vector2, Matrix4x4, Transform, Quaternion, AngleAxis
from uipc import builtin

from uipc.core import *
from uipc.geometry import *
from uipc.constitution import *
from uipc.gui import *

from uipc.unit import MPa, GPa, kPa

from pathlib import Path

from load_mesh import load_mesh_uv

scene_dir = Path(__file__).resolve().parent

Rot_Theta = 0.0
one_dir_frames = 100
Delta_Theta = np.pi/one_dir_frames

rot_stop_frame = int((one_dir_frames / 2) - 1)

throw_frames = 10

plane_raw_obj_path = str(scene_dir / "meshes" / "paper_plane_raw.obj")
plane_final_obj_path = str(scene_dir / "meshes" / "paper_plane_final.obj")


output_dir = str(scene_dir / "output" / "fold_plane" / "meshes")

workspace = str(scene_dir / "output" / "fold_plane" / "workspace")
os.makedirs(workspace, exist_ok=True)

engine = Engine('cuda', workspace)
world = World(engine)

MASS_DENSITY = 992.875
THICKNESS = 0.00008

# Fixed isotropic stretching stiffness and learned shear parameters, in Pa.
Warp_Stiffness = 188180336.22351965
Weft_Stiffness = 188180336.22351965
Up_Direction = "warp"
K11 = Weft_Stiffness if Up_Direction == "warp" else Warp_Stiffness
K22 = Warp_Stiffness if Up_Direction == "warp" else Weft_Stiffness
K12 = 0.0

K33_0 = 9874486.76382516
K33_1 = 11047956.090603216
Fric_K33 = 11993285.28583167
Hardening_K33 = 104082.21194229477
Thres_uv = 0.008529096038169046
Yield_UV = 0.021834275670439288

# Best isotropic bending checkpoint from warp five-parameter training.
BENDING_STIFFNESS = 45680.40496350644
BENDING_FRICTION = 41512.9186031141
THRES_THETA = 0.18225525128754413
Hardening_K = 1003.7625913305119
YEILD_THETA = 0.29320661294010936
PLASTIC_THETA = 0.0
ANCHOR_THETA = 0.0

dt = 0.01
config = Scene.default_config()
config['dt'] = dt
config['contact']['d_hat'] = 0.001
config['contact']['enable'] = True
# config['gravity'] = [[0.0], [0.0], [0.0]]
# config['newton']['velocity_tol'] = 1e-2

print(config)
scene = Scene(config)

# setup the scene
t = Transform.Identity()
t.scale(1.0)
io = SimplicialComplexIO(t)

cloth_mesh = io.read(plane_raw_obj_path)
label_surface(cloth_mesh)

nlstckfhps = StVKNonlinearFrictionHardeningPlasticShell()
dsfhpb = DiscreteShellFrictionHardeningPlasticBending()

nlstckfhps.apply_to(cloth_mesh, K11, K12, K22, K33_0, K33_1, hardening_k33=Hardening_K33, friction_k33=Fric_K33,
                  thres_uv=Thres_uv, anchor_strain_uv=0.0, yield_uv=Yield_UV, plastic_strain_uv=0.0,
                  mass_density=MASS_DENSITY, thickness=THICKNESS)

dsfhpb.apply_to(cloth_mesh, bending_stiffness=BENDING_STIFFNESS, hardening_k=Hardening_K, yield_theta=YEILD_THETA, 
                plastic_theta=PLASTIC_THETA, friction_k=BENDING_FRICTION, thres_theta=THRES_THETA, anchor_theta=ANCHOR_THETA)

# Constraints
spc = SoftPositionConstraint()
spc.apply_to(cloth_mesh, 200000.0)  # constraint strength ratio

cloth = scene.objects().create("cloth")
cloth_geo_slot, _ = cloth.geometries().create(cloth_mesh)

# Load cloth mesh rest shape from wavefront uv coordinate
cloth_x_bar = load_mesh_uv(plane_raw_obj_path)[:,:, np.newaxis] * 1.0
view(cloth_geo_slot.geometry().triangles().find("rest_x"))[:] = cloth_x_bar

cloth_ver_pos = cloth_geo_slot.geometry().vertices().find(builtin.position).view()
cloth_ver_rest_pos = cloth_ver_pos.copy()

plane_final_mesh = io.read(plane_final_obj_path)
plane_final_ver_pos = plane_final_mesh.vertices().find(builtin.position).view().copy()
plane_final_fold_path = np.zeros_like(plane_final_ver_pos)
phase_one_ver_pos = np.zeros_like(plane_final_ver_pos)

EndFrame = 300

verts_left = [i for i, pos in enumerate(cloth_ver_pos) if pos[0] > 0.0 ]
verts_right = [i for i, pos in enumerate(cloth_ver_pos) if pos[0] < 0.0]
verts_const = [i for i, pos in enumerate(cloth_ver_pos) if pos[0] < 0.02 and pos[0] > -0.02 ]

const_list = []
left_vert_list = []
right_vert_list = []

for i in range(cloth_ver_pos.shape[0]):
    if i in verts_const:
        const_list.append(1)
    else:
        const_list.append(0)

    if i in verts_left:
        left_vert_list.append(1)
    else:
        left_vert_list.append(0)

    if i in verts_right:
        right_vert_list.append(1)
    else:
        right_vert_list.append(0)

const_list_np = np.array(const_list)
left_vert_list_np = np.array(left_vert_list)
right_vert_list_np = np.array(right_vert_list)

def throw_paper_plane(info):

    global Rot_Theta, Delta_Theta, phase_one_ver_pos, plane_final_fold_path

    if world.frame() == one_dir_frames:
        Delta_Theta = -Delta_Theta

    geo_slots = info.geo_slots()
    geo = geo_slots[0].geometry()

    is_constrainted = geo.vertices().find(builtin.is_constrained)
    view(is_constrainted)[:] = 1

    if world.frame() > 100:
        is_constrainted = geo.vertices().find(builtin.is_constrained)
        view(is_constrainted)[:] = const_list_np
        
        return

    if world.frame() < rot_stop_frame:

        Rot_Theta += Delta_Theta

        Rot_Z = np.array([[np.cos(Rot_Theta), -np.sin(Rot_Theta), 0.0],
                        [np.sin(Rot_Theta), np.cos(Rot_Theta), 0.0],
                        [0.0, 0.0, 1.0]])
        
        Rot_Z_oppo = np.array([[np.cos(-Rot_Theta), -np.sin(-Rot_Theta), 0.0],
                        [np.sin(-Rot_Theta), np.cos(-Rot_Theta), 0.0],
                        [0.0, 0.0, 1.0]])
        
        aim_position = geo.vertices().find(builtin.aim_position)
        aim_position_view = view(aim_position)[:]

        for i, r_pos in enumerate(cloth_ver_rest_pos):
            if i in verts_left: 
                aim_position_view[i] = Rot_Z @ r_pos
            elif i in verts_right:
                aim_position_view[i] = Rot_Z_oppo @ r_pos
    elif world.frame() >= rot_stop_frame and world.frame() < 100:

        if world.frame() == rot_stop_frame:
            phase_one_ver_pos = geo.vertices().find(builtin.position).view()
            plane_final_fold_path = (plane_final_ver_pos - phase_one_ver_pos) / 50.0

        aim_position = geo.vertices().find(builtin.aim_position)
        aim_position_view = view(aim_position)[:]

        for i, r_pos in enumerate(cloth_ver_rest_pos):
            aim_position_view[i] = phase_one_ver_pos[i] + plane_final_fold_path[i]

scene.animator().insert(cloth, throw_paper_plane)

world.init(scene)

world.dump()

def save_mesh_sequence(frame: int):
    os.makedirs(output_dir, exist_ok=True)

    io = SimplicialComplexIO()

    io.write(
        f"{output_dir}/cloth_{frame: 04}.obj",
        cloth_geo_slot.geometry(),
    )

if GUIInfo.enabled():

    ps.init()
    # ps.set_ground_plane_height(0.0)
    ps.set_ground_plane_mode('none')
    sgui = SceneGUI(scene)
    sgui.register()
    run = False

    def update():
        global run
        imgui.Text(f'frame: {world.frame()}')

        if(imgui.Button('One Step')):
            if(world.frame() < EndFrame):
                world.advance()
                world.retrieve()
                sgui.update()
            run = False
        
        if(imgui.Button('Run & Stop')):
            run = not run
            if run:
                world.recover(0)

        if(imgui.Button('Stop & Continue')):
            run = not run
        
        if run:
            if(world.frame() < EndFrame):
                world.advance()
                world.retrieve()
                save_mesh_sequence(world.frame())
                sgui.update()
                # Timer.report()
            else:
                run = False
                pass

    ps.set_user_callback(update)
    ps.show()