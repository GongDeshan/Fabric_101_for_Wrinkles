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

manipulate_frames = 50
manipuate_start_frame = 1

return_frames = 48

cloth_obj_path = str(scene_dir / "meshes" / "lamp.obj")
cloth_bend_obj_path = str(scene_dir / "meshes" / "lamp_bend.obj")


output_dir = str(scene_dir / "output" / "make_lamp_sphere" / "meshes")

workspace = str(scene_dir / "output" / "make_lamp_sphere" / "workspace")
os.makedirs(workspace, exist_ok=True)

engine = Engine('cuda', workspace)
world = World(engine)

# Paper material fitted on PC2, 2026-11-08.
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
config['contact']['d_hat'] = 0.0001
config['contact']['enable'] = True
config['contact']["friction"]['enable'] = False
# config['gravity'] = [[0.0], [0.0], [0.0]]
# config['newton']['velocity_tol'] = 1e-4

print(config)
scene = Scene(config)

scene.contact_tabular().default_model(0.0, 1e9)
garment_elem = scene.contact_tabular().create("garment")
obs_part_one_elem = scene.contact_tabular().create("obs_1")
obs_part_two_elem = scene.contact_tabular().create("obs_2")

scene.contact_tabular().insert(obs_part_one_elem, obs_part_one_elem, 0, 0, False)
scene.contact_tabular().insert(obs_part_two_elem, obs_part_two_elem, 0, 0, False)

mesh_scale = 1.0

# setup the scene
t = Transform.Identity()
t.scale(mesh_scale)
io = SimplicialComplexIO(t)

cloth_mesh = io.read(cloth_obj_path)
label_surface(cloth_mesh)

cloth_bend_mesh = io.read(cloth_bend_obj_path)
bend_vert_pos = cloth_bend_mesh.vertices().find(builtin.position).view()[:].copy()

manipulate_path = np.zeros_like(bend_vert_pos)
stop_vert_pos = np.zeros_like(bend_vert_pos)
nlstckfhps = StVKNonlinearFrictionHardeningPlasticShell()
dsfhpb = DiscreteShellFrictionHardeningPlasticBending()

nlstckfhps.apply_to(cloth_mesh, K11, K12, K22, K33_0, K33_1, hardening_k33=Hardening_K33, friction_k33=Fric_K33,
                  thres_uv=Thres_uv, anchor_strain_uv=0.0, yield_uv=Yield_UV, plastic_strain_uv=0.0,
                  mass_density=MASS_DENSITY, thickness=THICKNESS)

dsfhpb.apply_to(cloth_mesh, bending_stiffness=BENDING_STIFFNESS, hardening_k=Hardening_K, yield_theta=YEILD_THETA, 
                plastic_theta=PLASTIC_THETA, friction_k=BENDING_FRICTION, thres_theta=THRES_THETA, anchor_theta=ANCHOR_THETA)

# Constraints
spc = SoftPositionConstraint()
spc.apply_to(cloth_mesh, 50000)  # constraint strength ratio
garment_elem.apply_to(cloth_mesh)

cloth = scene.objects().create("cloth")
cloth_geo_slot, _ = cloth.geometries().create(cloth_mesh)

num_cloth_vert = cloth_mesh.vertices().size()

# Load cloth mesh rest shape from wavefront uv coordinate
cloth_x_bar = load_mesh_uv(cloth_obj_path)[:,:, np.newaxis] * mesh_scale
view(cloth_geo_slot.geometry().triangles().find("rest_x"))[:] = cloth_x_bar

EndFrame = 100

cloth_ver_pos = cloth_geo_slot.geometry().vertices().find(builtin.position).view()
cloth_ver_rest_pos = cloth_ver_pos.copy()

fix_indices = []

for p in cloth_ver_rest_pos:
    if p[1] > 0.21 or p[1] < -0.21 or p[1] == 0.0 or abs(abs(p[1]) - 0.106) < 0.001:
        fix_indices.append(1)
    else:
        fix_indices.append(0)

def cloth_manipulate(info):
    
    geo_slots = info.geo_slots()
    geo = geo_slots[0].geometry()

    global manipulate_path, stop_vert_pos

    if world.frame() == manipuate_start_frame:

        cloth_current_pos = geo.vertices().find(builtin.position)
        cloth_current_pos_view = cloth_current_pos.view()[:]

        aim_position = geo.vertices().find(builtin.aim_position)
        aim_position_view = view(aim_position)[:]

        for i in range(aim_position_view.shape[0]):
            aim_position_view[i] = cloth_current_pos_view[i]

        manipulate_path = (bend_vert_pos - cloth_current_pos_view) / manipulate_frames
        stop_vert_pos = cloth_current_pos_view.copy()

    is_constrainted = geo.vertices().find(builtin.is_constrained)
    view(is_constrainted)[:] = np.array(fix_indices)

    if world.frame() > manipuate_start_frame and world.frame() < (manipuate_start_frame + manipulate_frames):

        cloth_current_pos = geo.vertices().find(builtin.position)
        cloth_current_pos_view = cloth_current_pos.view()[:]

        aim_position = geo.vertices().find(builtin.aim_position)
        aim_position_view = view(aim_position)[:]

        for i in range(aim_position_view.shape[0]):
            aim_position_view[i] = stop_vert_pos[i] + manipulate_path[i] * (world.frame() - manipuate_start_frame)

        is_constrainted = geo.vertices().find(builtin.is_constrained)
        view(is_constrainted)[:] = 1

    if world.frame() >= (manipuate_start_frame + manipulate_frames) and world.frame() < (manipulate_frames + return_frames):

        if world.frame() == (manipuate_start_frame + manipulate_frames):

            cloth_current_pos = geo.vertices().find(builtin.position)
            cloth_current_pos_view = cloth_current_pos.view()[:]

            stop_vert_pos = cloth_current_pos_view.copy()

        aim_position = geo.vertices().find(builtin.aim_position)
        aim_position_view = view(aim_position)[:]

        for i in range(aim_position_view.shape[0]):
            aim_position_view[i] = stop_vert_pos[i] - manipulate_path[i] * (world.frame() - manipuate_start_frame - manipulate_frames)

scene.animator().insert(cloth, cloth_manipulate)

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