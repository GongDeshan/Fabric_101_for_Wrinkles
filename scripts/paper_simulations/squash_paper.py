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

scale_factor = 0.0
delta_scale = 0.005

shrink_frames = 160

cloth_obj_path = str(scene_dir / "meshes" / "squared_28cm_v100_center.obj")
table_obj_path = str(scene_dir / "meshes" / "ball.obj")

output_dir = str(scene_dir / "output" / "square_paper" / "meshes")

workspace = str(scene_dir / "output" / "square_paper" / "workspace")
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
config['contact']['d_hat'] = 0.001
config['contact']['enable'] = True
config['contact']['friction']['enable'] = False
# config['gravity'] = [[0.0], [0.0], [0.0]]
# config['newton']['velocity_tol'] = 1e-4

print(config)
scene = Scene(config)

# setup the scene
t = Transform.Identity()
t.scale(1.0)
io = SimplicialComplexIO(t)

cloth_mesh = io.read(cloth_obj_path)
label_surface(cloth_mesh)

nlstckfhps = StVKNonlinearFrictionHardeningPlasticShell()
nlstckfhps.apply_to(cloth_mesh, K11, K12, K22, K33_0, K33_1, hardening_k33=Hardening_K33, friction_k33=Fric_K33,
                  thres_uv=Thres_uv, anchor_strain_uv=0.0, yield_uv=Yield_UV, plastic_strain_uv=0.0,
                  mass_density=MASS_DENSITY, thickness=THICKNESS)

dsfhpb = DiscreteShellFrictionHardeningPlasticBending()
dsfhpb.apply_to(
    cloth_mesh, bending_stiffness=BENDING_STIFFNESS, hardening_k=Hardening_K,
    yield_theta=YEILD_THETA, plastic_theta=PLASTIC_THETA, friction_k=BENDING_FRICTION,
    thres_theta=THRES_THETA, anchor_theta=ANCHOR_THETA,
)

cloth = scene.objects().create("cloth")
cloth_geo_slot, _ = cloth.geometries().create(cloth_mesh)

# Load cloth mesh rest shape from wavefront uv coordinate
cloth_x_bar = load_mesh_uv(cloth_obj_path)[:,:, np.newaxis] * 1.0
view(cloth_geo_slot.geometry().triangles().find("rest_x"))[:] = cloth_x_bar

# Object Table
table = scene.objects().create("table")
io = SimplicialComplexIO(t)
table_mesh = io.read(table_obj_path)
label_surface(table_mesh)

table_rest_pos = table_mesh.vertices().find(builtin.position).view().copy()

empty = Empty()
empty.apply_to(table_mesh, thickness=0.001)

spc = SoftPositionConstraint()
spc.apply_to(table_mesh, 500.0)  # constraint strength ratio

table_geo, _ = table.geometries().create(table_mesh)

EndFrame = 300

def square_object(info):

    global delta_scale, scale_factor

    geo_slots = info.geo_slots()
    geo = geo_slots[0].geometry()

    is_constrainted = geo.vertices().find(builtin.is_constrained)
    view(is_constrainted)[:] = 1

    aim_position = geo.vertices().find(builtin.aim_position)
    aim_position_view = view(aim_position)[:]

    if world.frame() < shrink_frames:
        scale_factor -= delta_scale
    else:
        scale_factor += delta_scale

    for i, aim_pos in enumerate(aim_position_view):
        aim_position_view[i] = table_rest_pos[i] * (1.0 + scale_factor)

scene.animator().insert(table, square_object)

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

                if world.frame() == 1:
                    # Stop updating the anchor/plastic strains from tri properties from this step
                    view(cloth_geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 0

                save_mesh_sequence(world.frame())

                sgui.update()
                # Timer.report()
            else:
                run = False
                pass

    ps.set_user_callback(update)
    ps.show()