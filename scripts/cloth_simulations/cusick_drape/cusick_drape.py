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

lift_frame = 300
lift_motion_frames = 100
wm_rot_alter_direction = [100, 150, 200, 250]
wm_rot_stop_frame = 300
wm_rot_start_frame = 50

Rot_Theta = 0.0
Delta_Theta = 0.03

scene_dir = Path(__file__).resolve().parent
cloth_obj_path = str(scene_dir / "meshes" / "drape_circle.obj")

output_dir = str(scene_dir / "runs" / "drape_01" / "meshes")

workspace = str(scene_dir / "runs" / "drape_01" / "workspace")
os.makedirs(workspace, exist_ok=True)

engine = Engine('cuda', workspace)
world = World(engine)

# Constants
Up_Direction = "warp"
Up_Direction = "weft"

Warp_Stiffness = 3387343.740058945
Weft_Stiffness = 3062594.2045124643

K11 = Weft_Stiffness if Up_Direction == "warp" else Warp_Stiffness # Init Warp
K22 = Warp_Stiffness if Up_Direction == "warp" else Weft_Stiffness # Init Warp Stiffness

K12 = 0.0
K33_0 = 4066.9668728850215
K33_1 = 18547.83884261949

Hardening_K33 = 188338.8273688658
Fric_K33 = 145409.12660630906

Thres_uv = 0.0025026123105050435
Yield_UV = 0.024616595352738743

# Thres_uv = 1.0
# Yield_UV = 1.0

MASS_DENSITY = 203
THICKNESS = 0.0003

# Fitted orthotropic bending parameters for Drape 01.
BENDING_STIFFNESS_WARP = 126.79989100312847
BENDING_STIFFNESS_WEFT = 172.2776830725535
# BENDING_STIFFNESS_TORSION = sqrt(BENDING_STIFFNESS_WARP * BENDING_STIFFNESS_WEFT) / 3.0
BENDING_STIFFNESS_TORSION = 0.0
BENDING_POISSON = 0.0
Hardening_K_WARP = 55.41993404324105
Hardening_K_WEFT = 61.73375344917775
YEILD_THETA_WARP = 0.9997496005974875
YEILD_THETA_WEFT = 1.0073696441357394
PLASTIC_THETA = 0.0

BENDING_FRICTION_WARP = 97.83147778261547
BENDING_FRICTION_WEFT = 138.1102029023938
ANCHOR_THETA = 0.0
THRES_THETA_WARP = 0.06330667521608432
THRES_THETA_WEFT = 0.06978301932313209

kb_11 = BENDING_STIFFNESS_WEFT if Up_Direction == "warp" else BENDING_STIFFNESS_WARP
kb_22 = BENDING_STIFFNESS_WARP if Up_Direction == "warp" else BENDING_STIFFNESS_WEFT
kb_12 = 0.0
kb_33 = K33_0 * THICKNESS**3 / 12
Yeild_Theta_Warp = YEILD_THETA_WEFT if Up_Direction == "warp" else YEILD_THETA_WARP
Yeild_Theta_Weft = YEILD_THETA_WARP if Up_Direction == "warp" else YEILD_THETA_WEFT

fric_kb_11 = BENDING_FRICTION_WEFT if Up_Direction == "warp" else BENDING_FRICTION_WARP
fric_kb_22 = BENDING_FRICTION_WARP if Up_Direction == "warp" else BENDING_FRICTION_WEFT

Thres_Theta_Warp = THRES_THETA_WEFT if Up_Direction == "warp" else THRES_THETA_WARP
Thres_Theta_Weft = THRES_THETA_WARP if Up_Direction == "warp" else THRES_THETA_WEFT

dt = 0.01
config = Scene.default_config()
config['dt'] = dt
config['contact']['d_hat'] = 0.001
config['contact']['enable'] = False
# config['gravity'] = [[0.0], [0.0], [0.0]]
# config['newton']['velocity_tol'] = 1e-4

cloth_scale = 1.0

print(config)
scene = Scene(config)

# setup the scene
t = Transform.Identity()
t.scale(cloth_scale)
io = SimplicialComplexIO(t)

cloth_mesh = io.read(cloth_obj_path)
label_surface(cloth_mesh)

nlstckfhps = StVKNonlinearFrictionHardeningPlasticShell()
dsfhpob = DiscreteShellFrictionHardeningPlasticOrthotropicBending()

nlstckfhps.apply_to(cloth_mesh, K11, K12, K22, K33_0, K33_1, hardening_k33=Hardening_K33, friction_k33=Fric_K33,
                  thres_uv=Thres_uv, anchor_strain_uv=0.0, yield_uv=Yield_UV, plastic_strain_uv=0.0,
                  mass_density=MASS_DENSITY, thickness=THICKNESS)

dsfhpob.apply_to(cloth_mesh, 
                 bending_stiffness_warp=kb_11,
                 bending_stiffness_weft=kb_22,
                 bending_torsion= kb_33,
                 bending_poisson= kb_12,
                 hardening_k_warp=Hardening_K_WEFT if Up_Direction == "warp" else Hardening_K_WARP,
                 hardening_k_weft=Hardening_K_WARP if Up_Direction == "warp" else Hardening_K_WEFT,
                 yield_theta_warp=Yeild_Theta_Warp,
                 yield_theta_weft=Yeild_Theta_Weft,
                 plastic_theta_warp=0.0,
                 plastic_theta_weft=0.0,
                 friction_k_warp=fric_kb_11,
                 friction_k_weft=fric_kb_22,
                 thres_theta_warp=Thres_Theta_Warp,
                 thres_theta_weft=Thres_Theta_Weft,
                 anchor_theta_warp=0.0,
                 anchor_theta_weft=0.0)

# Constraints
spc = SoftPositionConstraint()
spc.apply_to(cloth_mesh, 1e5)  # constraint strength ratio

cloth = scene.objects().create("cloth")
cloth_geo_slot, _ = cloth.geometries().create(cloth_mesh)

# Load cloth mesh rest shape from wavefront uv coordinate
cloth_x_bar = load_mesh_uv(cloth_obj_path)[:,:, np.newaxis] * cloth_scale
view(cloth_geo_slot.geometry().triangles().find("rest_x"))[:] = cloth_x_bar

cloth_ver_pos = cloth_geo_slot.geometry().vertices().find(builtin.position).view()
cloth_ver_rest_pos = cloth_ver_pos.copy()

print(cloth_ver_rest_pos.shape)

EndFrame = 200

fix_indices = []
num_cloth_verts = cloth_mesh.vertices().size()
fix_indices_list = []

for p in range(num_cloth_verts):
    dist_to_center = np.sqrt(cloth_ver_rest_pos[p][0]**2 + cloth_ver_rest_pos[p][2]**2).item()
    if dist_to_center < 0.09:
        fix_indices_list.append(1)
        print(dist_to_center)
    else:
        fix_indices_list.append(0)

cloth_fixed_verts = cloth_geo_slot.geometry().vertices().find(builtin.is_fixed)
view(cloth_fixed_verts)[:] = np.array(fix_indices_list)

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