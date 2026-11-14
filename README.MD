<div align="center">

# Fabric-101 & Learnable Persistent Wrinkles

**Fabric measurements, microscopy images, and tools for studying persistent wrinkles and learning cloth material parameters.**

[![SIGGRAPH Asia 2026](https://img.shields.io/badge/SIGGRAPH%20Asia-2026-ff69b4?style=for-the-badge)](https://asia.siggraph.org/2026/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.13707-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2609.13707)

<img src="assets/teaser.jpg" width="100%" alt="Real and simulated persistent wrinkles in different fabrics"/>

*Real and simulated persistent wrinkles in different fabrics.*

The fixture shown above is available as [3D-printable OBJ models](assets/Teaser_3D_Print).

</div>

## Overview

This repository accompanies:

> **Learnable Persistent Wrinkle Formation in Cloth Simulation**  
> Deshan Gong, Ningtao Mao, Xiaoyuan Yang, Xinyu Lu, He Wang, Taku Komura  
> SIGGRAPH Asia 2026 · [Paper](https://arxiv.org/abs/2609.13707)

Fabric-101 provides fabric measurements and microscopy images for studying
material behavior. Our simulation model combines internal friction and plasticity
to fit measured loading–unloading behavior and reproduce persistent wrinkles.

The repository includes raw measurements, microscopy images, data-processing
scripts, the simulation backend in `libuipc-diff/`, and parameter-training tools
in `training/`.

[Browse fabrics](FABRICS.md) · [Download catalog CSV](fabric_catalog.csv)

The differentiable simulator is based on the [**uipc**](https://github.com/spiriMirror/libuipc).

# Known Issues & Problems to Solve


- Under axial compression, thin-shell buckling can lead to different equilibrium shapes and variations in the predicted net force. Tighter Newton solver tolerances can reduce numerical variation in some cases, at the cost of longer simulation times, but do not guarantee a unique deformation path.
- After fixing some bugs, some parameters, e.g., initialization and learning rate, are different from our paper. Please use our default parameters in the code to train.
- Our tests cover moderate deformations, with limited data at larger deformations.
- Fitting bending plasticity from compression curves alone can lead to unrealistically sharp residual wrinkles in simulations of wrinkle-resistant fabrics such as wool and silk.
- Fabric viscosity is omitted in our current work.
- Some fabric data have errors. I will test them again.

## Testing instrument

The measurements in this dataset were collected using the [**Leeds University
Fabric Handle Evaluation System (LUFHES)**](https://litac.leeds.ac.uk/roaches-international-takes-fabric-evaluation-commercial/), developed at the University of Leeds
under the leadership of Professor Ningtao Mao. For the instrument design and measurement method, see Ningtao Mao and
Mark Taylor, *Evaluation apparatus and method*, international patent
application publication [WO 2012/104627 A2](https://patents.google.com/patent/WO2012104627A2/en). 

### Instrument and test setup

The renderings show the LUFHES instrument, a mounted specimen, and the twist and compression setups.

| Instrument | Front view | Mounted fabric | Twist | Compression |
| :---: | :---: | :---: | :---: | :---: |
| <img src="assets/lufhes_render/LUFHES.png" width="135" alt="LUFHES rendering, instrument"/> | <img src="assets/lufhes_render/LUFHES_Front.png" width="135" alt="LUFHES rendering, front view"/> | <img src="assets/lufhes_render/LUFHES_Fabric.png" width="135" alt="LUFHES rendering, mounted fabric"/> | <img src="assets/lufhes_render/LUFHES_Twist.png" width="135" alt="LUFHES rendering, twist"/> | <img src="assets/lufhes_render/LUFHES_Compressed.png" width="135" alt="LUFHES rendering, compression"/> |

The photographs show the instrument, a mounted specimen, and the tests in
measurement order: twist, compression, then stretch.

| LUFHES | Mounted fabric | Twist | Compression | Stretch |
| :---: | :---: | :---: | :---: | :---: |
| <img src="assets/lufhes/LUFHES_Front_View_No_Logo.png" width="135" alt="Front view of the LUFHES fabric testing instrument"/> | <img src="assets/lufhes/Fabric.png" width="135" alt="Fabric specimen mounted in the LUFHES fixture"/> | <img src="assets/lufhes/Twist.png" width="135" alt="LUFHES twist test configuration"/> | <img src="assets/lufhes/Compress.png" width="135" alt="Fabric buckling during a LUFHES compression test"/> | <img src="assets/lufhes/Stretch.png" width="135" alt="LUFHES stretch test configuration"/> |

We thank [**Roaches**](https://www.roaches.co.uk) and their team for their technical support.  
For more information about the instrument, please [**contact Roaches**](https://www.roaches.co.uk/textile-testing-equipment/sentire-fabric-handle/).

## Repository structure

```text
Fabric_101_for_Wrinkles/
├── data/                 # Raw measurement files and test settings
│   └── Cotton/
│       └── 001/          # Test/session index
│           ├── head.dat
│           ├── warp/
│           │   ├── s1/   # Specimen index
│           │   │   ├── tarevalues.csv
│           │   │   ├── compression/compression.dat
│           │   │   ├── friction/friction.dat
│           │   │   └── twist/twist.dat
│           │   ├── s2/
│           │   └── s3/
│           └── weft/
├── photos/               # Microscopy images by material
│   ├── Cotton/
│   ├── Linen/
│   ├── Wool/
│   ├── PET/
│   └── Silk/
├── libuipc-diff/          # Simulation backend source and Python bindings
├── scripts/
│   ├── compress_twist.py # Compression and twist processing
│   ├── friction.py       # Stretch loading-curve processing
│   ├── paper_simulations/ # Paper examples
│   └── cloth_simulations/ # Cloth examples
├── training/             # Interactive and batch parameter fitting
│   ├── stretch/
│   ├── shear/
│   └── bending/
│       ├── isotropic/
│       └── orthotropic/
├── processed_data/       # Processed example curves (NumPy)
├── plots/                # Saved plots
├── assets/               # README illustrations
├── FABRICS.md            # Category-sorted fabric information
└── fabric_catalog.csv    # Fabric properties (CSV)
```

Keep the raw directory structure intact: the scripts
construct paths from material, session, direction, and specimen names.

The five principal example materials used in our paper are:

| Material | Raw-data directory | Photo directory |
|---|---|---|
| Cotton | `data/Cotton/` | `photos/Cotton/` |
| Linen | `data/Linen/` | `photos/Linen/` |
| Wool | `data/Wool/` | `photos/Wool/` |
| PET | `data/PET/` | `photos/PET/` |
| Silk | `data/Silk/` | `photos/Silk/` |

## Fabric catalog

Browse the [fabric catalog](FABRICS.md) or download the
[CSV](fabric_catalog.csv). The catalog lists 103 fabric IDs, including paper,
grouped by category and sorted by fabric ID.

| CSV field | Description |
|---|---|
| `fabric_id` | Fabric ID |
| `category` | Material category |
| `name` | English name |
| `thickness_mm` | Thickness (mm) |
| `areal_density_g_m2` | Areal density (g/m^2, gsm) |

## Microscopy photographs

`photos/` contains microscope photographs taken at two magnification settings,
identified by the image filenames `1` and `5`. Fabric photographs can be compared
with the UK two-pound coin and ruler images below, taken at the corresponding
settings, to judge the scale of the visible fabric structure.

| UK two-pound coin - setting 1 | UK two-pound coin - setting 5 | Ruler - setting 1 | Ruler - setting 5 |
| :---: | :---: | :---: | :---: |
| <img src="photos/Two_Pounds/1.png" width="180" alt="UK two-pound coin - setting 1"/> | <img src="photos/Two_Pounds/5.png" width="180" alt="UK two-pound coin - setting 5"/> | <img src="photos/Ruler/1.png" width="180" alt="Ruler - setting 1"/> | <img src="photos/Ruler/5.png" width="180" alt="Ruler - setting 5"/> |

**Yarns Direction in Fabric Photos:**

- **Vertical: warp\wale\machine direction.**
- **Horizontal: weft\course\cross-machine direction.**

## Raw measurement files

The `.dat` files contain test settings and measurements. `head.dat` contains
material and session information; `tarevalues.csv` contains the recorded tare
values. Keep these files with their measurement folders.

| File | Processing entry point | Processed curve axes |
|---|---|---|
| `compression.dat` | `compress_twist.py`, `test_type = "compression"` | Compression displacement (mm), force (N) |
| `twist.dat` | `compress_twist.py`, `test_type = "twist"` | Rotation angle (degrees), torque (N·m) |
| `friction.dat` | `friction.py` | Drag/stretch displacement (mm), load force (N) |

`friction.dat` contains the drag/stretch test. The processing script extracts
its initial tensile loading curve for stretch training.

## Process the measurements

### Requirements

Use Python with NumPy, SciPy, and Matplotlib:

```bash
python -m pip install numpy scipy matplotlib
```

Run the scripts from the repository root:

```bash
python scripts/compress_twist.py
python scripts/friction.py
```

The scripts use settings near the top of each file rather than command-line
arguments. Set the material, session, direction, and test type before running.
If a variable has multiple active assignments, the last assignment takes effect;
leave only the intended selection active.

### Compression or twisting

In `scripts/compress_twist.py`, select, for example:

```python
fabric_name = "Cotton"
test_idx = "001"
test_direction = "warp"       # "warp" or "weft"
test_type = "compression"     # "compression" or "twist"
PLOT_TYPE = "AVG"             # "AVG" or "ALL"
CANCEL_TARE = False
IS_SAVE_PLOT = False
```

By default, the script reads `s1`, `s2`, and `s3`. For materials with fewer
specimens, update `FILE_PATHS` and `TARE_VALUE_PATH` to include only the available
specimens. The script separates loading and unloading, smooths and resamples
each segment, and averages the selected specimens. In `AVG` mode, it writes:

```text
processed_data/<fabric_name>_<direction>_<test_type>.npy
```

Use `PLOT_TYPE = "ALL"` to inspect individual curves. `SELECTED_SAMPLE` controls
which specimens are displayed in that mode; the averaging inputs are set by
`FILE_PATHS` and `TARE_VALUE_PATH`.

### Friction/stretch loading data

In `scripts/friction.py`, select the material, session, direction, and specimen
paths in `FILE_PATH`. The script writes the extracted loading curve to:

```text
processed_data/<fabric_name>_<direction>_friction.npy
```

### Outputs

Figures are displayed with Matplotlib. Processed curves are saved under
`processed_data/`; files with the same name are overwritten. `IS_SAVE_PLOT`
controls plot saving in `compress_twist.py`, separately from NumPy export.

## Building and installing the simulation backend

Follow the [Windows build and installation guide](https://spirimirror.github.io/libuipc-doc/build_install/windows/),
using the bundled [`libuipc-diff/`](libuipc-diff) as the source directory.

### Tested Windows environments

| Component | RTX 3070 Ti laptop | RTX 5070 Ti desktop | RTX 5090 desktop |
|---|---|---|---|
| Operating system | Windows, x64 | Windows, x64 | Windows, x64 |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU | NVIDIA GeForce RTX 5070 Ti | NVIDIA GeForce RTX 5090 |
| NVIDIA driver | 617.14 | 617.14 | 616.64 |
| CUDA Toolkit used for the build | 12.4, `nvcc` 12.4.99 | 12.8, `nvcc` 12.8.61 | 12.9 Update 1, `nvcc` 12.9.86 |
| Visual Studio | Visual Studio 2022 Community | Visual Studio 2022 Professional | Visual Studio 2022 Professional |
| MSVC toolset | v143 (`14.43.34808`) | v143 (`14.43.34808`) | v143 (`14.44.35207`) |
| CMake used for configuration | 3.31.6 | 4.0.0 | 4.2.1 |
| Python | Conda Python 3.13.2, x64 | Conda Python 3.11.17, x64 | Conda Python 3.11.18, x64 |
| pybind11 installed in the Python environment | 2.13.6 | 3.1.0 | 3.1.0 |

Compilation succeeded with the CUDA/MSVC combinations listed in the table.

## Parameter training

After installing the backend, install the frontend dependencies in the same
Python environment:

```powershell
python -m pip install numpy torch polyscope matplotlib
```

The training scripts open a Polyscope window. Select the data, set the material
properties and optimizer settings, click **Apply settings and reset**, then
**optim**. The exact reset-button label varies by workflow.

The window shows loss, parameters, epoch time, and predicted curves against
ground truth (GT). Select historical epochs to compare curves.
**Pause optimization** pauses between epochs.

**Save training records** is on by default; turning it off makes the panel dark
red. Bending best parameters are saved even when detailed logging is off.
GUI records do not include Adam state for resuming training.

Density inputs use **kg/m^3** and thickness inputs use **m**. Convert catalog
areal density to volumetric density using
`density = areal_density_g_m2 / (1000 * thickness_m)`.

### Stretch

From the repository root:

```powershell
cd training/stretch
python stretch_model.py
```

This fits the axial stretching stiffness (`K22` in simulation coordinates).
The default target is Cotton weft. Choose a supplied dataset or enter a custom
NumPy path, then set density, thickness, initial stiffness, learning rate, and
epoch limit. Enter density and thickness for the selected material.

The target array must have shape `(2, 31)`: 31 uniformly spaced, strictly
increasing displacements in mm, starting at zero, and corresponding forces in N.
The simulation performs 30 steps over that displacement range.

Adam optimizes `U`, with physical stiffness `P = U^2`. The learning rate therefore
controls the update in `U`. **Run & Stop** runs a forward simulation.

Records are saved to `training/stretch/logs/gui_training_<timestamp>/`, including
settings, per-epoch loss and curves, `best_result.npz`, and a summary. When saving
is off, those epochs are omitted. 

### Shear

The shear model uses the original imposed-angle stiffness correction in Python.
Run `python train.py` from `training/shear`. Training uses two stages:

| Stage | Loss frames (inclusive) | Trainable parameters |
| --- | --- | --- |
| 1 | 0–40 | `k33_0`, `k33_1`, friction stiffness, threshold |
| 2 | 0–60 | All six, including hardening and yield |

Targets require at least 61 angle/torque samples, following
`0 -> 5 -> 0 -> 5` degrees over the first 61 samples (20 steps per branch).

Stage 2 starts from the stage-1 best parameters and resets Adam. Set learning
rates and epoch limits in the panel. Each stage runs for up to 500 epochs, with
plateau early stopping.

Batch example, from `training/shear`:

```powershell
python train.py --headless-train --fabric 59_Cotton --direction warp --epochs 500
```

The epoch limit applies to each stage. Records are stored under
`logs/gui_staged_<timestamp>/stage1` and `stage2`. The stage-2 best parameters are
restored at completion. 

### Isotropic bending

From the repository root:

```powershell
cd training/bending/isotropic
python train.py
```

This compression workflow uses the supplied slit mesh and fixed stretching/shear
parameters. **Activate bending plasticity (isotropic)** is on by default.
With plasticity on, training proceeds as follows:

| Stage | Initialization | Trainable parameters | Loss frames (inclusive) |
| --- | --- | --- | --- |
| 1: warp | Configured initial values | Stiffness, friction stiffness, threshold | 30–70 |
| 2: warp | Stage-1 best; reset Adam | All five, including hardening and yield | 30–70 and 120–150 |
| 3: weft | Stage-2 warp best; reset Adam | All five | 30–70 and 120–150 |

With plasticity off, all stages optimize only the first three parameters;
hardening is fixed at 50 and yield at 100. The selected mode is saved with the
best parameters for orthotropic initialization.

Stages 2 and 3 simulate continuously to frame 150 and evaluate loss on the 72
selected frames. Set initial parameters, individual learning rates, solver
tolerances, and collision options in the panel. Each stage runs for up to 100
epochs, with plateau early stopping. Shear sliding and plasticity are inactive.

Records are saved under `logs/gui_bending_<timestamp>/`. Best parameters and
curves are saved after each improved forward evaluation, including when detailed
logging is off. Interrupted runs retain their previously saved bests. Keep the
entire run folder for orthotropic training.

Batch examples, from the isotropic folder:

```powershell
python train.py --headless-train --fabric 59_Cotton --epochs 500
python train.py --headless-train --fabric 59_Cotton --epochs 500 --no-plastic
```

The epoch limit applies to each stage. 

### Orthotropic joint bending

Complete the isotropic warp and weft stages first. From the repository root:

```powershell
cd training/bending/orthotropic
python train.py
```

Enter the isotropic run folder and click **Load isotropic warp and weft bests**.
Both best records must come from the same run, with the same plasticity setting.
Keep `run_manifest.json` and its referenced best files together. The original
target data and matching mesh must remain available.
Missing, incompatible, or damaged records produce an error.

Plasticity follows the isotropic log: when on, all ten directional parameters
are optimized; when off, only six stiffness/friction parameters are optimized,
with hardening fixed at 50 and yield at 100 in both directions.

Training minimizes the mean warp/weft MSE using frames 30–70 and 120–150,
with up to 100 epochs and plateau early stopping. Set individual learning rates
and solver options in the panel. Collision is on by default, and additional
history gradients are off.

Results are saved under `logs/gui_joint_<timestamp>/`. Best paired parameters
and curves are saved even when detailed logging is off. Closing the window
also stops its simulation processes.

Batch example, from the orthotropic folder:

```powershell
python train.py --headless-train --isotropic-run "<isotropic run folder>" --epochs 500
```

## Fitted material parameters

### Cotton and Linen

These are the Cotton and Linen parameter sets used in the Sit on Shirt simulation.

Stretch and shear stiffness coefficients are in Pa. Shear thresholds and yield
strains are dimensionless.

| Parameter | Cotton | Linen |
| --- | ---: | ---: |
| Warp stretch stiffness | 5753754.3771 | 7268873.8665 |
| Weft stretch stiffness | 1229799.9644 | 4894094.1377 |
| Shear stiffness coefficient k33_0 | 418811.2270 | 42933.8220 |
| Shear stiffness coefficient k33_1 | 449440.2107 | 51143.0707 |
| Shear friction stiffness | 607111.0712 | 197314.8504 |
| Shear friction threshold | 0.005315771 | 0.001399683 |
| Shear yield strain | 0.022090133 | 0.021238335 |
| Shear plastic hardening | 108828.1354 | 89593.0812 |

Orthotropic bending coefficients are listed as simulator inputs in the physical
warp and weft directions. The friction threshold and yield angle are in radians.

| Parameter | Cotton warp | Cotton weft | Linen warp | Linen weft |
| --- | ---: | ---: | ---: | ---: |
| Bending stiffness | 488.9805 | 187.8062 | 1165.1620 | 606.2312 |
| Bending friction stiffness | 540.7861 | 155.8220 | 1043.0364 | 613.8104 |
| Bending friction threshold | 0.098170933 | 0.062618447 | 0.155133893 | 0.124392230 |
| Bending plastic hardening | 52.0413 | 21.8654 | 44.3990 | 12.1666 |
| Bending yield angle | 1.008076516 | 0.952237515 | 0.991530028 | 0.925513305 |

### Paper

The paper airplane, lantern, and crumpled paper simulations use the same
fitted parameters. Stretch stiffness is shared across both directions,
and bending uses the isotropic model.

| Parameter | Value |
| --- | ---: |
| Density (kg/m³) | 992.875 |
| Thickness (mm) | 0.080 |
| Stretch stiffness, both directions (Pa) | 188180336.2235 |
| Shear stiffness coefficient k33_0 (Pa) | 9874486.7638 |
| Shear stiffness coefficient k33_1 (Pa) | 11047956.0906 |
| Shear friction stiffness (Pa) | 11993285.2858 |
| Shear friction threshold | 0.008529096 |
| Shear yield strain | 0.021834276 |
| Shear plastic hardening (Pa) | 104082.2119 |
| Isotropic bending stiffness | 45680.4050 |
| Bending friction stiffness | 41512.9186 |
| Bending friction threshold (rad) | 0.182255251 |
| Bending plastic hardening | 1003.7626 |
| Bending yield angle (rad) | 0.293206613 |

### Drape 01

The Cusick drape scene uses the following fitted parameters.
Density is 203 kg/m³ and thickness is 0.30 mm.

| Parameter | Value |
| --- | ---: |
| Warp stretch stiffness (Pa) | 3387343.7401 |
| Weft stretch stiffness (Pa) | 3062594.2045 |
| Shear stiffness coefficient k33_0 (Pa) | 4066.9669 |
| Shear stiffness coefficient k33_1 (Pa) | 18547.8388 |
| Shear friction stiffness (Pa) | 145409.1266 |
| Shear friction threshold | 0.002502612 |
| Shear yield strain | 0.024616595 |
| Shear plastic hardening (Pa) | 188338.8274 |

| Orthotropic bending parameter | Warp | Weft |
| --- | ---: | ---: |
| Bending stiffness | 126.7999 | 172.2777 |
| Bending friction stiffness | 97.8315 | 138.1102 |
| Bending friction threshold (rad) | 0.063306675 | 0.069783019 |
| Bending plastic hardening | 55.4199 | 61.7338 |
| Bending yield angle (rad) | 0.999749601 | 1.007369644 |

## Simulation

Run the commands below from the repository root. The scenes load their bundled meshes and motion data.

### Cloth simulations

| Sit on Shirt | Washing Cloth | Sit with Pants | Wrap Legs Pants |
| :---: | :---: | :---: | :---: |
| <img src="assets/Simulation_Figure/Compressed_State.png" width="175" alt="Sit on Shirt simulation"/> | <img src="assets/Simulation_Figure/Washing.png" width="175" alt="Washing Cloth simulation"/> | <img src="assets/Simulation_Figure/Cotton_Sit_High_Res.png" width="175" alt="Sit with Pants simulation"/> | <img src="assets/Simulation_Figure/Cotton_Rolled_State.png" width="175" alt="Wrap Legs Pants simulation"/> |

| Scene | Command |
|---|---|
| Sit on Shirt | `python scripts/cloth_simulations/sit_on_shirt/sit_on_shirt.py --fabric cotton` |
| Washing Cloth | `python scripts/cloth_simulations/washing_cloth/washing_cloth.py --fabric cotton` |
| Sit with Pants | `python scripts/cloth_simulations/sit_with_pants/sit_with_pants.py --fabric cotton` |
| Wrap Legs Pants | `python scripts/cloth_simulations/wrap_legs_pants/wrap_legs_pants.py --fabric cotton` |

Set `--fabric` to `cotton`, `linen`, `wool`, `pet`, or `silk`. In Polyscope, use **Run / Pause** for playback, **One Step** to advance a frame, and **Reset** to return to the initial state. Each scene saves its OBJ sequence and run configuration in its own `runs/` directory.

### Paper simulations

| Paper airplane | Paper lantern | Crumpled paper |
| :---: | :---: | :---: |
| <img src="assets/Simulation_Figure/Folded_Paper_Plane.png" width="225" alt="Paper airplane simulation"/> | <img src="assets/Simulation_Figure/Lamp_Finished_Light_On.png" width="225" alt="Paper lantern simulation"/> | <img src="assets/Simulation_Figure/Squash_Paper.png" width="225" alt="Crumpled paper simulation"/> |

| Scene | Command | Output under `scripts/paper_simulations/output/` |
|---|---|---|
| Fold a paper airplane | `python scripts/paper_simulations/fold_plane.py` | `fold_plane/` |
| Shape a paper lantern | `python scripts/paper_simulations/make_lamp_sphere.py` | `make_lamp_sphere/` |
| Crumple a sheet of paper | `python scripts/paper_simulations/squash_paper.py` | `square_paper/` |

The scenes use the fitted Paper parameters. In Polyscope, **Run & Stop** starts from frame 0 or pauses a running scene. **Stop & Continue** pauses or resumes from the current frame. **One Step** advances a single frame. Continuous playback exports OBJ files to `meshes/`, with solver state in `workspace/`.

#### Fold and throw an airplane

Run either command and click **Play** in Polyscope. The wind scenes require the bundled backend with TriangleWind.

| Material | Command |
|---|---|
| Paper | `python scripts/paper_simulations/fold_throw_paper_plane_tailwind_copy.py` |
| Cotton | `python scripts/paper_simulations/fold_throw_cotton_plane_tailwind_copy.py` |

### Cusick drape

Run the circular drape example with the fitted Drape 01 parameters.

```powershell
python scripts/cloth_simulations/cusick_drape/cusick_drape.py
```

In Polyscope, click **Run & Stop** to start. The scene exports OBJ frames to `scripts/cloth_simulations/cusick_drape/runs/drape_01/meshes/`.

## License

Original project software contributions are licensed under **AGPL-3.0-only**
([LICENSE](LICENSE)). Project-provided research data, microscopy photographs,
processed curves, and the fabric catalog are licensed under **CC BY-NC 4.0**
([LICENSE-DATA](LICENSE-DATA)): noncommercial use with attribution.

The data restriction does not apply to the software. Upstream and third-party
code retains its existing licenses. See [LICENSING.md](LICENSING.md) for scope
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices,
including the bundled NVTX terms that still require redistribution review.
