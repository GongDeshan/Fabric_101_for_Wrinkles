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

Install the CUDA backend and the Python dependencies in the same environment:

```powershell
python -m pip install numpy torch polyscope matplotlib
```

From the repository root, use `python training/train.py --fabric <material> --stage <stage>`.
Choose `cotton`, `linen`, `wool`, `pet`, or `silk` as the material.

```powershell
python training/train.py --fabric cotton --stage stretch --direction warp
python training/train.py --fabric cotton --stage shear
python training/train.py --fabric cotton --stage bending
python training/train.py --fabric cotton --stage joint
```

Use `--direction weft` for the other stretch direction. For Cotton and Linen,
fit stretch and shear first, then fit isotropic bending with those values
fixed. Load the saved isotropic run in the joint panel.

For Wool, `shear` runs the staged warp fit and `bending` runs the full
no-bending-plasticity workflow using the archived shear fit. For PET and Silk,
`bending` runs the joint fit from the bundled isotropic bests. Their training
settings and reference fits are included with the scripts.

## Fitted material parameters

### Cotton, Linen, Wool, PET, and Silk

| Parameter | Cotton | Linen | Wool | PET | Silk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Density (kg/m3) | 430 | 503 | 590 | 1000 | 410 |
| Thickness (mm) | 0.40 | 0.30 | 0.30 | 0.10 | 0.30 |
| Warp stretch stiffness (Pa) | 5753754.3771 | 7268873.8665 | 3368709.2254 | 7131122.6556 | 2333543.1573 |
| Weft stretch stiffness (Pa) | 1229799.9644 | 4894094.1377 | 2448119.5590 | 11186885.7799 | 1794466.0650 |
| Shear stiffness coefficient k33_0 (Pa) | 418811.2270 | 42933.8220 | 236633.2179 | 227048.7928 | 25364.5250 |
| Shear stiffness coefficient k33_1 (Pa) | 449440.2107 | 51143.0707 | 1370382.7490 | 327524.7409 | 24971.7106 |
| Shear friction stiffness (Pa) | 607111.0712 | 197314.8504 | 310143.1569 | 666146.0542 | 167369.0520 |
| Shear friction threshold | 0.005315771 | 0.001399683 | 0.003679236 | 0.005004784 | 0.001446049 |
| Shear yield strain | 0.022090133 | 0.021238335 | 0.002583266 | 0.020459324 | 0.020019438 |
| Shear plastic hardening (Pa) | 108828.1354 | 89593.0812 | 1174590.6922 | 76694.0874 | 70260.4084 |

Orthotropic bending inputs are listed by physical fabric direction.

| Material | Direction | Bending stiffness | Friction stiffness | Friction threshold (rad) | Plastic hardening | Yield angle (rad) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cotton | Warp | 488.9805 | 540.7861 | 0.098170933 | 52.0413 | 1.008076516 |
| Cotton | Weft | 187.8062 | 155.8220 | 0.062618447 | 21.8654 | 0.952237515 |
| Linen | Warp | 1165.1620 | 1043.0364 | 0.155133893 | 44.3990 | 0.991530028 |
| Linen | Weft | 606.2312 | 613.8104 | 0.124392230 | 12.1666 | 0.925513305 |
| Wool | Warp | 370.7885 | 118.0779 | 0.077464770 | 50 | 3.15 |
| Wool | Weft | 170.8004 | 41.7732 | 0.055399366 | 50 | 3.15 |
| PET | Warp | 2743.8785 | 2852.9286 | 0.291435259 | 89.3376 | 2.179243428 |
| PET | Weft | 499.9323 | 624.1867 | 0.125999176 | 50 | 3.15 |
| Silk | Warp | 203.4537 | 257.0839 | 0.232387225 | 50 | 3.15 |
| Silk | Weft | 60.8402 | 230.2581 | 0.252466627 | 50 | 3.15 |

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

Run either command and click **Play** in Polyscope.

| Material | Command |
|---|---|
| Paper | `python scripts/paper_simulations/fold_throw_paper_plane.py` |
| Cotton | `python scripts/paper_simulations/fold_throw_cotton_plane.py` |

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
