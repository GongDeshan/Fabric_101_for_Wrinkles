import json
import argparse
import copy
import datetime
import os
import sys
import tempfile
import numpy as np
from math import sqrt
import polyscope as ps
from polyscope import imgui
import torch
from torch import optim

from uipc import view
from uipc import Logger, Timer
from uipc import Vector3, Vector2, Matrix4x4, Matrix4x4i, Transform, Quaternion, AngleAxis
from uipc import builtin
from uipc.core import *
from uipc.geometry import *
from uipc.constitution import SoftPositionConstraint
from uipc.constitution import StVKNonlinearFrictionHardeningPlasticShell, DiscreteShellFrictionHardeningPlasticBending, DiscreteShellFrictionHardeningPlasticOrthotropicBending
from uipc.unit import MPa, GPa, kPa
from uipc.gui import *
from diff_sim_friction_plastic_ortho_bending_normalized_training_copy import *
from orthotropic_normalized_gates_training_copy import effective_directional_gates

from load_mesh import load_mesh_uv, edge_bias_angle

from asset_dir import AssetDir


_cli = argparse.ArgumentParser(add_help=True)
_cli.add_argument('--user-style-synthetic', action='store_true',
                  help='Isolated synthetic recovery: soft membrane preset, square parameters and original DiffSimLR; keep per-frame ReLU force corrections.')
_cli.add_argument('--bending-history-autograd', action='store_true',
                  help='Replay bending return-mapping histories with autograd; validate against CUDA records.')
_cli.add_argument("--headless-train", action="store_true",
                  help="Run bending optimization immediately instead of waiting for the GUI button.")
_cli.add_argument("--epochs", type=int, help="Override NUM_EPOCH for this run.")
_cli.add_argument('--training-contact', action='store_true',
                  help='Enable forward collision and full contact adjoint for measured-data training.')
_cli.add_argument('--training-end-frame', type=int,
                  help='Stop the rollout earlier without changing the loading animation.')
_cli.add_argument('--training-monitor', action='store_true',
                  help='Log Newton iterations, physical gradients, and save each new best checkpoint.')
_cli.add_argument(
    "--learning-rate", type=float, metavar="LR",
    help="Override the phase-default Adam learning rate for this run.",
)
_cli.add_argument(
    "--loss-frame-range", type=int, nargs=2, metavar=("START", "END"),
    help=("Override curriculum loss frames with the half-open physical-frame "
          "range [START, END), e.g. 30 50 uses frames 30 through 49."),
)
_cli.add_argument(
    "--fabric", choices=("59_Cotton", "60_Linen", "61_Wool", "62_pet",
                         "63_Silk", "187_Drape 1"),
    help="Material to train; required for headless training and gate reports.",
)
_cli.add_argument(
    "--stage", choices=("warp", "weft"),
    help=("Direction-specific bending-identification stage. Only the five "
          "matching orthotropic parameters are updated; the other five are fixed."),
)
_cli.add_argument(
    "--init-result", metavar="RESULT_NPZ",
    help=("Optional result .npz from the preceding stage. Its ten learned "
          "parameters become this stage's initial/fixed values."),
)
_cli.add_argument(
    "--mechanism-phase",
    choices=("stiffness", "friction", "plastic", "refine", "legacy-all"),
    default="refine",
    help=("Select the curriculum phase and its informative frames. "
          "'refine' uses the first complete loop and second loading, but "
          "does not duplicate the second unloading."),
)
_cli.add_argument(
    "--gate-report-only", action="store_true",
    help=("Run one forward rollout and print bending friction/plastic angle "
          "quantiles for choosing threshold/yield bounds; do not optimize."),
)
_cli.add_argument(
    "--net-force-only", action="store_true",
    help=("Disable the simulator adjoint contribution. The optimization then "
          "uses only the explicit derivative of net_force with respect to P."),
)
_cli.add_argument(
    "--net-force-terms", choices=("all", "bending-total", "bending-friction"),
    default="all",
    help=("Diagnostic force composition used in --net-force-only mode: "
          "'bending-friction' retains only the bending-friction reaction; "
          "'bending-total' adds elastic bending; 'all' additionally includes "
          "the stretching/shear reaction."),
)
_cli.add_argument(
    "--active-parameter", action="append", default=[], metavar="NAME",
    help=("Diagnostic override for the optimization mask. Repeat to optimize "
          "exactly the named physical parameters, e.g. --active-parameter "
          "friction_warp. The normal stage/phase mask is used when omitted."),
)
_cli.add_argument(
    "--frozen-rollout-friction-residual", action="store_true",
    help=("Strict bending-friction gradient diagnostic: simulate once, freeze "
          "the trajectory and all history variables, subtract the frozen "
          "non-friction force from the experimental GT, and optimize only the "
           "explicit bending-friction force against that residual."),
)
_cli.add_argument(
    "--synthetic-friction-target", type=float, metavar="VALUE",
    help=("Generate a synthetic full-force target with the selected stage's "
          "orthotropic bending-friction stiffness set to VALUE, restore the "
          "requested initial parameters, and then identify that one friction "
          "parameter through complete rollouts."),
)
_cli.add_argument(
    "--skip-synthetic-sweep", action="store_true",
    help=("Skip the fixed six-point synthetic diagnostic sweep. The same-"
          "parameter repeat/reset check is still performed."),
)
_cli.add_argument(
    "--guarded-optimization", action="store_true",
    help=("After every Adam step, evaluate the candidate with a complete "
          "rollout. Reject a materially worse candidate, clear Adam momentum, "
          "reduce the learning rate, and stop after a no-improvement patience."),
)
_cli.add_argument(
    "--early-stop-patience", type=int, default=0, metavar="N",
    help=("Stop after N consecutive evaluated epochs without a meaningful "
          "new best complete-rollout loss. Zero disables this for ordinary "
          "optimization; guarded optimization keeps its legacy patience."),
)
_cli.add_argument(
    "--early-stop-relative-delta", type=float, default=1.0e-4,
    metavar="FRACTION",
    help=("Minimum relative best-loss reduction counted as improvement by "
          "early stopping (default: 1e-4)."),
)
_cli.add_argument(
    "--early-stop-window", type=int, default=10, metavar="N",
    help=("Use the median loss over the latest N evaluated epochs for ordinary "
          "early stopping. This suppresses isolated buckling/solver outliers "
          "(default: 10)."),
)
_cli.add_argument(
    "--early-stop-min-epochs", type=int, default=40, metavar="N",
    help=("Do not stop ordinary optimization before N evaluated epochs, even "
          "when its rolling loss has plateaued (default: 40)."),
)
_cli.add_argument(
    "--guard-relative-tolerance", type=float, default=0.10, metavar="FRACTION",
    help=("Maximum temporary relative complete-rollout loss increase accepted "
          "by guarded optimization. The default 0.10 permits crossing small "
          "buckling branch barriers while the best iterate is still retained."),
)
_cli.add_argument(
    "--coarse-search-points", type=int, default=0, metavar="N",
    help=("Before Adam, perform one coordinate-wise complete-rollout search "
          "over every active parameter using N logarithmically spaced values. "
          "Zero (the default) disables coarse initialization."),
)
_cli.add_argument(
    "--coarse-search-min-factor", type=float, default=0.5, metavar="FACTOR",
    help="Smallest multiplier of each active parameter in coarse search.",
)
_cli.add_argument(
    "--coarse-search-max-factor", type=float, default=2.0, metavar="FACTOR",
    help="Largest multiplier of each active parameter in coarse search.",
)
_cli.add_argument(
    "--pure-elastic-stiffness", action="store_true",
    help=("For a stiffness-only diagnostic, hold both bending friction and "
          "plastic gates inactive using thresholds above observed gate maxima."),
)
_cli.add_argument(
    "--enable-shear-hysteresis", action="store_true",
    help=("Use the learned shear friction/plastic thresholds. By default, "
          "bending identification locks shear hysteresis to decouple it."),
)
_cli.add_argument(
    "--repeat-eval", type=int, metavar="N",
    help=("Replay the identical parameter vector N times without updating it. "
          "Use this to test reset/solver determinism before optimization."),
)
_cli.add_argument(
    "--imperfection-amplitude", type=float, default=0.0, metavar="METERS",
    help=("Apply a deterministic low-frequency normal displacement to the "
          "initial cloth mesh before scene construction. Zero (default) "
          "preserves the original geometry; 5e-5 means 0.05 mm."),
)
_cli.add_argument(
    "--initial-parameter", action="append", default=[], metavar="NAME=VALUE",
    help=("Override one physical initial bending parameter after loading any "
          "checkpoint; repeat this option for multiple parameters."),
)
_cli.add_argument(
    "--parameter-fd", metavar="NAME",
    help=("Run a complete-rollout central finite-difference check for one "
          "physical bending parameter, then exit."),
)
_cli.add_argument(
    "--fd-epsilon", type=float, metavar="EPS",
    help=("Physical perturbation used by --parameter-fd. Typical values are "
          "1.0 for stiffness/friction and 0.001 for angles."),
)
_cli.add_argument(
    "--save-mesh-sequence", action="store_true",
    help=("Export the final replayed simulation trajectory as OBJ meshes. "
          "Training exports only the restored best-parameter rollout."),
)
_cli.add_argument(
    "--mesh-save-stride", type=int, default=1, metavar="N",
    help="When exporting meshes, save every N-th frame (default: every frame).",
)
_cli.add_argument(
    "--mesh-output-dir", metavar="DIR",
    help=("Optional OBJ output directory. By default a unique timestamped "
          "directory is created under animation_save_path."),
)
_cli.add_argument('--stable-contact-repeat', action='store_true',
                  help='Repeat-eval only: contact on, Newton 0.0005, linear tol 1e-5; no backward.')
_cli.add_argument('--contact-direct-training', action='store_true',
                  help='Approximate training: collision on, simulator adjoint off, all net_force terms, and strict full-rollout guarded updates.')
_cli.add_argument('--stop-contact-adjoint', action='store_true',
                  help='Experimental: collision on, material adjoint retained, contact derivatives excluded in backward only. Requires patched CUDA backend.')
_cli.add_argument('--quick-compression-check', action='store_true',
                  help='Short optimization diagnostic: first 50 simulation frames, loss 25..50, one directional kb, no per-step guard rollout. Not a gradient correctness test.')
_cli.add_argument('--quick-solver-tolerances', type=float, nargs=2, metavar=('NEWTON', 'LINEAR'),
                  help='Override solver tolerances ONLY for --quick-compression-check. Omit to keep existing strict settings.')
_cli.add_argument('--solver-tolerances', type=float, nargs=2, metavar=('NEWTON', 'LINEAR'),
                  help=('Override Newton velocity and linear-system tolerances for any training phase. '
                        'This keeps the full phase trajectory (for example friction loading-unloading) '
                        'while using the faster approximate solve.'))
_cli.add_argument('--quick-tolerance-check', type=float, nargs=2, metavar=('NEWTON', 'LINEAR'),
                  help='Forward-only first-loading repeat check at specified solver tolerances; requires repeat-eval/result and stiffness phase.')
_cli.add_argument('--repeat-result', help='Save repeat-eval force curves and MSE to this NPZ.')
_cli.add_argument('--protocol-snapshot-dir', help='Diagnostic only: save each evaluated trajectory, histories, force and protocol for mode comparison.')
_cli.add_argument('--contact-gradient-audit', metavar='PARAMETER',
                  help='Isolate fixed-state autograd/FD versus full contact-rollout FD; no simulator backward.')
_cli.add_argument('--audit-relative-steps', type=float, nargs='+', default=[0.01, 0.003, 0.001],
                  help='Relative physical parameter perturbations for contact gradient audit.')
_cli.add_argument('--audit-synthetic-friction-target', type=float,
                  help='Contact force audit only: generate simulated GT at this friction stiffness.')
_cli.add_argument('--audit-disable-bending-plastic', action='store_true',
                  help='Contact force audit only: force both bending yields to 100; retain friction/contact.')
_cli.add_argument('--audit-contact-off', action='store_true',
                  help='Exclusive force gradient audit: disable forward collision for a matched control.')
_cli.add_argument('--audit-soft-constraint-strength', type=float,
                  help='Force gradient audit only: override soft constraint strength ratio (default 200).')
_cli.add_argument('--audit-adjoint', action='store_true',
                  help='Also execute implemented simulator backward; requires rebuilt CUDA empty-range fix.')
_cli.add_argument('--audit-skip-fd', action='store_true',
                  help='With --audit-adjoint, perform only direct/adjoint smoke checks, no FD rollouts.')
_cli.add_argument('--audit-position-frames', type=int, nargs='+',
                  help='Position-only adjoint/FD audit at these rollout horizons; bypass net_force and GT loss.')
_cli.add_argument('--position-joint-target', metavar='NPZ',
                  help='Opt-in short synthetic position-training diagnostic using a saved audited trajectory.')
_cli.add_argument('--short-force-gradient-audit', action='store_true',
                  help='Opt-in production net_force gradient check on a short controlled rollout.')
_cli.add_argument('--allow-long-force-audit', action='store_true',
                  help='Explicitly allow force-gradient diagnostics up to 200 frames; does not change training defaults.')
_cli.add_argument('--short-force-gradient-reference', metavar='JSON',
                  help='Reuse an identical completed short-force center/adjoint; FD rollouts remain fresh.')
_cli.add_argument('--short-force-joint', action='store_true',
                  help='Opt-in short joint Adam diagnostic of the production force objective.')
_cli.add_argument('--short-force-joint-epochs', type=int, default=6)
_cli.add_argument('--short-force-joint-lr', type=float, default=0.001)
_cli.add_argument('--short-force-direction-source', metavar='JSON',
                  help='Opt-in directional FD at epochs 3 and 4 of a saved short force joint diagnostic.')
_cli.add_argument('--short-force-direction-scales', type=float, nargs='+', default=[0.01])
_cli.add_argument('--position-joint-epochs', type=int, default=6)
_cli.add_argument('--position-joint-lr', type=float, default=0.01)
_cli.add_argument('--position-joint-direction-check', action='store_true',
                  help='Diagnose the initial Adam direction with fresh-rollout FD; no training.')
_cli.add_argument('--position-joint-direction-scales', type=float, nargs='+', default=[0.1, 0.01])
_cli.add_argument('--position-audit-contact', choices=['on', 'off'], default=None,
                  help='Position audit only: controlled contact ablation, not a training option.')
_cli.add_argument('--position-audit-newton-tol', type=float, default=None)
_cli.add_argument('--position-audit-linear-tol', type=float, default=None)
_cli.add_argument('--position-audit-relative-tol', type=float, default=None,
                  help='Position audit only: Newton relative stopping; 0 disables the shortcut, omitted preserves backend default.')
_cli.add_argument('--position-audit-info-log', action='store_true',
                  help='Position audit only: log Newton iterations and contact diagnostics.')
_cli.add_argument('--audit-fixture-force', action='store_true',
                  help='Forward-only: export soft-constraint reaction and raw trajectory for force audit.')
_cli.add_argument('--allow-contact-force-audit', action='store_true',
                  help='Explicit collision-on force-gradient diagnostic only; no training defaults changed.')
_cli_args, _unknown_cli_args = _cli.parse_known_args()
if _cli_args.allow_contact_force_audit and (
        not _cli_args.short_force_gradient_audit or _cli_args.short_force_joint
        or _cli_args.short_force_direction_source or _cli_args.position_audit_contact != 'on'):
    _cli.error('--allow-contact-force-audit requires exclusive collision-on force-gradient diagnostic')
if _cli_args.short_force_gradient_reference and not _cli_args.short_force_gradient_audit:
    _cli.error('Short force gradient reference requires --short-force-gradient-audit')
if _cli_args.allow_long_force_audit and (
        not _cli_args.short_force_gradient_audit or _cli_args.short_force_joint
        or _cli_args.short_force_direction_source):
    _cli.error('--allow-long-force-audit requires exclusive force-gradient diagnostic mode')
if (_cli_args.short_force_gradient_audit or _cli_args.short_force_joint
        or _cli_args.short_force_direction_source):
    _force_audit_max_horizon = 200 if _cli_args.allow_long_force_audit else 25
    if (not _cli_args.audit_position_frames or len(_cli_args.audit_position_frames) != 1
            or _cli_args.audit_position_frames[0] > _force_audit_max_horizon or not _cli_args.audit_adjoint
            or _cli_args.position_joint_target or (_cli_args.position_audit_contact != 'off'
                and not _cli_args.allow_contact_force_audit)
            or _cli_args.loss_frame_range is None):
        _cli.error(f'Force audit requires one horizon <= {_force_audit_max_horizon}, adjoint, contact off (or explicit contact audit opt-in) and explicit loss frame range')
if _cli_args.short_force_direction_source:
    if (_cli_args.short_force_joint or _cli_args.short_force_gradient_audit
            or _cli_args.net_force_only or _cli_args.pure_elastic_stiffness
            or any(not np.isfinite(x) or x <= 0 for x in _cli_args.short_force_direction_scales)):
        _cli.error('Short force direction audit requires full adjoint and finite positive scales; do not combine audit modes')
if _cli_args.short_force_joint:
    if (_cli_args.short_force_gradient_audit or _cli_args.net_force_only
            or _cli_args.pure_elastic_stiffness or _cli_args.short_force_joint_epochs < 1
            or not np.isfinite(_cli_args.short_force_joint_lr) or _cli_args.short_force_joint_lr <= 0):
        _cli.error('Short force joint training requires positive epochs/lr, full adjoint and all mechanisms; do not combine audit modes')
if _cli_args.position_joint_direction_check:
    if (not _cli_args.position_joint_target or any(not np.isfinite(x) or x <= 0
            for x in _cli_args.position_joint_direction_scales)):
        _cli.error('Direction check requires joint target and positive finite step scales')
if _cli_args.position_joint_target:
    if (not _cli_args.audit_position_frames or len(_cli_args.audit_position_frames) != 1
            or _cli_args.position_audit_contact != 'off' or not _cli_args.audit_adjoint
            or _cli_args.position_joint_epochs < 1
            or not np.isfinite(_cli_args.position_joint_lr) or _cli_args.position_joint_lr <= 0
            or _cli_args.net_force_only or _cli_args.pure_elastic_stiffness):
        _cli.error('Position joint training requires one position horizon, contact off, adjoint, positive epochs/lr, and all mechanisms')
if _cli_args.quick_solver_tolerances is not None:
    if not _cli_args.quick_compression_check:
        _cli.error('--quick-solver-tolerances requires --quick-compression-check')
    if any(not np.isfinite(v) or v <= 0 for v in _cli_args.quick_solver_tolerances):
        _cli.error('Quick solver tolerances must be finite and positive')
if _cli_args.solver_tolerances is not None:
    if any(not np.isfinite(v) or v <= 0 for v in _cli_args.solver_tolerances):
        _cli.error('Solver tolerances must be finite and positive')
    if _cli_args.quick_solver_tolerances is not None:
        _cli.error('Use only one of --solver-tolerances and --quick-solver-tolerances')
if _cli_args.quick_tolerance_check is not None:
    if (not _cli_args.repeat_eval or not _cli_args.repeat_result
            or _cli_args.mechanism_phase != 'stiffness' or not _cli_args.init_result
            or not _cli_args.stage or not _cli_args.fabric
            or _cli_args.headless_train or _cli_args.contact_gradient_audit
            or _cli_args.stop_contact_adjoint or _cli_args.contact_direct_training
            or _cli_args.quick_compression_check or _cli_args.audit_fixture_force
            or _cli_args.frozen_rollout_friction_residual or _cli_args.synthetic_friction_target is not None
            or _cli_args.save_mesh_sequence or _cli_args.net_force_terms != 'all'):
        _cli.error('Quick tolerance check requires exclusive forward repeat-eval/result, fabric/stage/checkpoint and stiffness phase; no training or mesh export')
    if any(not np.isfinite(v) or v <= 0 for v in _cli_args.quick_tolerance_check):
        _cli.error('Solver tolerances must be positive and finite')
if _cli_args.stop_contact_adjoint:
    _stop_contact_primary_modes = int(_cli_args.headless_train) + int(
        _cli_args.parameter_fd is not None)
    if (_stop_contact_primary_modes != 1 or _cli_args.net_force_only
            or _cli_args.contact_direct_training or _cli_args.contact_gradient_audit
            or _cli_args.repeat_eval
            or _cli_args.gate_report_only or _cli_args.stable_contact_repeat
            or _cli_args.frozen_rollout_friction_residual
            or _cli_args.synthetic_friction_target is not None
            or _cli_args.net_force_terms != 'all'):
        _cli.error('--stop-contact-adjoint requires exactly one of exclusive --headless-train or --parameter-fd; do not combine with other diagnostic modes')
if _cli_args.contact_direct_training:
    if (not _cli_args.headless_train or _cli_args.contact_gradient_audit
            or _cli_args.repeat_eval or _cli_args.parameter_fd
            or _cli_args.gate_report_only or _cli_args.stable_contact_repeat
            or _cli_args.frozen_rollout_friction_residual
            or _cli_args.synthetic_friction_target is not None
            or _cli_args.net_force_terms != 'all'):
        _cli.error('--contact-direct-training requires exclusive --headless-train with all force terms and experimental GT')
    _cli_args.net_force_only = True
    _cli_args.guarded_optimization = True
    _cli_args.guard_relative_tolerance = 0.0
if _cli_args.quick_compression_check:
    if (not _cli_args.stop_contact_adjoint or _cli_args.stage is None
            or _cli_args.mechanism_phase != 'stiffness'
            or _cli_args.active_parameter != [f'kb_{_cli_args.stage}']
            or _cli_args.coarse_search_points != 0 or _cli_args.save_mesh_sequence):
        _cli.error('Quick compression requires --stop-contact-adjoint --stage warp/weft --mechanism-phase stiffness --active-parameter kb_<stage>; no coarse search or mesh export')
    _cli_args.guarded_optimization = False
    if _cli_args.epochs is None:
        _cli_args.epochs = 3
if (_cli_args.position_audit_contact is not None
        or _cli_args.position_audit_newton_tol is not None
        or _cli_args.position_audit_linear_tol is not None
        or _cli_args.position_audit_relative_tol is not None
        or _cli_args.position_audit_info_log) and _cli_args.audit_position_frames is None:
    _cli.error('Position audit overrides require --audit-position-frames')
if (_cli_args.position_audit_relative_tol is not None
        and (not np.isfinite(_cli_args.position_audit_relative_tol)
             or _cli_args.position_audit_relative_tol < 0)):
    _cli.error('Position audit relative tolerance must be finite and nonnegative')
for _audit_tol in (_cli_args.position_audit_newton_tol, _cli_args.position_audit_linear_tol):
    if _audit_tol is not None and (not np.isfinite(_audit_tol) or _audit_tol <= 0):
        _cli.error('Position audit tolerances must be finite and positive')
if (_cli_args.audit_adjoint or _cli_args.audit_skip_fd) and not _cli_args.contact_gradient_audit:
    _cli.error('--audit-adjoint/--audit-skip-fd require --contact-gradient-audit')
if _cli_args.audit_skip_fd and not _cli_args.audit_adjoint:
    _cli.error('--audit-skip-fd requires --audit-adjoint')
if _cli_args.audit_position_frames is not None:
    if (not _cli_args.contact_gradient_audit or not _cli_args.audit_adjoint
            or _cli_args.audit_skip_fd or any(f < 1 for f in _cli_args.audit_position_frames)):
        _cli.error('Position audit needs --contact-gradient-audit --audit-adjoint and positive horizons; no --audit-skip-fd')
if _cli_args.contact_gradient_audit:
    if (not _cli_args.fabric or not _cli_args.init_result or not _cli_args.stage
            or _cli_args.headless_train or _cli_args.repeat_eval or _cli_args.parameter_fd
            or _cli_args.gate_report_only or _cli_args.stable_contact_repeat
            or _cli_args.repeat_result or _cli_args.frozen_rollout_friction_residual
            or _cli_args.synthetic_friction_target or _cli_args.net_force_terms != 'all'):
        _cli.error('Contact gradient audit requires fabric/stage/init-result and exclusive diagnostic mode with all force terms')
    if any(not np.isfinite(v) or v <= 0 for v in _cli_args.audit_relative_steps):
        _cli.error('Audit relative steps must be finite and positive')
    _cli_args.net_force_only = True
if (_cli_args.audit_synthetic_friction_target is not None or _cli_args.audit_disable_bending_plastic):
    if (not _cli_args.contact_gradient_audit or not _cli_args.audit_adjoint
            or _cli_args.audit_position_frames is not None or _cli_args.short_force_gradient_audit):
        _cli.error('Synthetic/plastic-disabled force audit requires exclusive contact-gradient-audit + audit-adjoint')
    if _cli_args.audit_synthetic_friction_target is not None and (
            not _cli_args.contact_gradient_audit.startswith('friction_')
            or not np.isfinite(_cli_args.audit_synthetic_friction_target)
            or _cli_args.audit_synthetic_friction_target <= 0):
        _cli.error('Synthetic friction target must be positive and audit a friction stiffness')
if _cli_args.audit_contact_off and (not _cli_args.contact_gradient_audit
        or _cli_args.audit_position_frames is not None or _cli_args.training_contact):
    _cli.error('--audit-contact-off requires force contact-gradient-audit without training-contact or position audit')
if _cli_args.audit_soft_constraint_strength is not None and (
        not _cli_args.contact_gradient_audit or _cli_args.audit_position_frames is not None
        or not np.isfinite(_cli_args.audit_soft_constraint_strength)
        or _cli_args.audit_soft_constraint_strength <= 0):
    _cli.error('Soft constraint override requires force gradient audit and finite positive strength')
if _cli_args.audit_fixture_force and not (
        _cli_args.stable_contact_repeat and _cli_args.repeat_result
        and _cli_args.repeat_eval == 1):
    _cli.error('--audit-fixture-force requires --stable-contact-repeat --repeat-eval 1 --repeat-result')
if (_cli_args.stable_contact_repeat or _cli_args.repeat_result) and (
        not _cli_args.repeat_eval or _cli_args.headless_train or _cli_args.parameter_fd
        or _cli_args.gate_report_only):
    _cli.error('Contact repeat/result options require exclusive --repeat-eval mode')
if _cli_args.epochs is not None and _cli_args.epochs < 1:
    _cli.error("--epochs must be positive")
if _cli_args.early_stop_patience < 0:
    _cli.error("--early-stop-patience must be non-negative")
if _cli_args.early_stop_window < 1:
    _cli.error("--early-stop-window must be positive")
if _cli_args.early_stop_min_epochs < 0:
    _cli.error("--early-stop-min-epochs must be non-negative")
if (not np.isfinite(_cli_args.early_stop_relative_delta)
        or _cli_args.early_stop_relative_delta < 0.0):
    _cli.error("--early-stop-relative-delta must be finite and non-negative")
if _cli_args.mesh_save_stride < 1:
    _cli.error("--mesh-save-stride must be positive")
if (_cli_args.learning_rate is not None
        and (not np.isfinite(_cli_args.learning_rate)
             or _cli_args.learning_rate <= 0.0)):
    _cli.error("--learning-rate must be a positive finite number")
if _cli_args.repeat_eval is not None and _cli_args.repeat_eval < 1:
    _cli.error("--repeat-eval must be at least 1")
if (not np.isfinite(_cli_args.imperfection_amplitude)
        or _cli_args.imperfection_amplitude < 0.0):
    _cli.error("--imperfection-amplitude must be a finite non-negative value")
if (not np.isfinite(_cli_args.guard_relative_tolerance)
        or _cli_args.guard_relative_tolerance < 0.0):
    _cli.error("--guard-relative-tolerance must be a finite non-negative value")
if _cli_args.coarse_search_points not in (0,) and _cli_args.coarse_search_points < 3:
    _cli.error("--coarse-search-points must be zero or at least 3")
if (not np.isfinite(_cli_args.coarse_search_min_factor)
        or not np.isfinite(_cli_args.coarse_search_max_factor)
        or _cli_args.coarse_search_min_factor <= 0.0
        or _cli_args.coarse_search_max_factor
        <= _cli_args.coarse_search_min_factor):
    _cli.error("coarse-search factors must satisfy 0 < min < max")
if _cli_args.net_force_terms != "all" and not _cli_args.net_force_only:
    _cli.error("--net-force-terms other than 'all' requires --net-force-only")
if _cli_args.frozen_rollout_friction_residual:
    if not _cli_args.net_force_only:
        _cli.error("--frozen-rollout-friction-residual requires --net-force-only")
    if _cli_args.net_force_terms != "bending-friction":
        _cli.error(
            "--frozen-rollout-friction-residual requires "
            "--net-force-terms bending-friction")
if _cli_args.user_style_synthetic:
    if (_cli_args.synthetic_friction_target is None or _cli_args.init_result
            or _cli_args.bending_history_autograd or _cli_args.guarded_optimization
            or _cli_args.net_force_only or _cli_args.training_contact):
        _cli.error('--user-style-synthetic requires a synthetic target, native adjoint, detached recorded histories, no checkpoint/guard/contact')
if _cli_args.synthetic_friction_target is not None:
    if not _cli_args.headless_train:
        _cli.error("--synthetic-friction-target requires --headless-train")
    if _cli_args.stage is None:
        _cli.error("--synthetic-friction-target requires --stage warp or --stage weft")
    if _cli_args.frozen_rollout_friction_residual:
        _cli.error("--synthetic-friction-target is incompatible with --frozen-rollout-friction-residual")
    if _cli_args.net_force_terms != "all":
        _cli.error("--synthetic-friction-target requires --net-force-terms all")
    _synthetic_active_name = f"friction_{_cli_args.stage}"
    if _cli_args.active_parameter != [_synthetic_active_name]:
        _cli.error(
            "Synthetic friction recovery must isolate exactly one parameter; "
            f"pass --active-parameter {_synthetic_active_name}")
    if (not np.isfinite(_cli_args.synthetic_friction_target)
            or _cli_args.synthetic_friction_target <= 0.0):
        _cli.error("--synthetic-friction-target must be a positive finite number")
if (_cli_args.headless_train or _cli_args.gate_report_only or _cli_args.repeat_eval
        or _cli_args.parameter_fd) and _cli_args.fabric is None:
    _cli.error("--fabric is required for headless training, gate reports, repeat evaluation, and finite differences")
if (_cli_args.parameter_fd is None) != (_cli_args.fd_epsilon is None):
    _cli.error("--parameter-fd and --fd-epsilon must be provided together")
if _cli_args.fd_epsilon is not None and (not np.isfinite(_cli_args.fd_epsilon) or _cli_args.fd_epsilon <= 0.0):
    _cli.error("--fd-epsilon must be a positive finite number")
if _cli_args.pure_elastic_stiffness and _cli_args.mechanism_phase != "stiffness":
    _cli.error("--pure-elastic-stiffness requires --mechanism-phase stiffness")
if _cli_args.pure_elastic_stiffness and _cli_args.fabric != "59_Cotton":
    _cli.error("--pure-elastic-stiffness is calibrated only for 59_Cotton; run a gate report for other fabrics first")






























tested_fabric_type = _cli_args.fabric or "187_Drape 1"


fabric_type = tested_fabric_type.split("_")[-1].lower()


class _Tee:
    """Mirror console diagnostics into a durable per-run log file."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        if "\n" in data:
            self.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


run_type = "init"


Up_Direction = _cli_args.stage or "warp"

_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = os.path.join(
    _log_dir, f"{fabric_type}_bend_{Up_Direction}_{_log_stamp}.log")
_log_file = open(_log_path, "w", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f"Training log: {_log_path}")
if _cli_args.stage:
    print(f"Bending identification stage: {_cli_args.stage}")
if _cli_args.init_result:
    print(f"Initializing from previous stage: {_cli_args.init_result}")

_FABRIC_STRETCH_SHEAR_PRESETS = {
    "59_Cotton": dict(
        warp=5753754.3771, weft=1229799.9644, density=430.0, thickness=0.0004,
        k33_0=334029.5011, k33_1=1259237.5213, friction=1695927.8589,
        thres=0.0021882522, yield_uv=0.0258776372, hardening=722631.7525),
    "60_Linen": dict(
        warp=7228603.5304, weft=4893617.8892, density=503.0, thickness=0.0003,
        k33_0=28408.9631, k33_1=32443.6754, friction=176153.2763,
        thres=0.0021241809, yield_uv=0.0202538434, hardening=73859.1906),
    "61_Wool": dict(
        warp=3427594.1520, weft=2486771.7833, density=590.0, thickness=0.0003,
        k33_0=171355.3944, k33_1=1146718.4832, friction=431119.6343,
        thres=0.0036142537, yield_uv=0.0273776760, hardening=424829.3265),
    "62_pet": dict(
        warp=7129746.1581, weft=11180626.2428, density=1000.0, thickness=0.0001,
        k33_0=111333.8131, k33_1=789550.0115, friction=571503.5113,
        thres=0.0081331147, yield_uv=0.0276715419, hardening=636764.3220),
    "63_Silk": dict(
        warp=2334613.4180, weft=1795268.7842, density=410.0, thickness=0.0003,
        k33_0=26178.3819, k33_1=29147.9289, friction=153566.9181,
        thres=0.0015487866, yield_uv=0.0163570211, hardening=40075.0182),
}

_stretch_shear = _FABRIC_STRETCH_SHEAR_PRESETS.get(
    tested_fabric_type,
    dict(warp=3043588.1339, weft=2765151.0849, density=203.0, thickness=0.0003,
         k33_0=16693.5329, k33_1=100.0 * kPa, friction=360019.4933,
         thres=1.0, yield_uv=1.0, hardening=10.0 * kPa),
)
if _cli_args.user_style_synthetic:
    _stretch_shear = dict(warp=300000.0, weft=200000.0,
        density=249.0, thickness=0.0007, k33_0=100000.0, k33_1=100000.0,
        friction=100000.0, hardening=70000.0, thres=0.0018, yield_uv=0.0022)
print("Fixed stretch/shear presets (revision stretch + new twist learning):")
for _preset_name, _preset in _FABRIC_STRETCH_SHEAR_PRESETS.items():
    print(
        f"  {_preset_name}: stretch(warp={_preset['warp']:.7g}, "
        f"weft={_preset['weft']:.7g}), density={_preset['density']:g}, "
        f"thickness={_preset['thickness']:.7g}; "
        f"shear(k33_0={_preset['k33_0']:.7g}, k33_1={_preset['k33_1']:.7g}, "
        f"friction={_preset['friction']:.7g}, thres={_preset['thres']:.7g}, "
        f"yield={_preset['yield_uv']:.7g}, hardening={_preset['hardening']:.7g})")
print(f"Selected fixed preset for {tested_fabric_type}: {_stretch_shear}")
Warp_Stiffness = _stretch_shear["warp"]
Weft_Stiffness = _stretch_shear["weft"]
K11 = Weft_Stiffness if Up_Direction == "warp" else Warp_Stiffness
K22 = Warp_Stiffness if Up_Direction == "warp" else Weft_Stiffness
K12 = 0.0
K33_0 = _stretch_shear["k33_0"]
K33_1 = _stretch_shear["k33_1"]
Hardening_K33 = _stretch_shear["hardening"]
Fric_K33 = _stretch_shear["friction"]
Thres_uv = _stretch_shear["thres"]
Yield_UV = _stretch_shear["yield_uv"]
if not _cli_args.enable_shear_hysteresis:
    Thres_uv = 100.0
    Yield_UV = 100.0
    print("Shear hysteresis locked for bending identification: "
          "Thres_uv=Yield_UV=100.0 (friction_k33 retained as elastic stiffness).")
MASS_DENSITY = _stretch_shear["density"]
THICKNESS = _stretch_shear["thickness"]





BENDING_STIFFNESS_WARP = 600.0
BENDING_STIFFNESS_WEFT = 600.0
BENDING_STIFFNESS_TORSION = sqrt(BENDING_STIFFNESS_WARP * BENDING_STIFFNESS_WEFT) / 3.0
BENDING_STIFFNESS_TORSION = 0.0
BENDING_POISSON = 0.0
Hardening_K_WARP = 50.0
Hardening_K_WEFT = 50.0
YEILD_THETA_WARP = 1.0
YEILD_THETA_WEFT = 1.0
PLASTIC_THETA = 0.0

BENDING_FRICTION_WARP = 400.0
BENDING_FRICTION_WEFT = 400.0
ANCHOR_THETA = 0.0
THRES_THETA_WARP = 0.1
THRES_THETA_WEFT = 0.1

if _cli_args.user_style_synthetic:
    BENDING_STIFFNESS_WARP = BENDING_STIFFNESS_WEFT = 200.0
    Hardening_K_WARP = Hardening_K_WEFT = 100.0
    YEILD_THETA_WARP = YEILD_THETA_WEFT = 0.5
    THRES_THETA_WARP = THRES_THETA_WEFT = 0.08
    BENDING_FRICTION_WARP = BENDING_FRICTION_WEFT = 300.0
    if _cli_args.stage == 'warp':
        BENDING_FRICTION_WARP = 400.0
    else:
        BENDING_FRICTION_WEFT = 400.0

kb_11 = BENDING_STIFFNESS_WEFT if Up_Direction == "warp" else BENDING_STIFFNESS_WARP
kb_22 = BENDING_STIFFNESS_WARP if Up_Direction == "warp" else BENDING_STIFFNESS_WEFT
kb_12 = 0.0
kb_33 = 0.0  
Hardening_K_Model_Warp = Hardening_K_WEFT if Up_Direction == "warp" else Hardening_K_WARP
Hardening_K_Model_Weft = Hardening_K_WARP if Up_Direction == "warp" else Hardening_K_WEFT
Yeild_Theta_Warp = YEILD_THETA_WEFT if Up_Direction == "warp" else YEILD_THETA_WARP
Yeild_Theta_Weft = YEILD_THETA_WARP if Up_Direction == "warp" else YEILD_THETA_WEFT

fric_kb_11 = BENDING_FRICTION_WEFT if Up_Direction == "warp" else BENDING_FRICTION_WARP
fric_kb_22 = BENDING_FRICTION_WARP if Up_Direction == "warp" else BENDING_FRICTION_WEFT

Thres_Theta_Warp = THRES_THETA_WEFT if Up_Direction == "warp" else THRES_THETA_WARP
Thres_Theta_Weft = THRES_THETA_WARP if Up_Direction == "warp" else THRES_THETA_WEFT

COMP_DIS = 0.016

LOOP_STEPS = 100
ONE_DIR_ROT_STEPS = LOOP_STEPS / 2
NUM_LOOPS = 2
EndFrame = LOOP_STEPS * NUM_LOOPS
if _cli_args.training_end_frame is not None:
    if not 1 <= _cli_args.training_end_frame <= EndFrame:
        _cli.error('--training-end-frame must be within the original trajectory')
    EndFrame = _cli_args.training_end_frame
if _cli_args.quick_compression_check or _cli_args.quick_tolerance_check is not None:
    EndFrame = 50

_CURRICULUM_FRAME_SEGMENTS = {
    "stiffness": ((25, 51),),
    "friction": ((25, 101),),
    "plastic": ((30, 51), (101, 151)),
    "refine": ((25, 151),),
    "legacy-all": ((25, EndFrame),),
}
_PHASE_PARAMETER_INDICES = {
    "stiffness": (0, 1),
    "friction": (2, 3, 8, 9),
    "plastic": (4, 5, 6, 7),
    "refine": tuple(range(10)),
    "legacy-all": tuple(range(10)),
}
_phase_segments = _CURRICULUM_FRAME_SEGMENTS[_cli_args.mechanism_phase]
_training_frames = np.concatenate(
    [np.arange(start, end, dtype=np.int32) for start, end in _phase_segments])
if _cli_args.loss_frame_range is not None:
    _loss_start, _loss_end = _cli_args.loss_frame_range
    if not 1 <= _loss_start < _loss_end <= EndFrame + 1:
        _cli.error(
            "--loss-frame-range must satisfy "
            f"1 <= START < END <= {EndFrame + 1}")
    _training_frames = np.arange(_loss_start, _loss_end, dtype=np.int32)
    print(
        "Explicit loss-frame override: "
        f"physical frames [{_loss_start}, {_loss_end}) "
        f"({_training_frames.size} samples)")
if _cli_args.quick_compression_check:
    assert _training_frames[0] >= 1 and _training_frames[-1] <= EndFrame
    print(f'QUICK COMPRESSION CHECK: simulate frames 1..50; loss frames '
          f'{_training_frames[0]}..{_training_frames[-1]}; '
          'one kb parameter; no candidate rollout per step. Friction/plastic mechanisms remain active. '
          'This tests optimization progress, NOT exact gradient agreement.')



PARAM_LR = 0.002 if _cli_args.mechanism_phase in ("stiffness", "friction", "refine") else 0.02
if _cli_args.learning_rate is not None:
    PARAM_LR = _cli_args.learning_rate
print(f"Bending phase: {_cli_args.mechanism_phase}; learning rate: {PARAM_LR}")
NUM_EPOCH = 200
if _cli_args.epochs is not None:
    NUM_EPOCH = _cli_args.epochs

TRAIN_START_FRAME = 25
TRAIN_END_FRAME = EndFrame



MESH_PATH = "D:\\Research_4\\libipc\\libuipc-samples\\assets\\sim_data\\trimesh\\LUFHES_uv.obj"





SAVE_MESH_SEQUENCE = _cli_args.save_mesh_sequence
output_dir = (_cli_args.mesh_output_dir or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "animation_save_path",
    f"{fabric_type}_bend_{Up_Direction}_{_log_stamp}"))

with open(f"D:\\Research_4\\LUFHES_process\\Data\\{tested_fabric_type}_{Up_Direction}_compression.npy", "rb") as f:
    LUFEHS_compress_dis_force = np.load(f)
if LUFEHS_compress_dis_force.ndim != 2 or LUFEHS_compress_dis_force.shape[0] < 2:
    raise ValueError(
        f"Expected compression data [displacement, force], got "
        f"{LUFEHS_compress_dis_force.shape}")
if LUFEHS_compress_dis_force.shape[1] < EndFrame:
    raise ValueError(
        f"Compression data has {LUFEHS_compress_dis_force.shape[1]} frames, "
        f"but the simulation requires {EndFrame}")
gt_comp_force_all = torch.from_numpy(LUFEHS_compress_dis_force[1]).to(torch.float64)
gt_comp_force = gt_comp_force_all[torch.from_numpy(_training_frames)]

COMP_DIS = max(LUFEHS_compress_dis_force[0]) / 1000.0

print("Maximum Compress Distance: ", COMP_DIS)
print(
    f"Bending mechanism phase: {_cli_args.mechanism_phase}; "
    f"loss frames={_training_frames.tolist()}; segments={_phase_segments}")

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)
if _cli_args.audit_fixture_force or _cli_args.position_audit_info_log or _cli_args.training_monitor:
    Logger.set_level(Logger.Level.Info)  

_backend_runs = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'logs', 'backend_runs')
os.makedirs(_backend_runs, exist_ok=True)
workspace = None
if _cli_args.contact_gradient_audit:
    _gradient_audit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs',
        'contact_gradient_audit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    os.makedirs(_gradient_audit_dir, exist_ok=False)
    workspace = os.path.join(_gradient_audit_dir, 'backend')
    os.makedirs(workspace, exist_ok=False)
    print(f'Contact gradient audit output: {_gradient_audit_dir}')
if _cli_args.repeat_result:
    _repeat_parent = os.path.dirname(os.path.abspath(_cli_args.repeat_result))
    os.makedirs(_repeat_parent, exist_ok=True)
    workspace = tempfile.mkdtemp(
        prefix=os.path.basename(_cli_args.repeat_result) + '.backend_',
        dir=_repeat_parent)
if workspace is None:
    workspace = tempfile.mkdtemp(prefix='bending_', dir=_backend_runs)
print(f'Backend recovery workspace (unique per run): {workspace}', flush=True)
this_folder = AssetDir.folder(__file__)
tetmesh_path = AssetDir.tetmesh_path()
trimesh_path = AssetDir.trimesh_path()

engine = Engine('cuda', workspace)
if _cli_args.stop_contact_adjoint:
    if os.name != 'nt':
        raise RuntimeError('--stop-contact-adjoint currently verifies the patched backend on Windows only')
    import ctypes
    import hashlib
    from pathlib import Path
    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    _kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    _kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    _dll_handle = _kernel32.GetModuleHandleW('uipc_backend_cuda.dll')
    _dll_path_buf = ctypes.create_unicode_buffer(32768)
    if not _dll_handle or not _kernel32.GetModuleFileNameW(_dll_handle, _dll_path_buf, len(_dll_path_buf)):
        raise RuntimeError('Cannot identify loaded CUDA backend; refusing experimental training')
    _dll_bytes = Path(_dll_path_buf.value).read_bytes()
    if b'[adjoint-contact-stop-v1]' not in _dll_bytes:
        raise RuntimeError('Loaded CUDA DLL lacks contact-stop patch. Rebuild build_new and reinstall uipc first: ' + _dll_path_buf.value)
    print('Verified contact-stop backend:', _dll_path_buf.value,
          'SHA256:', hashlib.sha256(_dll_bytes).hexdigest())
    del _dll_bytes
world = World(engine)

dt = 0.01
config = Scene.default_config()
config["dt"] = dt
config['contact']['d_hat'] = THICKNESS
config['contact']['enable'] = False
config['contact']['friction']['enable'] = False
if (_cli_args.stable_contact_repeat or _cli_args.gate_report_only
        or _cli_args.contact_gradient_audit
        or _cli_args.contact_direct_training or _cli_args.stop_contact_adjoint):
    config['contact']['enable'] = True
    config['newton']['velocity_tol'] = 0.0005
    config['linear_system']['tol_rate'] = 1e-5
if _cli_args.training_contact:
    config['contact']['enable'] = True
    config['diff_sim']['contact_gradient'] = True
if _cli_args.audit_contact_off:
    config['contact']['enable'] = False
    print('Matched force audit: forward collision DISABLED; material adjoint retained.', flush=True)
if _cli_args.quick_tolerance_check is not None:
    config['contact']['enable'] = True
    config['newton']['velocity_tol'], config['linear_system']['tol_rate'] = _cli_args.quick_tolerance_check
if _cli_args.stop_contact_adjoint:
    config['diff_sim']['contact_gradient'] = False
if _cli_args.quick_solver_tolerances is not None:
    config['newton']['velocity_tol'], config['linear_system']['tol_rate'] = _cli_args.quick_solver_tolerances
    print('Quick optimization solver tolerances:', _cli_args.quick_solver_tolerances,
          '(explicit diagnostic override; not a gradient accuracy guarantee)')
if _cli_args.solver_tolerances is not None:
    config['newton']['velocity_tol'], config['linear_system']['tol_rate'] = _cli_args.solver_tolerances
    print('Training solver tolerances:', _cli_args.solver_tolerances,
          '(full trajectory with an explicit faster approximate solve)')
if _cli_args.audit_position_frames is not None:
    if _cli_args.position_audit_contact is not None:
        config['contact']['enable'] = _cli_args.position_audit_contact == 'on'
    if _cli_args.position_audit_newton_tol is not None:
        config['newton']['velocity_tol'] = _cli_args.position_audit_newton_tol
    if _cli_args.position_audit_linear_tol is not None:
        config['linear_system']['tol_rate'] = _cli_args.position_audit_linear_tol
    if _cli_args.position_audit_relative_tol is not None:
        config['newton']['relative_tol'] = _cli_args.position_audit_relative_tol

print(config)
if _cli_args.stop_contact_adjoint:
    print('APPROXIMATE ADJOINT: forward collision ON; material/inertia adjoint ON; '
          'contact H[ii], H[i][i-1], and contact pGpP OFF. net_force direct gradients retained. '
          'Not the exact derivative of the collision-enabled forward simulation.')
if _cli_args.contact_direct_training:
    print('APPROXIMATE TRAINING: collision ON; contact friction OFF; simulator adjoint OFF; '
          'all net_force terms; trajectory/history refreshed per evaluation. '
          'This is NOT a full end-to-end simulator gradient. '
          'Full-rollout guard allows no relative loss increase.')
config["diff_sim"]["orthotropic_bending_history_gradient"] = False
scene = Scene(config)

cloth_obj = scene.objects().create('cloth')
nlstckfhps = StVKNonlinearFrictionHardeningPlasticShell()
dsfhpob = DiscreteShellFrictionHardeningPlasticOrthotropicBending()

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


def apply_deterministic_mesh_imperfection(mesh, amplitude: float) -> None:
    """Add one smooth, fixed normal mode to break exact buckling symmetry."""
    if amplitude == 0.0:
        print("Initial mesh imperfection: disabled")
        return

    position_view = mesh.positions().view()
    positions = np.asarray(position_view).reshape(-1, 3).copy()
    triangles = np.asarray(mesh.triangles().topo().view()).reshape(-1, 3)

    vertex_normals = np.zeros_like(positions)
    edge_01 = positions[triangles[:, 1]] - positions[triangles[:, 0]]
    edge_02 = positions[triangles[:, 2]] - positions[triangles[:, 0]]
    face_normals = np.cross(edge_01, edge_02)
    for corner in range(3):
        np.add.at(vertex_normals, triangles[:, corner], face_normals)
    normal_lengths = np.linalg.norm(vertex_normals, axis=1)
    valid = normal_lengths > np.finfo(np.float64).eps
    vertex_normals[valid] /= normal_lengths[valid, np.newaxis]

    extents = np.ptp(positions, axis=0)
    mode_axes = np.argsort(extents)[-2:]
    envelope = np.ones(positions.shape[0], dtype=np.float64)
    for axis in mode_axes:
        coordinate = positions[:, axis]
        span = extents[axis]
        normalized = ((coordinate - coordinate.min()) / span
                      if span > np.finfo(np.float64).eps
                      else np.zeros_like(coordinate))
        envelope *= np.sin(np.pi * normalized)

    symmetry_axis = 0
    symmetry_span = extents[symmetry_axis]
    symmetry_coordinate = positions[:, symmetry_axis]
    symmetry_normalized = (
        (symmetry_coordinate - symmetry_coordinate.min()) / symmetry_span
        if symmetry_span > np.finfo(np.float64).eps
        else np.zeros_like(symmetry_coordinate))
    envelope *= 0.5 + 0.5 * symmetry_normalized

    envelope_peak = np.max(np.abs(envelope))
    if envelope_peak > np.finfo(np.float64).eps:
        envelope /= envelope_peak

    displacement = amplitude * envelope[:, np.newaxis] * vertex_normals
    positions += displacement
    position_view[:] = positions.reshape(position_view.shape)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    print(
        "Initial mesh imperfection: deterministic asymmetric normal mode; "
        f"requested={amplitude:.12g} m, "
        f"actual max={displacement_norm.max():.12g} m, "
        f"rms={np.sqrt(np.mean(displacement_norm**2)):.12g} m, "
        f"mode_axes={mode_axes.tolist()}, symmetry_axis={symmetry_axis}")


apply_deterministic_mesh_imperfection(
    mesh, _cli_args.imperfection_amplitude)

nlstckfhps.apply_to(mesh, K11, K12, K22, K33_0, K33_1, hardening_k33=Hardening_K33, friction_k33=Fric_K33,
                  thres_uv=Thres_uv, anchor_strain_uv=0.0, yield_uv=Yield_UV, plastic_strain_uv=0.0,
                  mass_density=MASS_DENSITY, thickness=THICKNESS)

dsfhpob.apply_to(mesh, 
                 bending_stiffness_warp=kb_11,
                 bending_stiffness_weft=kb_22,
                 bending_torsion= kb_33,
                 bending_poisson= kb_12,
                 hardening_k_warp=Hardening_K_Model_Warp,
                 hardening_k_weft=Hardening_K_Model_Weft,
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

spc = SoftPositionConstraint()

spc.apply_to(mesh, 200.0 if _cli_args.audit_soft_constraint_strength is None
             else _cli_args.audit_soft_constraint_strength)  # constraint strength ratio


is_fixed = mesh.vertices().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
pos_view = mesh.positions().view()
is_fixed_view[:] = np.where(pos_view[:, 1].reshape(-1) < 0.02, 1, 0)

rest_yield_ori_uvs = mesh.triangles().find("yield_uv").view().copy()

rest_yield_theta_warp_ori = mesh.edges().find("yield_theta_warp").view().copy()
rest_yield_theta_weft_ori = mesh.edges().find("yield_theta_weft").view().copy()

rest_yield_ori_tensor = torch.from_numpy(rest_yield_ori_uvs)
rest_yield_theta_ori_warp_tensor = torch.from_numpy(rest_yield_theta_warp_ori)
rest_yield_theta_ori_weft_tensor = torch.from_numpy(rest_yield_theta_weft_ori)

_MODEL_TO_PHYSICAL_PARAMETER_INDICES = (
    (1, 0, 3, 2, 5, 4, 7, 6, 9, 8)
    if Up_Direction == "warp"
    else tuple(range(10))
)
_diff_bending_attribute_names = (
    "diff/bending_stiffness_warp",
    "diff/bending_stiffness_weft",
    "diff/friction_stiffness_warp",
    "diff/friction_stiffness_weft",
    "diff/hardening_k_warp",
    "diff/hardening_k_weft",
    "diff/yield_theta_ori_warp",
    "diff/yield_theta_ori_weft",
    "diff/thres_theta_warp",
    "diff/thres_theta_weft",
)
_diff_bending_attributes = tuple(zip(
    _diff_bending_attribute_names,
    _MODEL_TO_PHYSICAL_PARAMETER_INDICES,
))
for _attribute_name, _parameter_index in _diff_bending_attributes:
    mesh.edges().create(_attribute_name, -1)







# diff_friction_k: AttributeSlotI32 = mesh.edges().create('diff/friction_stiffness', -1)



print('mesh:', mesh)
geo_slot, _ = cloth_obj.geometries().create(mesh)

edge_topos = geo_slot.geometry().edges().find(builtin.topo).view()
face_topos = geo_slot.geometry().triangles().find(builtin.topo).view()

x_bar = load_mesh_uv(MESH_PATH)[:, :, np.newaxis]
view(geo_slot.geometry().triangles().find("rest_x"))[:] = x_bar

edge_vert_topos, bias_angles = edge_bias_angle(MESH_PATH)
view(geo_slot.geometry().edges().find("bias_angle"))[:] = bias_angles

comp_d = 0.0
compress_dis = COMP_DIS / ONE_DIR_ROT_STEPS
_fixture_audit_frames = {}

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

    if _cli_args.audit_soft_constraint_strength is not None:
        _fixture_audit_frames[int(info.frame())] = dict(
            aim=np.asarray(aim_position_view).reshape(-1,3).copy(),
            selected=(np.asarray(is_constrained_view).reshape(-1) != 0)
                     & (np.asarray(geo.vertices().find('is_fixed').view()).reshape(-1) == 0))
    if _cli_args.audit_fixture_force:
        volume = rest_geo.vertices().find('volume')
        if volume is None:
            raise RuntimeError('Fixture audit requires backend rest vertex volume; no approximate mass fallback')
        density = geo.vertices().find('mass_density')
        if density is None:
            density = geo.meta().find('mass_density')
        if density is None:
            raise RuntimeError('Fixture audit: mass_density attribute missing')
        strength = geo.vertices().find('strength_ratio')
        fixed = geo.vertices().find('is_fixed')
        if strength is None or fixed is None:
            raise RuntimeError('Fixture audit: missing constraint strength or fixed mask')
        _fixture_audit_frames[int(info.frame())] = dict(
            aim=np.asarray(aim_position_view).reshape(-1, 3).copy(),
            mass=(np.asarray(volume.view()).reshape(-1)
                  * np.asarray(density.view()).reshape(-1)),
            strength=np.asarray(strength.view()).reshape(-1).copy(),
            selected=(np.asarray(is_constrained_view).reshape(-1) != 0)
                     & (np.asarray(fixed.view()).reshape(-1) == 0),
            rest=np.asarray(rest_position_view).reshape(-1, 3).copy())

scene.animator().insert(cloth_obj, LUFHES_compress_cloth)



_BENDING_PARAMETER_NAMES = (
    "kb_warp", "kb_weft", "friction_warp", "friction_weft",
    "hardening_warp", "hardening_weft", "yield_warp", "yield_weft",
    "thres_warp", "thres_weft",
)


def _physical_to_model_parameters(P: torch.Tensor) -> torch.Tensor:
    """Map physical (warp,weft) pairs to the current experiment's model slots."""
    indices = torch.as_tensor(
        _MODEL_TO_PHYSICAL_PARAMETER_INDICES, dtype=torch.long, device=P.device)
    return P.index_select(0, indices)

_STAGE_ACTIVE_INDICES = {
    "warp": (0, 2, 4, 6, 8),
    "weft": (1, 3, 5, 7, 9),
}
_active_parameter_mask = torch.zeros(len(_BENDING_PARAMETER_NAMES), dtype=torch.bool)
_direction_indices = (
    _STAGE_ACTIVE_INDICES[_cli_args.stage]
    if _cli_args.stage else tuple(range(len(_BENDING_PARAMETER_NAMES))))
_phase_indices = _PHASE_PARAMETER_INDICES[_cli_args.mechanism_phase]
_active_parameter_mask[list(set(_direction_indices).intersection(_phase_indices))] = True
if _cli_args.active_parameter:
    _active_parameter_mask.zero_()
    for _name in _cli_args.active_parameter:
        if _name not in _BENDING_PARAMETER_NAMES:
            _cli.error(
                f"Unknown --active-parameter {_name!r}. Choose one of: "
                f"{', '.join(_BENDING_PARAMETER_NAMES)}")
        _active_parameter_mask[_BENDING_PARAMETER_NAMES.index(_name)] = True
_active_parameter_names = tuple(
    name for index, name in enumerate(_BENDING_PARAMETER_NAMES)
    if _active_parameter_mask[index])
print(
    f"Active orthotropic parameters for stage={_cli_args.stage or 'both'}, "
    f"phase={_cli_args.mechanism_phase}: {_active_parameter_names}")

_BENDING_PARAMETER_LOWER = np.asarray(
    (10.0, 10.0, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4),
    dtype=np.float64)
_BENDING_PARAMETER_UPPER = np.asarray(
    (5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 2.0, 2.0, 0.5, 0.5),
    dtype=np.float64)


class MyDiffSimParameter(DiffSimParameter):
    def __init__(self, scene, initial, lower, upper):
        self._initial = torch.as_tensor(initial, dtype=torch.float64)
        self._lower = torch.as_tensor(lower, dtype=torch.float64)
        self._upper = torch.as_tensor(upper, dtype=torch.float64)
        if not torch.all(self._upper > self._lower):
            raise ValueError("Every bending parameter upper bound must exceed its lower bound")
        if not torch.all((self._initial > self._lower) & (self._initial < self._upper)):
            raise ValueError("Initial bending parameters must lie strictly inside their bounds")
        super().__init__(scene=scene, size=len(initial))

    def forward(self) -> torch.Tensor:
        P = self._lower + (self._upper - self._lower) * torch.sigmoid(self.U())
        if _cli_args.user_style_synthetic:
            P = self.U().square()
        if _cli_args.pure_elastic_stiffness:
            P = torch.cat((
                P[:2], torch.zeros_like(P[2:4]), P[4:6],
                torch.full_like(P[6:8], 10.0), P[8:],
            ))
        if _cli_args.audit_disable_bending_plastic:
            P = torch.cat((P[:6], torch.full_like(P[6:8], 100.0), P[8:]))
        return P

    def set_physical(self, values) -> None:
        values = torch.as_tensor(values, dtype=torch.float64)
        if _cli_args.user_style_synthetic:
            if not torch.isfinite(values).all() or not (values > 0).all():
                raise ValueError('Square-parameter synthetic test requires positive finite parameters')
            self.U().copy_(values.sqrt())
            return
        unit = (values - self._lower) / (self._upper - self._lower)
        if not torch.isfinite(unit).all() or not ((unit > 0) & (unit < 1)).all():
            raise ValueError("Parameters must be finite and strictly inside bounds; no silent clipping")
        self.U().copy_(torch.logit(unit))

_initial_bending_parameters = np.asarray(
    (BENDING_STIFFNESS_WARP, BENDING_STIFFNESS_WEFT,
     BENDING_FRICTION_WARP, BENDING_FRICTION_WEFT,
     Hardening_K_WARP, Hardening_K_WEFT,
     YEILD_THETA_WARP, YEILD_THETA_WEFT,
     THRES_THETA_WARP, THRES_THETA_WEFT),
    dtype=np.float64)
if _cli_args.init_result:
    with np.load(_cli_args.init_result, allow_pickle=False) as _previous_result:
        _checkpoint_fields = {
            "schema_version", "fabric", "compression_direction",
            "mechanism_phase", "parameter_names", "parameters",
            "parameter_coordinate_system",
        }
        _missing_fields = _checkpoint_fields.difference(_previous_result.files)
        if _missing_fields:
            raise ValueError(
                "The initialization result lacks provenance metadata "
                f"{sorted(_missing_fields)}. Use a result generated by the "
                "current staged-training script.")
        _previous_schema = int(np.asarray(_previous_result["schema_version"]).item())
        _previous_fabric = str(np.asarray(_previous_result["fabric"]).item())
        _previous_direction = str(np.asarray(_previous_result["compression_direction"]).item())
        _previous_phase = str(np.asarray(_previous_result["mechanism_phase"]).item())
        _previous_coordinate_system = str(
            np.asarray(_previous_result["parameter_coordinate_system"]).item())
        _previous_names = tuple(_previous_result["parameter_names"].tolist())
        _previous_parameters = np.asarray(_previous_result["parameters"], dtype=np.float64)
    if _previous_schema != 2:
        raise ValueError(f"Unsupported initialization result schema: {_previous_schema}")
    if _previous_coordinate_system != "physical_warp_weft":
        raise ValueError(
            "Initialization result does not use physical warp/weft parameter "
            f"coordinates: {_previous_coordinate_system!r}")
    if _previous_fabric != tested_fabric_type:
        raise ValueError(
            f"Initialization result belongs to {_previous_fabric}, not "
            f"{tested_fabric_type}.")
    if _cli_args.stage:
        _same_direction = _previous_direction == _cli_args.stage
        _warp_to_weft_bootstrap = (
            _previous_direction == "warp"
            and _previous_phase in ("plastic", "refine")
            and _cli_args.stage == "weft" and _cli_args.mechanism_phase == "stiffness")
        if not (_same_direction or _warp_to_weft_bootstrap):
            raise ValueError(
                "Initialization direction mismatch: current run is "
                f"{_cli_args.stage}/{_cli_args.mechanism_phase}, while the "
                f"checkpoint is {_previous_direction}/{_previous_phase}. "
                "Only same-direction continuation or completed warp "
                "(plastic/refine) -> "
                "weft/stiffness is allowed.")
        _allowed_predecessors = {
            "stiffness": {"stiffness"},  
            "friction": {"stiffness", "friction"},
            "plastic": {"friction", "plastic"},
            "refine": {"plastic", "refine"},
            "legacy-all": {"legacy-all"},
        }
        _quick_refine_probe = (
            _cli_args.quick_compression_check and _same_direction
            and _previous_phase == "refine"
            and _cli_args.mechanism_phase == "stiffness")
        if (_same_direction and not _cli_args.repeat_eval and not _cli_args.contact_gradient_audit
                and not _quick_refine_probe
                and _previous_phase not in _allowed_predecessors[_cli_args.mechanism_phase]):
            raise ValueError(
                f"Invalid phase continuation: {_previous_phase} cannot "
                f"initialize {_cli_args.mechanism_phase}. Expected one of "
                f"{sorted(_allowed_predecessors[_cli_args.mechanism_phase])}.")
        if _quick_refine_probe:
            print("Quick compression checkpoint: allowing same-direction refine -> "
                  "stiffness probe; material, direction, schema and parameter checks retained.")
    if _previous_names != _BENDING_PARAMETER_NAMES:
        raise ValueError(
            "The initialization result has incompatible parameter names: "
            f"{_previous_names}")
    if _previous_parameters.shape != _initial_bending_parameters.shape:
        raise ValueError(
            "The initialization result has incompatible parameter shape: "
            f"{_previous_parameters.shape}")
    _initial_bending_parameters = _previous_parameters

if _cli_args.initial_parameter:
    _initial_bending_parameters = _initial_bending_parameters.copy()
    for _override in _cli_args.initial_parameter:
        if "=" not in _override:
            _cli.error(
                "--initial-parameter must have the form NAME=VALUE, e.g. "
                "friction_warp=450")
        _name, _value_text = _override.split("=", 1)
        _name = _name.strip()
        try:
            _value = float(_value_text)
        except ValueError:
            _cli.error(
                f"Invalid value in --initial-parameter {_override!r}; "
                "VALUE must be a finite number")
        if not np.isfinite(_value):
            _cli.error(
                f"Invalid value in --initial-parameter {_override!r}; "
                "VALUE must be finite")
        if _name not in _BENDING_PARAMETER_NAMES:
            _cli.error(
                f"Unknown bending parameter {_name!r}. Choose one of: "
                f"{', '.join(_BENDING_PARAMETER_NAMES)}")
        _index = _BENDING_PARAMETER_NAMES.index(_name)
        _lower = _BENDING_PARAMETER_LOWER[_index]
        _upper = _BENDING_PARAMETER_UPPER[_index]
        if not _lower < _value < _upper:
            _cli.error(
                f"Initial {_name}={_value:g} must be strictly within "
                f"({_lower:g}, {_upper:g})")
        _initial_bending_parameters[_index] = _value
        print(f"Initial-parameter override: {_name}={_value:g}")

if _cli_args.pure_elastic_stiffness:
    _initial_bending_parameters = _initial_bending_parameters.copy()
    _initial_bending_parameters[6:8] = 1.90  
    _initial_bending_parameters[8:10] = 0.49  
    print("Pure-elastic stiffness diagnostic: friction stiffness is forced to "
          "zero and yield is forced to 10.0 during simulation.")

parm = MyDiffSimParameter(
    scene=scene,
    initial=_initial_bending_parameters,
    lower=_BENDING_PARAMETER_LOWER,
    upper=_BENDING_PARAMETER_UPPER)

with torch.no_grad():
    parm.set_physical(_initial_bending_parameters)
    _initial_model_parameters = _physical_to_model_parameters(
        parm.forward()).detach().cpu().numpy()
    parm.sync()

_bending_edges = geo_slot.geometry().edges()
for _model_index, (_attribute_name, _parameter_index) in enumerate(
        _diff_bending_attributes):
    _registered_index = (
        _parameter_index if _active_parameter_mask[_parameter_index] else -1)
    view(_bending_edges.find(_attribute_name))[:] = _registered_index
    _material_attribute_name = _attribute_name.removeprefix("diff/")
    view(_bending_edges.find(_material_attribute_name))[:] = (
        _initial_model_parameters[_model_index])
    if _material_attribute_name.startswith("yield_theta_ori_"):
        _runtime_yield_name = _material_attribute_name.replace("_ori_", "_")
        view(_bending_edges.find(_runtime_yield_name))[:] = (
            _initial_model_parameters[_model_index])
    print(f"Bending diff mapping: {_attribute_name}={_registered_index}")

rest_yield_theta_warp_ori = _bending_edges.find("yield_theta_warp").view().copy()
rest_yield_theta_weft_ori = _bending_edges.find("yield_theta_weft").view().copy()
rest_yield_theta_ori_warp_tensor = torch.from_numpy(rest_yield_theta_warp_ori)
rest_yield_theta_ori_weft_tensor = torch.from_numpy(rest_yield_theta_weft_ori)

world.init(scene)


rest_plastic_thetas_warp = geo_slot.geometry().edges().find("plastic_theta_warp").view().copy()
rest_plastic_thetas_weft = geo_slot.geometry().edges().find("plastic_theta_weft").view().copy()

rest_anchor_thetas_warp = geo_slot.geometry().edges().find("anchor_theta_warp").view().copy()
rest_anchor_thetas_weft = geo_slot.geometry().edges().find("anchor_theta_weft").view().copy()

thetas_tensor_init = torch.from_numpy(geo_slot.geometry().edges().find("dihedral_theta").view().copy())

plastic_theta_warp_tensor = torch.from_numpy(rest_plastic_thetas_warp)
plastic_theta_weft_tensor = torch.from_numpy(rest_plastic_thetas_weft)


rest_anchor_uvs = geo_slot.geometry().triangles().find("anchor_strain_uv").view().copy()
rest_plastic_uvs = geo_slot.geometry().triangles().find("plastic_strain_uv").view().copy()

node_force_indices = []

egde_rest_lengths = geo_slot.geometry().edges().find("L0").view()
V_bars = geo_slot.geometry().edges().find("V_bar").view()
h_bars = geo_slot.geometry().edges().find("h_bar").view()
bending_edge_stencils = geo_slot.geometry().edges().find("stencil").view()

is_edge_cal_force = np.zeros(mesh.edges().size(), dtype=bool)
is_face_cal_force = np.zeros(mesh.triangles().size(), dtype=bool)

x_init = mesh.vertices().find(builtin.position).view().copy()
_bending_history_initial_positions = np.asarray(x_init).reshape(-1, 3).copy()

for i, node_x in enumerate(x_init):
    if node_x[1] > 0.07:
        node_force_indices.append(i)
        is_edge_cal_force = is_edge_cal_force | np.any(
            np.isin(bending_edge_stencils.reshape(-1, 4), i), axis=1)
        is_face_cal_force = is_face_cal_force | np.any(
            np.isin(face_topos, i), axis=1).squeeze()

is_edge_cal_force &= egde_rest_lengths.reshape(-1) != 0.0

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

bias_angles_list = []
backend_bias = np.asarray(
    geo_slot.geometry().edges().find("bias_angle").view()).reshape(-1)
_bias_edge_topology = np.asarray(
    geo_slot.geometry().edges().find(builtin.topo).view()).reshape(-1, 2)
_bias_edge_lookup = {
    tuple(sorted(map(int, edge))): i
    for i, edge in enumerate(_bias_edge_topology)
}
assert len(_bias_edge_lookup) == len(backend_bias), "Duplicate/missing bias edges"

for e_idx, l0 in enumerate(egde_rest_lengths):

    if l0.item() != 0.0 and np.any(np.isin(edge_cal_force_indices, e_idx)):












        _shared_edge = tuple(sorted(map(int, bending_edge_stencils[e_idx].reshape(-1)[1:3])))
        bias_angles_list.append(float(backend_bias[_bias_edge_lookup[_shared_edge]]))


        valid_idx_tensor = torch.zeros(
            egde_rest_lengths.shape, dtype=torch.double)

        one_BF_index_tensor = torch.zeros(
            (NUM_VERTICES * 3, 12), dtype=torch.double)

        valid_idx_tensor[e_idx] = 1.0

        valid_edge_index.append(valid_idx_tensor)

        one_idx_tensor = torch.zeros((4, NUM_VERTICES), dtype=torch.double)



        force_indices = [0, 1, 2, 3]

        for i in range(4):
            one_idx_tensor[i][bending_edge_stencils[e_idx][i]] = 1.0

            for j in range(3):
                one_BF_index_tensor[bending_edge_stencils[e_idx]
                                    [force_indices[i]] * 3 + j, i * 3 + j] = 1.0





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
print(
    "Reaction-force bending hinges: "
    f"{cloth_valid_indices.shape[0]} / {len(egde_rest_lengths)} edges")

cloth_l0s = torch.mv(cloth_valid_indices,
                     torch.from_numpy(egde_rest_lengths.copy()))
cloth_v_bars = torch.mv(cloth_valid_indices, torch.from_numpy(V_bars.copy()))
cloth_h_bars = torch.mv(cloth_valid_indices, torch.from_numpy(h_bars.copy()))

cloth_bias_angles = torch.tensor(bias_angles_list, dtype=torch.float64) + 1.570796

cloth_select_force_indices = torch.zeros((NUM_VERTICES, 3), dtype=torch.double)

for node_f_idx in node_force_indices:
    cloth_select_force_indices[node_f_idx] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.double)

def dihedral_angle(x0, x1, x2, x3):
    """Native backend stencil order: [opposite, shared, shared, opposite]."""
    e0 = x2 - x1
    e1 = x0 - x1
    e2 = x3 - x1
    e3 = x0 - x2
    e4 = x3 - x2

    n1 = torch.linalg.cross(e0, e1)
    n2 = torch.linalg.cross(e2, e0)

    n1SqNorm = n1.dot(n1)
    n2SqNorm = n2.dot(n2)
    e0norm = torch.norm(e0)

    grad = torch.cat([
        -e0norm / n1SqNorm * n1,
        -e0.dot(e3) / (e0norm * n1SqNorm) * n1 -
        e0.dot(e4) / (e0norm * n2SqNorm) * n2,
        e0.dot(e1) / (e0norm * n1SqNorm) * n1 +
        e0.dot(e2) / (e0norm * n2SqNorm) * n2,
        -e0norm / n2SqNorm * n2])

    dA = torch.atan2(torch.linalg.cross(n2, n1).dot(-e0 / e0norm), n1.dot(n2))

    return dA, grad

def stretch_strain_init(x_init_np):

    x_init = torch.from_numpy(x_init_np)

    xs_torch_stretching = x_init.flatten()

    Dm_w_1 = torch.mv(cloth_face_indices_1, xs_torch_stretching)
    Dm_w_2 = torch.mv(cloth_face_indices_2, xs_torch_stretching)

    Dm_w_l = torch.stack([Dm_w_1, Dm_w_2], 1)

    Dm_w_t = Dm_w_l.view(NUM_FACES, 3, 2)

    F_s = torch.bmm(Dm_w_t, cloth_faces_invDm)

    G = 0.5 * (torch.bmm(F_s.permute(0, 2, 1), F_s) - torch.eye(2, dtype=torch.double).unsqueeze(0)).reshape(-1, 4)

    G = G.T

    strain_uus = G[0]
    strain_vvs = G[3]
    strain_uvs = G[1]

    return strain_uus, strain_vvs, strain_uvs

strain_uus_init, strain_vvs_init, strain_uvs_init = stretch_strain_init(x_init)

def net_force(xs_torch, bending_k_warp, bending_k_weft, bending_k_torsion, bending_poisson, friction_k_warp, friction_k_weft, hardening_k_warp, hardening_k_weft,
              yield_theta_ori_warp, yield_theta_ori_weft, yield_thetas_warp, yield_thetas_weft, hardening_plastic_thetas_warp, hardening_plastic_thetas_weft, 
              plastic_thetas_warp, plastic_thetas_weft, thres_thetas_warp, thres_thetas_weft, anchor_thetas_warp, anchor_thetas_weft,
              K11, K22, K12, K33_0, K33_1, Friction_K33, thres_uvs, anchor_strain_uvs, hardening_k33, yield_uv_ori, yield_uvs, hardening_plastic_uvs, plastic_strain_uvs,
              force_terms="all"):

    e0s = torch.matmul(cloth_e0_indices, xs_torch).squeeze(1)
    e1s = torch.matmul(cloth_e1_indices, xs_torch).squeeze(1)
    e2s = torch.matmul(cloth_e2_indices, xs_torch).squeeze(1)
    e3s = torch.matmul(cloth_e3_indices, xs_torch).squeeze(1)
    e4s = torch.matmul(cloth_e4_indices, xs_torch).squeeze(1)

    e5s = torch.matmul(cloth_e0_n_indices, xs_torch).squeeze(1)
    e6s = torch.matmul(cloth_e2_n_indices, xs_torch).squeeze(1)

    n1s = torch.cross(e0s, e1s, dim=1)
    n2s = torch.cross(e2s, e0s, dim=1)

    n1s_norm = torch.linalg.norm(n1s, dim=1)
    n2s_norm = torch.linalg.norm(n2s, dim=1)

    e0s_norm = torch.linalg.norm(e0s, dim=1)

    n1s_sqnorm = n1s_norm**2
    n2s_sqnorm = n2s_norm**2

    n1s_theta = torch.cross(e0s, e1s, dim=1)
    n2s_theta = torch.cross(e5s, e6s, dim=1)


    n1s_sqnorm_theta = torch.linalg.norm(n1s_theta, dim=1)**2
    n2s_sqnorm_theta = torch.linalg.norm(n2s_theta, dim=1)**2



    n1_dot_n2 = (n1s_theta * n2s_theta).sum(dim=1)


    edge_hat = -e0s / e0s_norm.unsqueeze(1)
    sin_term = (torch.cross(n2s_theta, n1s_theta, dim=1) * edge_hat).sum(dim=1)
    thetas = torch.atan2(sin_term, n1_dot_n2)










    
    grad_part_1 = (-e0s_norm / n1s_sqnorm).unsqueeze(1) * n1s
    grad_part_2 = -(torch.bmm(e0s.unsqueeze(1), e3s.unsqueeze(2)).reshape(-1) / (e0s_norm * n1s_sqnorm)).unsqueeze(1) * n1s - (
        torch.bmm(e0s.unsqueeze(1), e4s.unsqueeze(2)).reshape(-1) / (e0s_norm * n2s_sqnorm)).unsqueeze(1) * n2s
    grad_part_3 = (torch.bmm(e0s.unsqueeze(1), e1s.unsqueeze(2)).reshape(-1) / (e0s_norm * n1s_sqnorm)).unsqueeze(1) * n1s + (
        torch.bmm(e0s.unsqueeze(1), e2s.unsqueeze(2)).reshape(-1) / (e0s_norm * n2s_sqnorm)).unsqueeze(1) * n2s
    grad_part_4 = (-e0s_norm / n2s_sqnorm).unsqueeze(1) * n2s


    grads = torch.concat(
        [grad_part_1, grad_part_2, grad_part_3, grad_part_4], dim=1)


    
    cos_phi = torch.cos(cloth_bias_angles)
    sin_phi = torch.sin(cloth_bias_angles)

    theta_warp = thetas * cos_phi * cos_phi
    theta_weft = thetas * sin_phi * sin_phi
    theta_torsion = thetas * sin_phi * cos_phi



    elastic_theta_warp = theta_warp - torch.mv(cloth_valid_indices, plastic_thetas_warp)
    elastic_theta_weft = theta_weft - torch.mv(cloth_valid_indices, plastic_thetas_weft)






    gates = effective_directional_gates(
        cos_phi, sin_phi,
        torch.mv(cloth_valid_indices, yield_theta_ori_warp),
        torch.mv(cloth_valid_indices, yield_theta_ori_weft),
        torch.mv(cloth_valid_indices, thres_thetas_warp),
        torch.mv(cloth_valid_indices, thres_thetas_weft))
    bending_normalization = gates.normalization
    initial_yield_warp_eff = gates.yield_warp
    initial_yield_weft_eff = gates.yield_weft
    dis_to_yield_warp = torch.abs(elastic_theta_warp) - torch.mv(cloth_valid_indices, yield_thetas_warp)
    dis_to_yield_weft = torch.abs(elastic_theta_weft) - torch.mv(cloth_valid_indices, yield_thetas_weft)

    hardening_plastic_thetas_valid_warp = torch.mv(cloth_valid_indices, hardening_plastic_thetas_warp)
    hardening_plastic_thetas_valid_warp = hardening_plastic_thetas_valid_warp + torch.relu(dis_to_yield_warp) * (bending_k_warp/(hardening_k_warp + bending_k_warp))

    hardening_plastic_thetas_valid_weft = torch.mv(cloth_valid_indices, hardening_plastic_thetas_weft)
    hardening_plastic_thetas_valid_weft = hardening_plastic_thetas_valid_weft + torch.relu(dis_to_yield_weft) * (bending_k_weft/(hardening_k_weft + bending_k_weft))

    new_yield_thetas_warp = initial_yield_warp_eff + hardening_plastic_thetas_valid_warp * (hardening_k_warp/bending_k_warp)
    elastic_theta_current_warp = theta_warp - (torch.mv(cloth_valid_indices, plastic_thetas_warp) + torch.sign(elastic_theta_warp) * torch.relu(torch.abs(elastic_theta_warp) - new_yield_thetas_warp))

    new_yield_thetas_weft = initial_yield_weft_eff + hardening_plastic_thetas_valid_weft * (hardening_k_weft/bending_k_weft)
    elastic_theta_current_weft = theta_weft - (torch.mv(cloth_valid_indices, plastic_thetas_weft) + torch.sign(elastic_theta_weft) * torch.relu(torch.abs(elastic_theta_weft) - new_yield_thetas_weft))

    BF_batch = (-cloth_l0s * (bending_k_warp * 2.0 * elastic_theta_current_warp * cos_phi * cos_phi + 
                              bending_k_weft * 2.0 * elastic_theta_current_weft * sin_phi * sin_phi +
                              bending_poisson * 2.0 * elastic_theta_current_warp * sin_phi * sin_phi + 
                              bending_poisson * 2.0 * elastic_theta_current_weft * cos_phi * cos_phi +
                              bending_k_torsion * 2.0 * theta_torsion * sin_phi * cos_phi) / cloth_h_bars * cloth_v_bars).unsqueeze(1) * grads 
    
    var_theta_warp = theta_warp - torch.mv(cloth_valid_indices, anchor_thetas_warp)
    dis_to_slide_warp = torch.abs(var_theta_warp) - gates.threshold_warp
    var_theta_current_warp = theta_warp - (torch.mv(cloth_valid_indices, anchor_thetas_warp) + torch.sign(var_theta_warp) * torch.relu(dis_to_slide_warp))

    var_theta_weft = theta_weft - torch.mv(cloth_valid_indices, anchor_thetas_weft)
    dis_to_slide_weft = torch.abs(var_theta_weft) - gates.threshold_weft
    var_theta_current_weft = theta_weft - (torch.mv(cloth_valid_indices, anchor_thetas_weft) + torch.sign(var_theta_weft) * torch.relu(dis_to_slide_weft))



    BFric_batch = (
        -2.0 * cloth_l0s / cloth_h_bars * cloth_v_bars
        * (
            friction_k_warp * var_theta_current_warp * cos_phi.square()
            + friction_k_weft * var_theta_current_weft * sin_phi.square()
        )
    ).unsqueeze(1) * grads





    BF_batch = BF_batch * bending_normalization.unsqueeze(1)
    BFric_batch = BFric_batch * bending_normalization.unsqueeze(1)

    Bending_Elastic_Force = torch.bmm(
        cloth_BF_index, BF_batch.unsqueeze(2)).sum(dim=0).squeeze().reshape(-1, 3)
    Bending_Friction_Force = torch.bmm(
        cloth_BF_index, BFric_batch.unsqueeze(2)).sum(dim=0).squeeze().reshape(-1, 3)
    Bending_Force = Bending_Elastic_Force + Bending_Friction_Force










    xs_torch_stretch = xs_torch  


    xs_torch_stretching = xs_torch_stretch.flatten()

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

    strain_uus = G[0] - strain_uus_init
    strain_vvs = G[3] - strain_vvs_init
    strain_uvs = G[1]


    K33 = K33_0 + K33_1 * torch.abs(strain_uvs)

    elastic_strain_uvs = strain_uvs - plastic_strain_uvs
    dis_to_yield_uv = torch.abs(elastic_strain_uvs) - yield_uvs
    hardening_plastic_uvs = hardening_plastic_uvs + torch.relu(dis_to_yield_uv) * (K33 / (hardening_k33 + K33))
    new_yield_uvs = yield_uv_ori + hardening_plastic_uvs * (hardening_k33 / K33)
    current_elastic_strain_uvs = strain_uvs - (plastic_strain_uvs + torch.sign(elastic_strain_uvs) * torch.relu(torch.abs(elastic_strain_uvs) - new_yield_uvs))



    shear_stress = (K33 * current_elastic_strain_uvs
                    + 0.5 * K33_1 * torch.sign(strain_uvs) * current_elastic_strain_uvs.square())
    grad_e = K11 * strain_uus * fuu.T + K22 * strain_vvs * fvv.T + K12 * (strain_uus * fvv.T + strain_vvs * fuu.T) + shear_stress * fuv.T

    strain_uvs_var = strain_uvs - anchor_strain_uvs
    dis_to_slide_uv = torch.abs(strain_uvs_var) - thres_uvs
    current_strain_uvs_var = strain_uvs - (anchor_strain_uvs + torch.sign(strain_uvs_var) * torch.relu(dis_to_slide_uv))


    grad_fric = Friction_K33 * current_strain_uvs_var * fuv.T

    Stretching_Force = (THICKNESS * torch.mv(cloth_SF_T, (-cloth_faces_m * (grad_e + grad_fric)).T.flatten())).reshape(-1, 3)


    if force_terms == "bending-friction":
        selected_force = Bending_Friction_Force
    elif force_terms == "bending-total":
        selected_force = Bending_Force
    elif force_terms == "all":
        selected_force = Bending_Force + Stretching_Force
    else:
        raise ValueError(f"Unsupported net-force term selection: {force_terms!r}")

    compress_force = torch.bmm(
        selected_force.unsqueeze(1), cloth_select_force_indices.unsqueeze(2)).squeeze()



    return compress_force.sum()

def dof_select(info: DiffSimModule.DofSelectInfo):
    geo = geo_slot.geometry()
    offset = geo.meta().find(builtin.dof_offset).view()[0]
    verts_idx = offset + np.arange(NUM_VERTICES * 3, dtype=np.int32)
    sdi = torch.from_numpy(verts_idx)
    return sdi


# optimizer = optim.Adam(diff_module.parameters(), lr=PARAM_LR)


# with torch.no_grad():
#     parm.U().fill_(u)



















#     for f in range(TRAIN_START_FRAME, TRAIN_END_FRAME):












#     loss.backward()





diff_module = DiffSimModule(
    world, geo_slot=geo_slot, end_frame=EndFrame, parm=parm, dof_select=dof_select)
optimizer = optim.Adam(diff_module.parameters(), lr=PARAM_LR)
_user_style_lr = DiffSimLR(diff_module, optimizer) if _cli_args.user_style_synthetic else None
losses_record = []
last_prediction = None
best_observed_loss = float("inf")
best_observed_parameters = None
best_observed_prediction = None
frozen_rollout_state = None
frozen_friction_target = None
synthetic_gt_force = None
synthetic_sweep_values = None
synthetic_sweep_losses = None
coarse_search_names = []
coarse_search_values = []
coarse_search_losses = []
experimental_gt_force = gt_comp_force.detach().clone()

with torch.no_grad():
    parm.set_physical(_initial_bending_parameters)
    parm.sync()


def _edge_history(getter, name):
    history = getter()
    expected = EndFrame
    if len(history) != expected:
        raise RuntimeError(
            f"{name} recorded {len(history)} frames, expected {expected}; "
            "cannot safely align loss frames.")
    return torch.from_numpy(np.stack(history)).to(dtype=torch.float64)


def report_bending_gate_statistics() -> None:
    """Report the pre-update gate arguments over the selected loss frames."""
    with torch.no_grad():
        diff_module()

    selector = cloth_valid_indices.to(dtype=torch.float64)
    theta = _edge_history(diff_module.get_dihedral_thetas, "dihedral_theta") @ selector.T
    anchor_warp = _edge_history(diff_module.get_anchor_thetas_warp, "anchor_theta_warp") @ selector.T
    anchor_weft = _edge_history(diff_module.get_anchor_thetas_weft, "anchor_theta_weft") @ selector.T
    plastic_warp = _edge_history(diff_module.get_plastic_thetas_warp, "plastic_theta_warp") @ selector.T
    plastic_weft = _edge_history(diff_module.get_plastic_thetas_weft, "plastic_theta_weft") @ selector.T
    rest_anchor_warp = selector @ torch.from_numpy(rest_anchor_thetas_warp).double()
    rest_anchor_weft = selector @ torch.from_numpy(rest_anchor_thetas_weft).double()
    rest_plastic_warp = selector @ torch.from_numpy(rest_plastic_thetas_warp).double()
    rest_plastic_weft = selector @ torch.from_numpy(rest_plastic_thetas_weft).double()
    bias = cloth_bias_angles.to(dtype=torch.float64)
    if theta.shape[1] != bias.numel():
        raise RuntimeError(
            f"dihedral/bias edge count mismatch: {theta.shape[1]}/{bias.numel()}")

    def previous_state(rest, current):
        initial = torch.as_tensor(rest, dtype=current.dtype, device=current.device)
        return torch.cat((initial.unsqueeze(0), current[:-1]), dim=0)

    selected = torch.from_numpy(_training_frames - 1).long()
    theta_warp = theta * torch.cos(bias).square().unsqueeze(0)
    theta_weft = theta * torch.sin(bias).square().unsqueeze(0)
    model_quantities = (
        (theta_warp, previous_state(rest_anchor_warp, anchor_warp),
         previous_state(rest_plastic_warp, plastic_warp)),
        (theta_weft, previous_state(rest_anchor_weft, anchor_weft),
         previous_state(rest_plastic_weft, plastic_weft)),
    )
    if Up_Direction == "warp":
        quantities = (
            ("warp", *model_quantities[1]),
            ("weft", *model_quantities[0]),
        )
    else:
        quantities = (
            ("warp", *model_quantities[0]),
            ("weft", *model_quantities[1]),
        )
    quantiles = torch.tensor((0.20, 0.50, 0.80, 0.95), dtype=torch.float64)
    print("Bending gate statistics over selected loss frames:")
    P = parm.forward().detach()
    model_P = _physical_to_model_parameters(P)
    w = torch.cos(bias).square()
    v = torch.sin(bias).square()
    t0_eff = w * model_P[8] + v * model_P[9]
    y0_eff = w * model_P[6] + v * model_P[7]
    cumulative = (
        _edge_history(diff_module.get_hardening_plastic_thetas_warp, "hardening_plastic_theta_warp") @ selector.T,
        _edge_history(diff_module.get_hardening_plastic_thetas_weft, "hardening_plastic_theta_weft") @ selector.T,
    )
    for direction, theta_component, prior_anchor, prior_plastic in quantities:
        friction_argument = (theta_component - prior_anchor).abs().index_select(0, selected).reshape(-1)
        plastic_argument = (theta_component - prior_plastic).abs().index_select(0, selected).reshape(-1)
        friction_q = torch.quantile(friction_argument, quantiles).tolist()
        plastic_q = torch.quantile(plastic_argument, quantiles).tolist()
        direction_index = 0 if direction == "warp" else 1
        model_index = 1 - direction_index if Up_Direction == "warp" else direction_index
        weight = (w, v)[model_index]
        prior_q = previous_state(torch.zeros_like(cumulative[model_index][0]), cumulative[model_index])
        friction_threshold = (weight * t0_eff).unsqueeze(0).expand_as(theta_component).index_select(0, selected).reshape(-1)
        plastic_yield = (weight * y0_eff + prior_q * model_P[4+model_index] / model_P[model_index]).index_select(0, selected).reshape(-1)
        friction_active_fraction = (friction_argument > friction_threshold).double().mean().item()
        plastic_active_fraction = (plastic_argument > plastic_yield).double().mean().item()
        print(
            f"  {direction}: |theta-anchor| q20/q50/q80/q95={friction_q}; "
            f"max={friction_argument.max().item():.8g}; "
            f"effective threshold min/max={friction_threshold.min().item():.8g}/{friction_threshold.max().item():.8g}; "
            f"active={friction_active_fraction:.3%}")
        print(
            f"  {direction}: |theta-plastic| q20/q50/q80/q95={plastic_q}; "
            f"max={plastic_argument.max().item():.8g}; "
            f"effective yield min/max={plastic_yield.min().item():.8g}/{plastic_yield.max().item():.8g}; "
            f"active={plastic_active_fraction:.3%}")

    strain_uv = _edge_history(diff_module.get_strain_uvs, "strain_uv")
    anchor_uv = _edge_history(diff_module.get_anchor_strains, "anchor_strain_uv")
    plastic_uv = _edge_history(diff_module.get_plastic_strains, "plastic_strain_uv")
    yield_uv = _edge_history(diff_module.get_yield_strains, "yield_uv")
    prior_anchor_uv = previous_state(rest_anchor_uvs, anchor_uv)
    prior_plastic_uv = previous_state(rest_plastic_uvs, plastic_uv)
    initial_yield_uv = torch.full_like(yield_uv[:1], Yield_UV)
    prior_yield_uv = torch.cat((initial_yield_uv, yield_uv[:-1]), dim=0)
    shear_friction_argument = (strain_uv - prior_anchor_uv).abs().index_select(0, selected).reshape(-1)
    shear_plastic_argument = (strain_uv - prior_plastic_uv).abs().index_select(0, selected).reshape(-1)
    shear_yield = prior_yield_uv.index_select(0, selected).reshape(-1)
    shear_friction_q = torch.quantile(shear_friction_argument, quantiles).tolist()
    shear_plastic_q = torch.quantile(shear_plastic_argument, quantiles).tolist()
    print("Shear gate statistics over selected loss frames:")
    print(
        f"  |strain_uv-anchor| q20/q50/q80/q95={shear_friction_q}; "
        f"max={shear_friction_argument.max().item():.8g}; "
        f"active(>{Thres_uv:.8g})="
        f"{(shear_friction_argument > Thres_uv).double().mean().item():.3%}")
    print(
        f"  |strain_uv-plastic| q20/q50/q80/q95={shear_plastic_q}; "
        f"max={shear_plastic_argument.max().item():.8g}; "
        f"active(>current yield)="
        f"{(shear_plastic_argument > shear_yield).double().mean().item():.3%}; "
        f"yield min/max={shear_yield.min().item():.8g}/{shear_yield.max().item():.8g}")


def _named_parameters(P):
    return "\n".join(
        f"  {name}: {value:.8g}" for name, value in zip(
            _BENDING_PARAMETER_NAMES, P.detach().cpu().tolist()))


def loss_fn(*, training=True, verbose=True, rollout_state=None, capture_state=None) -> torch.Tensor:
    global last_prediction, best_observed_loss, best_observed_parameters, best_observed_prediction
    global frozen_rollout_state, frozen_friction_target
    if training:
        optimizer.zero_grad()
    if (rollout_state is not None or (_cli_args.frozen_rollout_friction_residual
            and frozen_rollout_state is not None)):
        state = rollout_state if rollout_state is not None else frozen_rollout_state
        X = state["X"]
        plastic_warp = state["plastic_warp"]
        plastic_weft = state["plastic_weft"]
        anchor_warp = state["anchor_warp"]
        anchor_weft = state["anchor_weft"]
        yield_warp = state["yield_warp"]
        yield_weft = state["yield_weft"]
        hardening_warp = state["hardening_warp"]
        hardening_weft = state["hardening_weft"]
        plastic_uvs = state["plastic_uvs"]
        yield_uvs = state["yield_uvs"]
        hardening_uvs = state["hardening_uvs"]
        anchor_uvs = state["anchor_uvs"]
    else:
        if _cli_args.net_force_only:
            with torch.no_grad():
                X = diff_module()
            if X.requires_grad or X.grad_fn is not None:
                raise RuntimeError('net-force-only rollout unexpectedly retains an autograd path')
        else:
            X = diff_module()

        plastic_warp = _edge_history(
            diff_module.get_plastic_thetas_warp, "plastic_theta_warp")
        plastic_weft = _edge_history(
            diff_module.get_plastic_thetas_weft, "plastic_theta_weft")
        anchor_warp = _edge_history(
            diff_module.get_anchor_thetas_warp, "anchor_theta_warp")
        anchor_weft = _edge_history(
            diff_module.get_anchor_thetas_weft, "anchor_theta_weft")
        yield_warp = _edge_history(
            diff_module.get_yield_thetas_warp, "yield_theta_warp")
        yield_weft = _edge_history(
            diff_module.get_yield_thetas_weft, "yield_theta_weft")
        hardening_warp = _edge_history(
            diff_module.get_hardening_plastic_thetas_warp,
            "hardening_plastic_theta_warp")
        hardening_weft = _edge_history(
            diff_module.get_hardening_plastic_thetas_weft,
            "hardening_plastic_theta_weft")

        plastic_uvs = torch.from_numpy(
            np.stack(diff_module.get_plastic_strains())).double()
        yield_uvs = torch.from_numpy(
            np.stack(diff_module.get_yield_strains())).double()
        hardening_uvs = torch.from_numpy(
            np.stack(diff_module.get_hardening_strains())).double()
        anchor_uvs = torch.from_numpy(
            np.stack(diff_module.get_anchor_strains())).double()

        if _cli_args.frozen_rollout_friction_residual:
            frozen_rollout_state = {
                "X": X.detach().clone(),
                "plastic_warp": plastic_warp.detach().clone(),
                "plastic_weft": plastic_weft.detach().clone(),
                "anchor_warp": anchor_warp.detach().clone(),
                "anchor_weft": anchor_weft.detach().clone(),
                "yield_warp": yield_warp.detach().clone(),
                "yield_weft": yield_weft.detach().clone(),
                "hardening_warp": hardening_warp.detach().clone(),
                "hardening_weft": hardening_weft.detach().clone(),
                "plastic_uvs": plastic_uvs.detach().clone(),
                "yield_uvs": yield_uvs.detach().clone(),
                "hardening_uvs": hardening_uvs.detach().clone(),
                "anchor_uvs": anchor_uvs.detach().clone(),
            }
            print("Frozen-rollout diagnostic: cached the initial trajectory "
                  "and all bending/shear history variables.")

    if capture_state is not None:
        capture_state.update({name: value.detach().clone() for name, value in dict(
            X=X, plastic_warp=plastic_warp, plastic_weft=plastic_weft,
            anchor_warp=anchor_warp, anchor_weft=anchor_weft,
            yield_warp=yield_warp, yield_weft=yield_weft,
            hardening_warp=hardening_warp, hardening_weft=hardening_weft,
            plastic_uvs=plastic_uvs, yield_uvs=yield_uvs,
            hardening_uvs=hardening_uvs, anchor_uvs=anchor_uvs).items()})
        if _cli_args.audit_synthetic_friction_target is not None and rollout_state is None:
            capture_state['dihedral_theta'] = _edge_history(
                diff_module.get_dihedral_thetas, 'dihedral_theta').detach().clone()
        if _cli_args.audit_soft_constraint_strength is not None and rollout_state is None:
            records = [_fixture_audit_frames[f] for f in range(1,EndFrame+1)]
            capture_state['fixture_aim'] = torch.from_numpy(np.stack([r['aim'] for r in records]))
            capture_state['fixture_selected'] = torch.from_numpy(np.stack([r['selected'] for r in records]))
    P = parm.forward()
    model_P = _physical_to_model_parameters(P)

    if _cli_args.bending_history_autograd and rollout_state is None:
        from bending_history_autograd_normalized_training_copy import replay_bending_history
        _recorded_bending = dict(plastic_warp=plastic_warp, plastic_weft=plastic_weft,
            anchor_warp=anchor_warp, anchor_weft=anchor_weft,
            yield_warp=yield_warp, yield_weft=yield_weft,
            hardening_warp=hardening_warp, hardening_weft=hardening_weft)
        _replayed, _replay_errors = replay_bending_history(globals(), X, model_P, _recorded_bending)
        plastic_warp, plastic_weft = _replayed['plastic_warp'], _replayed['plastic_weft']
        anchor_warp, anchor_weft = _replayed['anchor_warp'], _replayed['anchor_weft']
        yield_warp, yield_weft = _replayed['yield_warp'], _replayed['yield_weft']
        hardening_warp, hardening_weft = _replayed['hardening_warp'], _replayed['hardening_weft']
        print('History autograd replay max errors: '+json.dumps(_replay_errors))

    def predict_force_terms(force_terms):
        predicted = []
        for f in _training_frames:
            frame = f - 1
            X_pred = X[
                frame * NUM_VERTICES * 3:
                (frame + 1) * NUM_VERTICES * 3].reshape(-1, 3)
            predicted.append(net_force(
                X_pred,
                model_P[0], model_P[1], kb_33, kb_12,
                model_P[2], model_P[3], model_P[4], model_P[5],
                model_P[6] * torch.ones_like(yield_warp[frame]),
                model_P[7] * torch.ones_like(yield_weft[frame]),
                yield_warp[frame], yield_weft[frame],
                hardening_warp[frame], hardening_weft[frame],
                plastic_warp[frame], plastic_weft[frame],
                model_P[8] * torch.ones_like(yield_warp[frame]),
                model_P[9] * torch.ones_like(yield_weft[frame]),
                anchor_warp[frame], anchor_weft[frame],
                K11, K22, K12, K33_0, K33_1, Fric_K33,
                Thres_uv * torch.ones_like(yield_uvs[frame]),
                anchor_uvs[frame], Hardening_K33,
                rest_yield_ori_tensor, yield_uvs[frame],
                hardening_uvs[frame], plastic_uvs[frame],
                force_terms=force_terms))
        return torch.stack(predicted)

    forces_pred = predict_force_terms(_cli_args.net_force_terms)
    loss_target = gt_comp_force
    if _cli_args.frozen_rollout_friction_residual:
        if frozen_friction_target is None:
            with torch.no_grad():
                full_initial = predict_force_terms("all")
                friction_initial = predict_force_terms("bending-friction")
                frozen_other_force = full_initial - friction_initial
                frozen_friction_target = (
                    gt_comp_force - frozen_other_force).detach().clone()
            print("Frozen-rollout diagnostic: constructed friction target as "
                  "GT minus the frozen non-friction force.")
        loss_target = frozen_friction_target
    loss = (forces_pred - loss_target).square().mean()
    if _cli_args.protocol_snapshot_dir and rollout_state is None:
        _snapshot_dir = os.path.abspath(_cli_args.protocol_snapshot_dir)
        os.makedirs(_snapshot_dir, exist_ok=True)
        _snapshot_index = getattr(loss_fn, '_protocol_snapshot_index', 0)
        loss_fn._protocol_snapshot_index = _snapshot_index + 1
        np.savez_compressed(os.path.join(_snapshot_dir, f'rollout_{_snapshot_index:03d}.npz'),
            X=X.detach().cpu().numpy(), parameters=P.detach().cpu().numpy(),
            prediction=forces_pred.detach().cpu().numpy(), gt=loss_target.detach().cpu().numpy(),
            frames=_training_frames, loss=loss.detach().cpu().item(),
            anchor_warp=anchor_warp.detach().cpu().numpy(), anchor_weft=anchor_weft.detach().cpu().numpy(),
            plastic_warp=plastic_warp.detach().cpu().numpy(), plastic_weft=plastic_weft.detach().cpu().numpy(),
            protocol=np.asarray(json.dumps(dict(config=config, training=training,
                net_force_only=_cli_args.net_force_only, grad_enabled=torch.is_grad_enabled(),
                active=_active_parameter_names, history_autograd=_cli_args.bending_history_autograd,
                end_frame=EndFrame, workspace=workspace, argv=sys.argv))))
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite bending loss; refusing optimizer update/result save.")
    last_prediction = forces_pred.detach().cpu().numpy()
    if training:
        observed_loss = loss.detach().cpu().item()
        losses_record.append(observed_loss)
        if observed_loss < best_observed_loss:
            best_observed_loss = observed_loss
            best_observed_parameters = P.detach().cpu().numpy().copy()
            best_observed_prediction = last_prediction.copy()
            print(f"New best observed loss: {best_observed_loss:.12g}")

    if verbose:
        print("predict force:", forces_pred.detach())
        if _cli_args.frozen_rollout_friction_residual:
            print("friction residual target:", loss_target)
            print("physical GT force:", gt_comp_force)
        else:
            print("GT force:", gt_comp_force)
        print("loss_fn:", loss.detach().item(), "\n", _named_parameters(P))
        if _cli_args.net_force_only:
            print("gradient mode: net_force only (simulator adjoint disabled); "
                  f"force terms={_cli_args.net_force_terms}")
        if _cli_args.frozen_rollout_friction_residual:
            print("rollout mode: frozen initial trajectory/history; target=GT-"
                  "frozen non-friction force")
    if not training:
        return loss
    physical_gradient = []
    if _cli_args.training_monitor:
        P.retain_grad()
    loss.backward()
    if parm.U().grad is None or not torch.isfinite(parm.U().grad).all().item():
        raise RuntimeError("Missing or non-finite bending parameter gradient; refusing update.")
    parm.U().grad[~_active_parameter_mask] = 0.0
    if _cli_args.training_monitor:
        physical_gradient = P.grad.detach().cpu().numpy() if P.grad is not None else np.asarray([])
        native_gradient = getattr(diff_module, 'last_dLdP', None)
        if native_gradient is not None and not _cli_args.net_force_only:
            physical_gradient = physical_gradient + native_gradient.detach().cpu().numpy()
        monitor_row = dict(epoch=len(losses_record)-1, loss=float(loss.detach()),
                           parameters=P.detach().cpu().numpy().tolist(),
                           gradient_U=parm.U().grad.detach().cpu().numpy().tolist(),
                           gradient_P=physical_gradient.tolist(),
                           active=list(_active_parameter_names),
                           contact_adjoint=bool(config['contact']['enable']) and not _cli_args.stop_contact_adjoint)
        with open(os.path.join(_log_dir, f'{fabric_type}_bend_{Up_Direction}_{_log_stamp}_monitor.jsonl'), 'a', encoding='utf-8') as monitor:
            monitor.write(json.dumps(monitor_row, allow_nan=False) + '\n')
        print('TRAIN MONITOR:', json.dumps(monitor_row, allow_nan=False))
        if observed_loss == best_observed_loss:
            checkpoint = os.path.join(_log_dir, f'{fabric_type}_bend_{Up_Direction}_{_log_stamp}_live_best.npz')
            temporary = checkpoint + '.tmp.npz'
            np.savez(temporary, parameters=best_observed_parameters,
                     parameter_names=np.asarray(_BENDING_PARAMETER_NAMES),
                     fabric=np.asarray(tested_fabric_type), compression_direction=np.asarray(Up_Direction),
                     stage=np.asarray(_cli_args.stage or 'all'), mechanism_phase=np.asarray(_cli_args.mechanism_phase),
                     schema_version=np.asarray(2), parameter_coordinate_system=np.asarray('physical_warp_weft'),
                     predicted_force=best_observed_prediction, gt_force=gt_comp_force.detach().cpu().numpy(),
                     loss=np.asarray(losses_record), best_observed_loss=np.asarray(best_observed_loss),
                     simulation_end_frame=np.asarray(EndFrame), loss_frames=np.asarray(_training_frames),
                     gradient_mode=np.asarray('autograd_plus_adjoint'),
                     final_loss=np.asarray(observed_loss), checkpoint_kind=np.asarray('live_best_unrepeated'))
            os.replace(temporary, checkpoint)
            print('Saved live best:', checkpoint)
    if _cli_args.net_force_only:
        print("simulator dLdP: disabled; using net_force parameter gradients only")
    else:
        print("dLdP:", getattr(diff_module, "last_dLdP", None))
    print("para gradient:", parm.U().grad)
    return loss


def coarse_initialize_active_parameters() -> None:
    """Coordinate-wise complete-rollout search before local optimization."""
    global coarse_search_names, coarse_search_values, coarse_search_losses
    if _cli_args.coarse_search_points == 0:
        return

    active_indices = np.flatnonzero(_active_parameter_mask.cpu().numpy())
    if active_indices.size == 0:
        raise RuntimeError("Coarse search requested without an active parameter.")

    with torch.no_grad():
        current = parm.forward().detach().cpu().numpy().copy()
        factors = np.geomspace(
            _cli_args.coarse_search_min_factor,
            _cli_args.coarse_search_max_factor,
            _cli_args.coarse_search_points)
        factors = np.unique(np.concatenate((factors, np.asarray([1.0]))))

        print(
            "Starting coordinate-wise full-rollout coarse initialization: "
            f"active={_active_parameter_names}, factors={factors.tolist()}")
        for parameter_index in active_indices:
            parameter_name = _BENDING_PARAMETER_NAMES[parameter_index]
            center = current[parameter_index]
            lower = _BENDING_PARAMETER_LOWER[parameter_index]
            upper = _BENDING_PARAMETER_UPPER[parameter_index]
            margin = max(1.0e-12, (upper - lower) * 1.0e-10)
            candidates = np.clip(center * factors, lower + margin, upper - margin)
            candidates = np.unique(np.concatenate((candidates, np.asarray([center]))))

            candidate_losses = []
            for candidate in candidates:
                trial = current.copy()
                trial[parameter_index] = candidate
                parm.set_physical(trial)
                parm.sync()
                trial_loss = loss_fn(training=False, verbose=False).item()
                candidate_losses.append(trial_loss)
                coarse_search_names.append(parameter_name)
                coarse_search_values.append(candidate)
                coarse_search_losses.append(trial_loss)
                print(
                    "Coarse full-rollout: "
                    f"{parameter_name}={candidate:.8g}, loss={trial_loss:.12g}")

            best_local = int(np.argmin(candidate_losses))
            current[parameter_index] = candidates[best_local]
            parm.set_physical(current)
            parm.sync()
            print(
                "Coarse selected: "
                f"{parameter_name}={current[parameter_index]:.8g}, "
                f"loss={candidate_losses[best_local]:.12g}")

        parm.set_physical(current)
        parm.sync()
        print("Coarse initialization complete:\n" + _named_parameters(parm.forward()))


def optimize_once():
    global gt_comp_force, synthetic_gt_force
    global synthetic_sweep_values, synthetic_sweep_losses
    global best_observed_loss, best_observed_parameters, best_observed_prediction
    if _cli_args.synthetic_friction_target is not None:
        target_name = f"friction_{_cli_args.stage}"
        target_index = _BENDING_PARAMETER_NAMES.index(target_name)
        target_value = float(_cli_args.synthetic_friction_target)
        lower = _BENDING_PARAMETER_LOWER[target_index]
        upper = _BENDING_PARAMETER_UPPER[target_index]
        if not lower < target_value < upper:
            raise ValueError(
                f"Synthetic target {target_name}={target_value:g} must lie "
                f"strictly within ({lower:g}, {upper:g})")

        initial_values = parm.forward().detach().cpu().numpy().copy()
        target_values = initial_values.copy()
        target_values[target_index] = target_value
        print("Synthetic friction recovery: generating a full-rollout target")
        print(f"  parameter: {target_name}")
        print(f"  initial: {initial_values[target_index]:.8g}")
        print(f"  target: {target_value:.8g}")
        with torch.no_grad():
            parm.set_physical(target_values)
            parm.sync()
            loss_fn(training=False)
            synthetic_gt_force = torch.from_numpy(
                last_prediction.copy()).to(dtype=torch.float64)
            gt_comp_force = synthetic_gt_force.detach().clone()
            np.savez(os.path.join(_log_dir, f'{fabric_type}_bend_{Up_Direction}_{_log_stamp}_synthetic_target.npz'),
                     parameters=target_values, parameter_names=np.asarray(_BENDING_PARAMETER_NAMES),
                     force=last_prediction.copy(), frames=np.asarray(_training_frames),
                     config=np.asarray(json.dumps(config)), user_style=np.asarray(_cli_args.user_style_synthetic))

            parm.set_physical(target_values)
            parm.sync()
            target_replay_losses = np.asarray([
                loss_fn(training=False).item(),
                loss_fn(training=False).item(),
            ])
            print(
                "Synthetic target repeat-check losses: "
                f"{target_replay_losses.tolist()}; "
                f"range={np.ptp(target_replay_losses):.12g}")
            if (not np.isfinite(target_replay_losses).all()
                    or target_replay_losses.max() > 1.0e-8):
                print(
                    "WARNING: synthetic target replay is not numerically "
                    "identical after reset; inspect this before trusting the "
                    "optimization gradient.")

            if not _cli_args.skip_synthetic_sweep:
                synthetic_sweep_values = np.linspace(
                    initial_values[target_index], target_value, 6)
                synthetic_sweep_losses = []
                for sweep_value in synthetic_sweep_values:
                    sweep_parameters = initial_values.copy()
                    sweep_parameters[target_index] = sweep_value
                    parm.set_physical(sweep_parameters)
                    parm.sync()
                    sweep_loss = loss_fn(training=False).item()
                    synthetic_sweep_losses.append(sweep_loss)
                    print(
                        "Synthetic full-rollout sweep: "
                        f"{target_name}={sweep_value:.8g}, "
                        f"loss={sweep_loss:.12g}")
                synthetic_sweep_losses = np.asarray(synthetic_sweep_losses)

            parm.set_physical(initial_values)
            parm.sync()
            if _user_style_lr is not None:
                _user_style_lr._old_P.copy_(parm.forward())
                _user_style_lr._old_U.copy_(parm.U())
        print("Synthetic friction recovery: target rollout cached; restored "
              "the requested initial parameters.")

    coarse_initialize_active_parameters()

    guard_no_improvement = 0
    unguarded_no_improvement = 0
    unguarded_robust_best = float("inf")
    guard_patience = (
        _cli_args.early_stop_patience
        if _cli_args.early_stop_patience > 0 else 12)
    guard_relative_tolerance = _cli_args.guard_relative_tolerance
    guard_lr_factor = 0.5
    guard_min_lr = 1.0e-4
    for epoch in range(NUM_EPOCH):
        print("Epoch:", epoch)
        if not _cli_args.guarded_optimization:
            optimizer.step(loss_fn)
            if _cli_args.early_stop_patience > 0:
                window = _cli_args.early_stop_window
                evaluated_epochs = len(losses_record)
                recent_losses = np.asarray(
                    losses_record[-min(window, evaluated_epochs):],
                    dtype=np.float64)
                robust_score = float(np.median(recent_losses))
                required_drop = (
                    max(abs(unguarded_robust_best)
                        * _cli_args.early_stop_relative_delta, 1.0e-12)
                    if np.isfinite(unguarded_robust_best) else 0.0)
                robust_improved = (
                    not np.isfinite(unguarded_robust_best)
                    or robust_score < unguarded_robust_best - required_drop)
                if robust_improved:
                    unguarded_robust_best = robust_score
                    unguarded_no_improvement = 0
                elif evaluated_epochs >= window:
                    unguarded_no_improvement += 1
                print(
                    "Robust early-stop monitor: "
                    f"best_single={best_observed_loss:.12g}, "
                    f"rolling_median={robust_score:.12g}, "
                    f"robust_best={unguarded_robust_best:.12g}, "
                    f"window={recent_losses.size}, "
                    f"no_improvement={unguarded_no_improvement}/"
                    f"{_cli_args.early_stop_patience}")
                if (evaluated_epochs >= max(
                        _cli_args.early_stop_min_epochs, window)
                        and unguarded_no_improvement
                        >= _cli_args.early_stop_patience):
                    print(
                        "Robust early stop: rolling-median loss made no "
                        "meaningful improvement for "
                        f"{_cli_args.early_stop_patience} evaluated epochs.")
                    break
            continue

        best_before_epoch = best_observed_loss
        physical_before_step = parm.forward().detach().cpu().numpy().copy()
        optimizer_before_step = copy.deepcopy(optimizer.state_dict())
        baseline_loss = loss_fn(training=True).detach().cpu().item()
        optimizer.step()
        with torch.no_grad():
            candidate_loss = loss_fn(training=False).detach().cpu().item()

        rejection_limit = baseline_loss * (1.0 + guard_relative_tolerance) + 1.0e-12
        if candidate_loss > rejection_limit:
            with torch.no_grad():
                parm.set_physical(physical_before_step)
                parm.sync()
            optimizer.load_state_dict(optimizer_before_step)
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = max(old_lr * guard_lr_factor, guard_min_lr)
            for group in optimizer.param_groups:
                group["lr"] = new_lr
            for state in optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = state.get(key)
                    if torch.is_tensor(value):
                        value.zero_()
            print(
                "Guard rejected candidate: "
                f"baseline={baseline_loss:.12g}, candidate={candidate_loss:.12g}, "
                f"restored parameters; lr {old_lr:.6g} -> {new_lr:.6g}")
        else:
            candidate_parameters = parm.forward().detach().cpu().numpy().copy()
            if candidate_loss < best_observed_loss:
                best_observed_loss = candidate_loss
                best_observed_parameters = candidate_parameters
                best_observed_prediction = last_prediction.copy()
                print(f"Guard accepted new best loss: {best_observed_loss:.12g}")
            print(
                "Guard accepted candidate: "
                f"baseline={baseline_loss:.12g}, candidate={candidate_loss:.12g}, "
                f"lr={optimizer.param_groups[0]['lr']:.6g}")

        if best_observed_loss < best_before_epoch - 1.0e-12:
            guard_no_improvement = 0
        else:
            guard_no_improvement += 1
        if guard_no_improvement >= guard_patience:
            print(
                "Guard early stop: no new best complete-rollout loss for "
                f"{guard_patience} epochs.")
            break
    if _cli_args.quick_compression_check:
        with torch.no_grad():
            last_candidate_loss = loss_fn(training=False).detach().cpu().item()
        print(f'Quick final candidate loss: {last_candidate_loss:.12g}')
        if last_candidate_loss < best_observed_loss:
            best_observed_loss = last_candidate_loss
            best_observed_parameters = parm.forward().detach().cpu().numpy().copy()
            best_observed_prediction = last_prediction.copy()
    if best_observed_parameters is None:
        raise RuntimeError("No finite bending training loss was observed.")
    with torch.no_grad():
        parm.set_physical(best_observed_parameters)
        parm.sync()
    print(f"Restored best observed parameters (loss={best_observed_loss:.12g}).")
    print("Final evaluation (replay of best observed parameters):")
    with torch.no_grad():
        final_loss = loss_fn(training=False).item()
    result_path = os.path.join(
        _log_dir, f"{fabric_type}_bend_{Up_Direction}_{_log_stamp}_result.npz")
    np.savez(result_path,
             loss=np.asarray(losses_record),
             final_loss=np.asarray(final_loss),
             best_observed_loss=np.asarray(best_observed_loss),
             best_observed_prediction=best_observed_prediction,
             learning_rate=np.asarray(PARAM_LR),
             user_style_synthetic=np.asarray(_cli_args.user_style_synthetic),
             epochs_requested=np.asarray(NUM_EPOCH),
             early_stop_patience=np.asarray(_cli_args.early_stop_patience),
             early_stop_relative_delta=np.asarray(
                 _cli_args.early_stop_relative_delta),
             early_stop_window=np.asarray(_cli_args.early_stop_window),
             early_stop_min_epochs=np.asarray(
                 _cli_args.early_stop_min_epochs),
             predicted_force=last_prediction,
             gt_force=gt_comp_force.detach().cpu().numpy(),
             loss_target_force=(
                 frozen_friction_target.detach().cpu().numpy()
                 if frozen_friction_target is not None
                 else gt_comp_force.detach().cpu().numpy()),
             parameters=parm.forward().detach().cpu().numpy(),
             parameter_names=np.asarray(_BENDING_PARAMETER_NAMES),
             schema_version=np.asarray(2),
             parameter_coordinate_system=np.asarray("physical_warp_weft"),
             fabric=np.asarray(tested_fabric_type),
             compression_direction=np.asarray(Up_Direction),
             stage=np.asarray(_cli_args.stage or "all"),
             mechanism_phase=np.asarray(_cli_args.mechanism_phase),
             net_force_only=np.asarray(_cli_args.net_force_only),
             contact_direct_training=np.asarray(_cli_args.contact_direct_training),
             stop_contact_adjoint=np.asarray(_cli_args.stop_contact_adjoint),
             quick_compression_check=np.asarray(_cli_args.quick_compression_check),
             simulation_end_frame=np.asarray(EndFrame),
             gradient_mode=np.asarray('fixed_rollout_net_force_partial' if _cli_args.net_force_only else
                                     'adjoint_without_contact' if _cli_args.stop_contact_adjoint else 'autograd_plus_adjoint'),
             contact_enabled=np.asarray(bool(config['contact']['enable'])),
             contact_friction_enabled=np.asarray(bool(config['contact']['friction']['enable'])),
             newton_velocity_tol=np.asarray(config['newton']['velocity_tol']),
             linear_tol_rate=np.asarray(config['linear_system']['tol_rate']),
             net_force_terms=np.asarray(_cli_args.net_force_terms),
             frozen_rollout_friction_residual=np.asarray(
                 _cli_args.frozen_rollout_friction_residual),
             synthetic_friction_target=np.asarray(
                 _cli_args.synthetic_friction_target
                 if _cli_args.synthetic_friction_target is not None else np.nan),
             skip_synthetic_sweep=np.asarray(
                 _cli_args.skip_synthetic_sweep),
             synthetic_gt_force=(
                 synthetic_gt_force.detach().cpu().numpy()
                 if synthetic_gt_force is not None else np.asarray([])),
             synthetic_sweep_values=(
                 synthetic_sweep_values
                 if synthetic_sweep_values is not None else np.asarray([])),
             synthetic_sweep_losses=(
                 synthetic_sweep_losses
                 if synthetic_sweep_losses is not None else np.asarray([])),
             experimental_gt_force=experimental_gt_force.detach().cpu().numpy(),
             guarded_optimization=np.asarray(_cli_args.guarded_optimization),
             guard_relative_tolerance=np.asarray(
                 _cli_args.guard_relative_tolerance),
             coarse_search_points=np.asarray(_cli_args.coarse_search_points),
             coarse_search_min_factor=np.asarray(
                 _cli_args.coarse_search_min_factor),
             coarse_search_max_factor=np.asarray(
                 _cli_args.coarse_search_max_factor),
             coarse_search_names=np.asarray(coarse_search_names),
             coarse_search_values=np.asarray(
                 coarse_search_values, dtype=np.float64),
             coarse_search_losses=np.asarray(
                 coarse_search_losses, dtype=np.float64),
             active_parameter_names=np.asarray(_active_parameter_names),
             training_frames=_training_frames)
    print(f"Saved bending training result: {result_path}")
    Timer.report()


def repeat_evaluate(repeats: int) -> None:
    """Check whether recover(0) makes identical forward rollouts reproducible."""
    replay_losses = []
    replay_predictions = []
    tolerance_audit = _cli_args.quick_tolerance_check is not None
    replay_seconds = []
    replay_positions = []
    import time
    with torch.no_grad():
        for repeat in range(repeats):
            print(f"Repeat evaluation: {repeat + 1}/{repeats}")
            started = time.perf_counter()
            replay_losses.append(loss_fn(training=False, verbose=not tolerance_audit).item())
            replay_seconds.append(time.perf_counter() - started)
            replay_predictions.append(last_prediction.copy())
            if tolerance_audit:
                replay_positions.append(diff_module._X.detach().cpu().numpy().reshape(EndFrame, NUM_VERTICES, 3).copy())
                print(f'Forward tolerance check: loss={replay_losses[-1]:.12g}, seconds={replay_seconds[-1]:.3f}', flush=True)
    replay_losses = np.asarray(replay_losses)
    print(
        f"Repeat-eval losses: {replay_losses.tolist()}; "
        f"range={np.ptp(replay_losses):.12g}")
    save_recorded_mesh_sequence()
    fixture_audit = {}
    if tolerance_audit:
        fixture_audit = dict(rollout_seconds=np.asarray(replay_seconds),
                             positions=np.stack(replay_positions),
                             simulation_end_frame=np.asarray(EndFrame),
                             adjoint_executed=np.asarray(False))
    if _cli_args.audit_fixture_force:
        if sorted(_fixture_audit_frames) != list(range(1, EndFrame + 1)):
            raise RuntimeError('Fixture audit is missing animation frames')
        records = [_fixture_audit_frames[f] for f in range(1, EndFrame + 1)]
        positions = diff_module._X.detach().cpu().numpy().reshape(EndFrame, NUM_VERTICES, 3)
        aims = np.stack([r['aim'] for r in records])
        masses = np.stack([r['mass'] for r in records])
        strengths = np.stack([r['strength'] for r in records])
        selected = np.stack([r['selected'] for r in records])
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise RuntimeError('Fixture audit: invalid vertex masses')
        reaction = np.sum(strengths * masses * selected
                          * (positions[:, :, 1] - aims[:, :, 1]), axis=1) / float(config['dt'])**2
        fixture_audit = dict(fixture_reaction_assuming_unit_substep=reaction,
                             audit_positions=positions, audit_aims=aims,
                             audit_mass=masses, audit_strength=strengths,
                             audit_selected=selected, audit_rest=records[0]['rest'],
                             audit_dt=np.asarray(config['dt']))
        print('Fixture reaction exported; no tare, gravity, inertia or contact correction applied.')
    if _cli_args.repeat_result:
        np.savez_compressed(
            _cli_args.repeat_result, losses=replay_losses,
            predicted_force=np.stack(replay_predictions),
            gt_force=gt_comp_force.detach().cpu().numpy(),
            frames=_training_frames, parameters=parm.forward().detach().cpu().numpy(),
            fabric=np.asarray(tested_fabric_type), direction=np.asarray(Up_Direction),
            contact=np.asarray(bool(config['contact']['enable'])),
            newton_velocity_tol=np.asarray(config['newton']['velocity_tol']),
            linear_tol_rate=np.asarray(config['linear_system']['tol_rate']),
            force_definition=np.asarray('existing net_force: bending + membrane; excludes explicit contact force'),
            **fixture_audit)
        print(f'Saved repeat force result: {_cli_args.repeat_result}')


def audit_position_gradient() -> None:
    """Linear position probe, no net_force/history-to-force or GT loss graph.

    Each horizon has its own fresh center rollout/backward before any other
    rollout overwrites the module's recorded adjoint state.
    """
    name = _cli_args.contact_gradient_audit
    if name not in _BENDING_PARAMETER_NAMES:
        raise ValueError(f'Unknown parameter {name}')
    idx = _BENDING_PARAMETER_NAMES.index(name)
    horizons = sorted(set(_cli_args.audit_position_frames))
    if horizons[-1] > EndFrame:
        raise ValueError(f'Position audit horizons must not exceed {EndFrame}')
    center = parm.forward().detach().cpu().numpy().copy()
    lo, hi = (t.detach().cpu().numpy() for t in (parm._lower, parm._upper))
    steps = [abs(center[idx])*r for r in _cli_args.audit_relative_steps]
    if any(e <= 0 or not lo[idx]+1e-6*(hi[idx]-lo[idx]) < center[idx]-e < center[idx]+e < hi[idx]-1e-6*(hi[idx]-lo[idx]) for e in steps):
        raise ValueError('Position FD steps outside invertible parameter bounds')
    initial = np.asarray(geo_slot.geometry().positions().view()).reshape(NUM_VERTICES, 3).copy()
    free = (initial[:, 1] >= 0.02) & (initial[:, 1] <= 0.07)
    if not free.any():
        raise RuntimeError('No interior vertices for position probe')
    weights = np.random.default_rng(20260920).choice([-1., 1.], size=initial.shape)
    weights[~free] = 0
    weights /= np.linalg.norm(weights)
    probe = torch.as_tensor(weights, dtype=torch.float64, device=parm.U().device)
    report = dict(mode='position_only', parameter=name, parameters=center.tolist(),
                  fabric=tested_fabric_type, direction=Up_Direction,
                  objective='unit-norm fixed random projection of final-frame interior XYZ; units m',
                  contact=bool(config['contact']['enable']), newton_velocity_tol=config['newton']['velocity_tol'],
                  linear_tol_rate=config['linear_system']['tol_rate'], horizons=[], status='running')
    report['newton_relative_tol'] = config['newton'].get('relative_tol', 0.001)
    report['initial_imperfection_amplitude'] = _cli_args.imperfection_amplitude
    path = os.path.join(_gradient_audit_dir, 'position_summary.json')
    def save():
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    def assign(values):
        with torch.no_grad():
            parm.set_physical(values)
            if not np.allclose(parm.forward().cpu().numpy(), values, rtol=1e-10, atol=1e-12):
                raise RuntimeError('Position audit parameter clipped')
            parm.sync()
    def rollout(grad):
        with torch.set_grad_enabled(grad):
            positions = diff_module().reshape(-1, NUM_VERTICES, 3)
            objective = (positions[-1]*probe).sum()
        if not torch.isfinite(positions).all().item():
            raise RuntimeError('Nonfinite position trajectory')
        return objective, positions
    original_horizon = diff_module._end_frame
    save()
    try:
        for horizon in horizons:
            diff_module._end_frame = horizon
            assign(center)
            print(f'POSITION AUDIT horizon={horizon}: center + adjoint', flush=True)
            objective, positions = rollout(True)
            base = positions.detach().cpu().numpy().copy()
            valid_hinges = np.asarray(egde_rest_lengths).reshape(-1) != 0.0
            state_arrays, plastic_history = {}, {}
            for axis in ('warp', 'weft'):
                history = np.asarray(getattr(diff_module, f'_hardening_plastic_thetas_{axis}')).copy()
                state_arrays[f'center_hardening_plastic_theta_{axis}'] = history
                valid_history = history.reshape(len(history), -1)[:, valid_hinges]
                plastic_history[f'model_{axis}'] = dict(
                    max_accumulated_plastic=float(np.max(valid_history)),
                    positive_count_by_frame=np.count_nonzero(valid_history > 1e-12, axis=1).tolist(),
                    max_by_frame=np.max(valid_history, axis=1).tolist(),
                    positive_count_tolerance=1e-12)
            print('POSITION PLASTIC HISTORY: ' + json.dumps(plastic_history), flush=True)
            value = float(objective.detach())
            grad_u, = torch.autograd.grad(objective, parm.U())
            with torch.no_grad():
                q = torch.sigmoid(parm.U())
                grad_p = (grad_u/((parm._upper-parm._lower)*q*(1-q))).cpu().numpy()
            if not np.isfinite(grad_p).all():
                raise RuntimeError('Nonfinite position adjoint gradient')
            row = dict(frame=horizon, center_objective=value,
                       adjoint=float(grad_p[idx]), all_adjoint=grad_p.tolist(),
                       plastic_history=plastic_history, steps=[])
            report['horizons'].append(row)
            save()
            arrays = dict(center_positions=base, probe=weights, parameters=center)
            arrays.update(state_arrays)
            for step_index, eps in enumerate(steps):
                values, ends = [], []
                for sign in (-1, 1):
                    changed = center.copy()
                    changed[idx] += sign*eps
                    assign(changed)
                    print(f'POSITION AUDIT frame={horizon}, delta={sign*eps:.12g}', flush=True)
                    obj, pos = rollout(False)
                    values.append(float(obj))
                    ends.append(pos.detach().cpu().numpy().copy())
                central = (values[1]-values[0])/(2*eps)
                derivative = (ends[1]-ends[0])/(2*eps)
                arrays[f'minus_{step_index}'] = ends[0]
                arrays[f'plus_{step_index}'] = ends[1]
                record = dict(epsilon=eps, minus=values[0], plus=values[1],
                              central=central, forward=(values[1]-value)/eps,
                              backward=(value-values[0])/eps,
                              absolute_error=abs(central-row['adjoint']),
                              final_position_derivative_rms=float(np.sqrt(np.mean(derivative[-1]**2))))
                row['steps'].append(record)
                save()
                print('POSITION FD: ' + json.dumps(record), flush=True)
            assign(center)
            repeated, repeated_x = rollout(False)
            drift = abs(float(repeated)-value)
            row.update(center_repeat=float(repeated), center_drift=drift,
                       repeat_position_rms=float(np.sqrt(np.mean((repeated_x.cpu().numpy()-base)**2))))
            for record in row['steps']:
                record['center_drift_over_epsilon'] = drift/record['epsilon']
            arrays['repeat_positions'] = repeated_x.cpu().numpy()
            np.savez_compressed(os.path.join(_gradient_audit_dir, f'positions_frame_{horizon:03d}.npz'), **arrays)
            save()
        report['status'] = 'complete_NOT_automatic_pass'
        save()
        print(f'Position audit complete: {path}', flush=True)
    finally:
        diff_module._end_frame = original_horizon
        assign(center)


def audit_contact_gradient() -> None:
    """Isolate Python derivative consistency from the missing rollout derivative.

    By default rollouts use net_force_only/no_grad; --audit-adjoint explicitly
    adds one implemented simulator backward after the direct-gradient check.
    Fixed-state FD also exposes any custom surrogate/history backward rules.
    """
    global gt_comp_force, synthetic_gt_force
    name = _cli_args.contact_gradient_audit
    if name not in _BENDING_PARAMETER_NAMES:
        raise ValueError(f'Unknown physical parameter: {name}')
    idx = _BENDING_PARAMETER_NAMES.index(name)
    center = parm.forward().detach().cpu().numpy().copy()
    steps = ([] if _cli_args.audit_skip_fd else
             [abs(center[idx]) * r for r in _cli_args.audit_relative_steps])
    lo = parm._lower.detach().cpu().numpy()
    hi = parm._upper.detach().cpu().numpy()
    for eps in steps:
        if eps <= 0 or not lo[idx]+1e-6*(hi[idx]-lo[idx]) < center[idx]-eps < center[idx]+eps < hi[idx]-1e-6*(hi[idx]-lo[idx]):
            raise ValueError('Audit perturbation exceeds invertible bounds; reduce --audit-relative-steps')

    def set_values(values):
        with torch.no_grad():
            parm.set_physical(values)
            actual = parm.forward().detach().cpu().numpy()
            if not np.allclose(actual, values, rtol=1e-10, atol=1e-12):
                raise RuntimeError('Physical parameter was clipped or changed by inverse mapping')
            parm.sync()

    report = dict(parameter=name, parameters=center.tolist(), fabric=tested_fabric_type,
                  direction=Up_Direction, frames=_training_frames.tolist(),
                  contact=bool(config['contact']['enable']), simulator_backward=False,
                  newton_velocity_tol=config['newton']['velocity_tol'],
                  linear_tol_rate=config['linear_system']['tol_rate'], rows=[])
    path = os.path.join(_gradient_audit_dir, 'summary.json')
    def save_report():
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)

    def fixture_deviation(captured):
        if 'fixture_aim' not in captured:
            return None
        delta = captured['X'].reshape(EndFrame,-1,3)-captured['fixture_aim']
        chosen = delta[captured['fixture_selected']]
        if chosen.numel() == 0:
            raise RuntimeError('No soft-constrained free vertices in constraint audit')
        distance = torch.linalg.vector_norm(chosen,dim=1)
        return dict(max_distance_m=float(distance.max()),rms_distance_m=float(distance.square().mean().sqrt()),
                    max_compression_axis_error_m=float(chosen[:,1].abs().max()))

    report['bending_plastic_disabled'] = _cli_args.audit_disable_bending_plastic
    report['soft_constraint_strength'] = (200.0 if _cli_args.audit_soft_constraint_strength is None
                                          else _cli_args.audit_soft_constraint_strength)
    if _cli_args.audit_synthetic_friction_target is not None:
        target_values = center.copy()
        target_values[idx] = _cli_args.audit_synthetic_friction_target
        if not lo[idx] < target_values[idx] < hi[idx]:
            raise ValueError('Synthetic target is outside physical bounds')
        print('SYNTHETIC CYLINDER: generating force GT; friction=' + str(target_values[idx]), flush=True)
        set_values(target_values)
        target_state = {}
        with torch.no_grad():
            loss_fn(training=False, verbose=False, capture_state=target_state)
            synthetic_gt_force = torch.from_numpy(last_prediction.copy()).to(dtype=torch.float64)
            gt_comp_force = synthetic_gt_force.clone()
            target_repeat = float(loss_fn(training=False, verbose=False))
        np.savez_compressed(os.path.join(_gradient_audit_dir, 'synthetic_target.npz'),
                            parameters=target_values, force=gt_comp_force.cpu().numpy(),
                            **{key: value.cpu().numpy() for key,value in target_state.items()})
        report.update(target_parameters=target_values.tolist(), target_repeat_loss=target_repeat,
                      target_fixture_deviation=fixture_deviation(target_state),
                      gt_source='same cylindrical simulator and production net_force; not experimental force')
        print('Synthetic target repeat loss: ' + str(target_repeat), flush=True)
        set_values(center)
        save_report()

    state = {}
    print('Contact gradient audit: center rollout; simulator backward DISABLED', flush=True)
    center_loss = loss_fn(training=False, verbose=False, capture_state=state)
    report['center_fixture_deviation'] = fixture_deviation(state)
    np.savez_compressed(os.path.join(_gradient_audit_dir, 'center_state.npz'), parameters=center,
                        **{key: value.cpu().numpy() for key,value in state.items()})
    if _cli_args.audit_disable_bending_plastic:
        plastic_changes = {key: float((state[key]-state[key][:1]).abs().max())
                           for key in ('plastic_warp','plastic_weft','hardening_warp','hardening_weft')}
        report['plastic_history_changes'] = plastic_changes
        if max(plastic_changes.values()) > 1e-10:
            raise RuntimeError('Plastic-disabled diagnostic still has changing bending plastic history')
    loss0 = float(center_loss.detach())
    grad_u, = torch.autograd.grad(center_loss, parm.U())
    with torch.no_grad():
        q = torch.sigmoid(parm.U())
        jac = (parm._upper-parm._lower)*q*(1-q)
        direct = float((grad_u/jac)[idx])
    report.update(center_loss=loss0, python_fixed_state_autograd=direct)
    save_report()
    if _cli_args.audit_adjoint:
        report['status'] = 'starting_simulator_backward'
        save_report()  
        print(f'ADJOINT SMOKE: executing simulator backward, contact={config["contact"]["enable"]}; no optimizer step', flush=True)
        _cli_args.net_force_only = False
        try:
            adjoint_loss = loss_fn(training=False, verbose=False)
            total_u, = torch.autograd.grad(adjoint_loss, parm.U())
            total_p = (total_u / jac).detach().cpu().numpy()
            if not np.isfinite(total_p).all():
                raise RuntimeError('Contact adjoint returned nonfinite gradients')
            report.update(simulator_backward=True,
                          adjoint_center_loss=float(adjoint_loss.detach()),
                          implemented_total_dLdP=total_p.tolist(),
                          implemented_total_selected=float(total_p[idx]),
                          status='adjoint_smoke_passed_NOT_gradient_check')
            save_report()
            print(f'Implemented autograd+adjoint dL/dP({name})={total_p[idx]:.12g}', flush=True)
        finally:
            _cli_args.net_force_only = True
        if _cli_args.audit_skip_fd:
            print(f'Adjoint smoke completed; numerical correctness NOT established. Saved {path}', flush=True)
            return
    try:
        for eps in steps:
            fixed, full = [], []
            for sign in (-1, 1):
                values = center.copy()
                values[idx] += sign*eps
                set_values(values)
                with torch.no_grad():
                    fixed.append(float(loss_fn(training=False, verbose=False, rollout_state=state)))
                    print(f'Full contact rollout: {name}={values[idx]:.12g}', flush=True)
                    candidate_state = {} if _cli_args.audit_synthetic_friction_target is not None else None
                    full.append(float(loss_fn(training=False, verbose=False, capture_state=candidate_state)))
                    if candidate_state is not None:
                        np.savez_compressed(os.path.join(_gradient_audit_dir,
                            f'rollout_eps_{eps:.9g}_{"minus" if sign < 0 else "plus"}.npz'),
                            parameters=values, force=last_prediction.copy(),
                            **{key: value.cpu().numpy() for key,value in candidate_state.items()})
            row = dict(epsilon=eps, fixed_losses=fixed, full_losses=full,
                       fixed_central=(fixed[1]-fixed[0])/(2*eps),
                       full_central=(full[1]-full[0])/(2*eps),
                       full_forward=(full[1]-loss0)/eps,
                       full_backward=(loss0-full[0])/eps)
            row['fixed_autograd_absolute_error'] = abs(row['fixed_central']-direct)
            report['rows'].append(row)
            save_report()
            print('Gradient audit: ' + json.dumps(row), flush=True)
        set_values(center)
        with torch.no_grad():
            repeat_loss = float(loss_fn(training=False, verbose=False))
        drift = abs(repeat_loss-loss0)
        report.update(center_repeat_loss=repeat_loss, center_loss_drift=drift,
                      status='complete',
                      interpretation='Fixed-state autograd is NOT the complete simulator gradient. Full FD includes state/contact evolution; difference is not proof of an adjoint bug.')
        for row in report['rows']:
            row['center_drift_over_epsilon'] = drift/row['epsilon']
        save_report()
        print(f'Center repeat drift: {drift:.12g}; saved {path}', flush=True)
    finally:
        set_values(center)


def finite_difference_parameter(parameter_name: str, epsilon: float) -> None:
    """Compare the used adjoint derivative to complete-rollout finite differences."""
    if parameter_name not in _BENDING_PARAMETER_NAMES:
        raise ValueError(
            f"Unknown --parameter-fd name {parameter_name!r}. Choose one of: "
            f"{', '.join(_BENDING_PARAMETER_NAMES)}")

    parameter_index = _BENDING_PARAMETER_NAMES.index(parameter_name)
    with torch.no_grad():
        center = parm.forward().detach().cpu().numpy().copy()
    lower = _BENDING_PARAMETER_LOWER[parameter_index]
    upper = _BENDING_PARAMETER_UPPER[parameter_index]
    minus = center.copy()
    plus = center.copy()
    minus[parameter_index] -= epsilon
    plus[parameter_index] += epsilon
    if not lower < minus[parameter_index] < upper or not lower < plus[parameter_index] < upper:
        raise ValueError(
            f"Central FD for {parameter_name} with epsilon={epsilon:g} leaves "
            f"its valid interval ({lower:g}, {upper:g})")

    print(f"Finite-difference check: {parameter_name}, center={center[parameter_index]:.12g}, "
          f"epsilon={epsilon:.12g}")
    center_loss = loss_fn(training=True).item()
    with torch.no_grad():
        center_P = parm.forward()
        dp_du = ((parm._upper - parm._lower)
                 * torch.sigmoid(parm.U())
                 * (1.0 - torch.sigmoid(parm.U())))
        used_dldp = (parm.U().grad / dp_du)[parameter_index].item()

        parm.set_physical(minus)
        parm.sync()
        minus_loss = loss_fn(training=False).item()

        parm.set_physical(plus)
        parm.sync()
        plus_loss = loss_fn(training=False).item()

        parm.set_physical(center)
        parm.sync()

    central_fd = (plus_loss - minus_loss) / (2.0 * epsilon)
    forward_fd = (plus_loss - center_loss) / epsilon
    backward_fd = (center_loss - minus_loss) / epsilon
    same_sign = (used_dldp == 0.0 or central_fd == 0.0
                 or np.sign(used_dldp) == np.sign(central_fd))
    print("FD losses: "
          f"L(P-eps)={minus_loss:.12g}, L(P)={center_loss:.12g}, "
          f"L(P+eps)={plus_loss:.12g}")
    print("FD derivatives: "
          f"used dL/dP={used_dldp:.12g}, central={central_fd:.12g}, "
          f"forward={forward_fd:.12g}, backward={backward_fd:.12g}, "
          f"same_sign={same_sign}")

tested_loads = [0.0]
compress_dists = [0.0]

def recover():

    global tested_torques, twist_angles

    tested_loads.clear()
    compress_dists.clear()

    tested_loads.append(0.0)
    compress_dists.append(0.0)

    if not world.recover(0):
        raise RuntimeError(f'Failed to recover frame 0 from {workspace}; stopping.')
    parm.sync()  

    view(geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 1
    view(geo_slot.geometry().edges().find("dihedral_theta"))[:] = 0.0


    view(geo_slot.geometry().edges().find("plastic_theta_warp"))[:] = rest_plastic_thetas_warp    
    view(geo_slot.geometry().edges().find("plastic_theta_weft"))[:] = rest_plastic_thetas_weft    
    view(geo_slot.geometry().edges().find("anchor_theta_warp"))[:] = rest_anchor_thetas_warp      
    view(geo_slot.geometry().edges().find("anchor_theta_weft"))[:] = rest_anchor_thetas_weft      
    view(geo_slot.geometry().edges().find("yield_theta_warp"))[:] = rest_yield_theta_warp_ori      
    view(geo_slot.geometry().edges().find("yield_theta_weft"))[:] = rest_yield_theta_weft_ori      
    
    view(geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 1

    view(geo_slot.geometry().triangles().find("anchor_strain_uv"))[:] = rest_anchor_uvs
    view(geo_slot.geometry().triangles().find("plastic_strain_uv"))[:] = rest_plastic_uvs
    view(geo_slot.geometry().triangles().find("yield_uv"))[:] = rest_yield_ori_uvs

    parm.sync()  


if not world.dump():
    raise RuntimeError(f'Failed to save initial backend state in {workspace}; stopping.')

def save_mesh_sequence(frame: int):
    if not SAVE_MESH_SEQUENCE:
        return
    os.makedirs(output_dir, exist_ok=True)
    io = SimplicialComplexIO()

    io.write(
        f"{output_dir}/cloth_{frame:04}.obj",
        geo_slot.geometry(),
    )


def save_recorded_mesh_sequence() -> None:
    """Export the most recently replayed headless trajectory."""
    if not SAVE_MESH_SEQUENCE:
        return
    if diff_module._X is None:
        raise RuntimeError("No simulated trajectory is available for mesh export")

    trajectory = diff_module._X.detach().cpu().numpy()
    expected_values = EndFrame * NUM_VERTICES * 3
    if trajectory.size != expected_values:
        raise RuntimeError(
            "Cannot export meshes: recorded trajectory contains "
            f"{trajectory.size} values, expected {expected_values}")
    trajectory = trajectory.reshape(EndFrame, NUM_VERTICES, 3)

    os.makedirs(output_dir, exist_ok=True)
    position_view = geo_slot.geometry().vertices().find(
        builtin.position).view()
    restore_positions = position_view.copy()
    initial_positions = np.asarray(x_init).copy()

    def assign_positions(values):
        values = np.asarray(values)
        if position_view.ndim == 3 and values.ndim == 2:
            values = values[..., np.newaxis]
        position_view[:] = values

    frames = list(range(0, EndFrame + 1, _cli_args.mesh_save_stride))
    if frames[-1] != EndFrame:
        frames.append(EndFrame)

    io = SimplicialComplexIO()
    try:
        for frame in frames:
            assign_positions(
                initial_positions if frame == 0 else trajectory[frame - 1])
            io.write(
                os.path.join(output_dir, f"cloth_{frame:04}.obj"),
                geo_slot.geometry(),
            )
    finally:
        position_view[:] = restore_positions

    print(
        f"Saved {len(frames)} mesh frames to: {output_dir} "
        f"(stride={_cli_args.mesh_save_stride})")

if _cli_args.audit_position_frames is not None:
    if _cli_args.short_force_direction_source:
        from cloth_short_force_direction_audit import run_force_direction_audit
        run_force_direction_audit(globals())
    elif _cli_args.short_force_joint:
        from cloth_short_force_joint_training import run_short_force_joint
        run_short_force_joint(globals())
    elif _cli_args.short_force_gradient_audit:
        from cloth_short_force_gradient_audit import run_short_force_audit
        run_short_force_audit(globals())
    elif _cli_args.position_joint_target:
        from cloth_position_joint_training import run_position_joint_training
        run_position_joint_training(globals())
    else:
        audit_position_gradient()
elif _cli_args.contact_gradient_audit:
    audit_contact_gradient()
elif _cli_args.parameter_fd:
    finite_difference_parameter(_cli_args.parameter_fd, _cli_args.fd_epsilon)
elif _cli_args.repeat_eval:
    repeat_evaluate(_cli_args.repeat_eval)
elif _cli_args.gate_report_only:
    report_bending_gate_statistics()
    save_recorded_mesh_sequence()
elif _cli_args.headless_train:
    print(f"Starting headless bending training for {NUM_EPOCH} epochs.")
    optimize_once()
    save_recorded_mesh_sequence()
elif GUIInfo.enabled():
    ps.init()
    ps.set_ground_plane_height_mode('manual')
    ps.set_ground_plane_height(-0.3)
    sgui = SceneGUI(scene)
    sgui.register()
    run = False

    def update():
        global tested_torques, compress_dis, compress_dists

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

                xs_numpy = np.stack([xs_np.squeeze() for xs_np in geo_slot.geometry().vertices().find(builtin.position).view()])
                xs_tensor = torch.from_numpy(xs_numpy)

                plastic_thetas_warp_tensor = torch.from_numpy(geo_slot.geometry().edges().find("plastic_theta_warp").view().copy())
                yield_thetas_warp_tensor = torch.from_numpy(geo_slot.geometry().edges().find("yield_theta_warp").view().copy())
                hardening_plasti_thetas_warp_tensor = torch.from_numpy(geo_slot.geometry().edges().find("hardening_plastic_theta_warp").view().copy())

                anchor_thetas_warp_tensor = torch.from_numpy(geo_slot.geometry().edges().find("anchor_theta_warp").view().copy())
                thres_thetas_warp_tensor = torch.from_numpy(geo_slot.geometry().edges().find("thres_theta_warp").view().copy())

                plastic_thetas_weft_tensor = torch.from_numpy(geo_slot.geometry().edges().find("plastic_theta_weft").view().copy())
                yield_thetas_weft_tensor = torch.from_numpy(geo_slot.geometry().edges().find("yield_theta_weft").view().copy())
                hardening_plasti_thetas_weft_tensor = torch.from_numpy(geo_slot.geometry().edges().find("hardening_plastic_theta_weft").view().copy())

                anchor_thetas_weft_tensor = torch.from_numpy(geo_slot.geometry().edges().find("anchor_theta_weft").view().copy())
                thres_thetas_weft_tensor = torch.from_numpy(geo_slot.geometry().edges().find("thres_theta_weft").view().copy())

                anchor_strain_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("anchor_strain_uv").view().copy())
                thres_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("thres_uv").view().copy())

                plastic_strain_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("plastic_strain_uv").view().copy())
                yield_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("yield_uv").view().copy())
                hardening_plastic_uvs_tensor = torch.from_numpy(geo_slot.geometry().triangles().find("hardening_plastic_uv").view().copy())

                compress_force = net_force(xs_tensor, kb_11, kb_22, kb_33, kb_12,
                                           fric_kb_11, fric_kb_22,
                                           Hardening_K_Model_Warp, Hardening_K_Model_Weft,
                                            rest_yield_theta_ori_warp_tensor, rest_yield_theta_ori_weft_tensor,
                                            yield_thetas_warp_tensor, yield_thetas_weft_tensor, hardening_plasti_thetas_warp_tensor, hardening_plasti_thetas_weft_tensor,
                                            plastic_thetas_warp_tensor, plastic_thetas_weft_tensor, thres_thetas_warp_tensor, thres_thetas_weft_tensor,
                                            anchor_thetas_warp_tensor, anchor_thetas_weft_tensor,
                                             K11, K22, K12, K33_0, K33_1, Fric_K33, thres_uvs_tensor, anchor_strain_uvs_tensor, 
                                            Hardening_K33, rest_yield_ori_tensor, yield_uvs_tensor, hardening_plastic_uvs_tensor, plastic_strain_uvs_tensor)

                tested_loads.append(compress_force)
                compress_dists.append(comp_d)





                sgui.update()
                Timer.report()
            else:
                run = False



    ps.set_user_callback(update)
    ps.show()

force_log_path = f"D:/Research_4/libipc/diffuipc-demos/python/final_experiments/save_plot_data/{fabric_type}_compress_{Up_Direction}_{run_type}.npy"













