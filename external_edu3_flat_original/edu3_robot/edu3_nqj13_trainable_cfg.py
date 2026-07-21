"""Isaac Lab asset configuration for the EDU3 nqj13 full-body capability probe.

Place this file in ``<asset package>/edu3_robot``.  Effort, velocity and armature
values are read from the generated manifest so the Isaac configuration cannot
silently drift from the URDF/MJCF contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
try:
    from .measured_friction_actuator import MeasuredFrictionDelayedPDActuatorCfg
except ImportError:
    from measured_friction_actuator import MeasuredFrictionDelayedPDActuatorCfg


ASSET_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ASSET_ROOT / "urdf" / "edu3_nqj13_trainable_fullbody.urdf"
USD_PATH = ASSET_ROOT / "usd" / "edu3_nqj13_trainable_fullbody.usd"
MANIFEST = json.loads((ASSET_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))
CONTACT_MATERIAL_CFG = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)
# Assign CONTACT_MATERIAL_CFG to the terrain/ground physics_material in the scene.


def _joint_value(name: str, field: str) -> float:
    return float(MANIFEST["limits"][name][field])


EDU3_NQJ13_TRAINABLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            "left_thigh_pitch_joint": -0.10,
            "right_thigh_pitch_joint": 0.10,
            "left_knee_joint": 0.30,
            "right_knee_joint": -0.30,
            "left_ankle_pitch_joint": -0.20,
            "right_ankle_pitch_joint": 0.20,
            "left_arm_pitch_joint": -0.15,
            "right_arm_pitch_joint": 0.15,
            "left_arm_roll_joint": 0.05,
            "right_arm_roll_joint": -0.05,
            "left_elbow_pitch_joint": 0.60,
            "right_elbow_pitch_joint": -0.60,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "hip_pitch_and_knee": MeasuredFrictionDelayedPDActuatorCfg(
            joint_names_expr=[".*_thigh_pitch_joint", ".*_knee_joint"],
            effort_limit=_joint_value("left_knee_joint", "effort"),
            effort_limit_sim=1.0e9,
            velocity_limit_sim=_joint_value("left_knee_joint", "velocity"),
            stiffness=100.0,
            damping=3.0,
            armature=_joint_value("left_knee_joint", "armature"),
            friction=0.0,
            coulomb_friction_nm=_joint_value("left_knee_joint", "coulomb_friction_nm"),
            viscous_damping_nm_s_per_rad=_joint_value("left_knee_joint", "viscous_damping_nm_s_per_rad"),
            min_delay=2,
            max_delay=8,
        ),
        "lateral_hip_and_waist": MeasuredFrictionDelayedPDActuatorCfg(
            joint_names_expr=[".*_thigh_yaw_joint", ".*_thigh_roll_joint", "torso_joint"],
            effort_limit=_joint_value("torso_joint", "effort"),
            effort_limit_sim=1.0e9,
            velocity_limit_sim=_joint_value("torso_joint", "velocity"),
            stiffness=80.0,
            damping=2.5,
            armature=_joint_value("torso_joint", "armature"),
            friction=0.0,
            coulomb_friction_nm=_joint_value("torso_joint", "coulomb_friction_nm"),
            viscous_damping_nm_s_per_rad=_joint_value("torso_joint", "viscous_damping_nm_s_per_rad"),
            min_delay=2,
            max_delay=8,
        ),
        "ankles": MeasuredFrictionDelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=_joint_value("left_ankle_pitch_joint", "effort"),
            effort_limit_sim=1.0e9,
            velocity_limit_sim=_joint_value("left_ankle_pitch_joint", "velocity"),
            stiffness=40.0,
            damping=1.5,
            armature=_joint_value("left_ankle_pitch_joint", "armature"),
            friction=0.0,
            coulomb_friction_nm=_joint_value("left_ankle_pitch_joint", "coulomb_friction_nm"),
            viscous_damping_nm_s_per_rad=_joint_value("left_ankle_pitch_joint", "viscous_damping_nm_s_per_rad"),
            min_delay=2,
            max_delay=8,
        ),
        "arms": MeasuredFrictionDelayedPDActuatorCfg(
            joint_names_expr=[".*_arm_.*_joint", ".*_elbow_pitch_joint"],
            effort_limit=_joint_value("left_arm_pitch_joint", "effort"),
            effort_limit_sim=1.0e9,
            velocity_limit_sim=_joint_value("left_arm_pitch_joint", "velocity"),
            stiffness=30.0,
            damping=1.2,
            armature=_joint_value("left_arm_pitch_joint", "armature"),
            friction=0.0,
            coulomb_friction_nm=_joint_value("left_arm_pitch_joint", "coulomb_friction_nm"),
            viscous_damping_nm_s_per_rad=_joint_value("left_arm_pitch_joint", "viscous_damping_nm_s_per_rad"),
            min_delay=2,
            max_delay=8,
        ),
    },
)

