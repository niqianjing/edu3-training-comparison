"""Student full-body asset generated from the versioned single-source contract."""

from __future__ import annotations

from collections import defaultdict

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from .contract import CONTRACT
from .measured_friction_actuator import MeasuredFrictionDelayedPDActuatorCfg


def motor_cfg(joint_names: list[str], cfg: dict) -> MeasuredFrictionDelayedPDActuatorCfg:
    actuator_delay = CONTRACT["control"]["actuator_delay_physics_steps"]
    return MeasuredFrictionDelayedPDActuatorCfg(
        joint_names_expr=joint_names,
        effort_limit=float(cfg["peak_effort_nm"]),
        # The explicit actuator owns the physical effort clamp.  A large PhysX
        # drive max force avoids a second hidden clamp; runtime readback must
        # therefore verify the explicit actuator value, not this sentinel.
        effort_limit_sim=1.0e9,
        velocity_limit_sim=float(cfg["velocity_limit_rad_s"]),
        stiffness=float(cfg["stiffness_nm_per_rad"]),
        damping=float(cfg["drive_damping_nm_s_per_rad"]),
        armature=float(cfg["armature_kg_m2"]),
        friction=0.0,
        coulomb_friction_nm=float(cfg["coulomb_friction_nm"]),
        viscous_damping_nm_s_per_rad=float(cfg["viscous_damping_nm_s_per_rad"]),
        min_delay=int(actuator_delay[0]),
        max_delay=int(actuator_delay[1]),
    )


groups: dict[tuple, list[str]] = defaultdict(list)
for joint_name in CONTRACT["robot"]["joint_order"]:
    cfg = CONTRACT["robot"]["joints"][joint_name]
    key = (
        cfg["module"], cfg["peak_effort_nm"], cfg["velocity_limit_rad_s"],
        cfg["stiffness_nm_per_rad"], cfg["drive_damping_nm_s_per_rad"],
        cfg["armature_kg_m2"], cfg["coulomb_friction_nm"],
        cfg["viscous_damping_nm_s_per_rad"],
    )
    groups[key].append(joint_name)

actuators = {}
for index, (key, joint_names) in enumerate(groups.items()):
    actuators[f"contract_group_{index:02d}_{key[0]}"] = motor_cfg(
        joint_names, CONTRACT["robot"]["joints"][joint_names[0]]
    )


EDU3_REFERENCE_25_10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CONTRACT["provenance"]["usd"]["path"],
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
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators=actuators,
)
