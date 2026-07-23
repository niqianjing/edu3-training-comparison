"""Fail-closed Isaac Lab smoke test for the EDU3 nqj13 delivery package."""

from pathlib import Path
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "edu3_robot"))
from edu3_legacy_friction_gate import set_legacy_joint_friction_checked
from edu3_nqj13_trainable_cfg import EDU3_NQJ13_TRAINABLE_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )
    robot = EDU3_NQJ13_TRAINABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
scene = InteractiveScene(SceneCfg(num_envs=16, env_spacing=2.0))
sim.reset()


class _Env:
    pass


env = _Env()
env.scene = scene
set_legacy_joint_friction_checked(env, None, SceneEntityCfg("robot", joint_names=".*"), value=0.0)

robot = scene["robot"]
target = robot.data.default_joint_pos.clone()
robot.set_joint_position_target(target)
for _ in range(4):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

if not torch.isfinite(robot.data.joint_pos).all():
    raise RuntimeError("EDU3 Isaac smoke failed: non-finite joint state")
if robot.num_joints != 21:
    raise RuntimeError(f"EDU3 Isaac smoke failed: joints={robot.num_joints}, expected=21")
print(f"EDU3_ISAACLAB_16ENV_SMOKE=PASS envs=16 joints={robot.num_joints} steps=4", flush=True)
simulation_app.close()
