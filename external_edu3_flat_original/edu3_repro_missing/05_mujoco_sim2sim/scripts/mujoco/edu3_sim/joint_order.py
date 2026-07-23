"""Isaac Lab / PhysX DOF order for EDU3 nqj13 (BFS over URDF kinematic tree)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# Exact ``robot.joint_names`` from Isaac Lab (printed 2026-07-21).
ISAAC_JOINT_NAMES: list[str] = [
    "left_thigh_pitch_joint",
    "right_thigh_pitch_joint",
    "torso_joint",
    "left_thigh_roll_joint",
    "right_thigh_roll_joint",
    "left_arm_pitch_joint",
    "right_arm_pitch_joint",
    "left_thigh_yaw_joint",
    "right_thigh_yaw_joint",
    "left_arm_roll_joint",
    "right_arm_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_arm_yaw_joint",
    "right_arm_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

REVOLUTE_SKIP_TYPES = frozenset({"fixed", "floating"})


def parse_urdf_bfs_joint_order(urdf_path: str | Path, root_link: str = "base_link") -> list[str]:
    root = ET.parse(urdf_path).getroot()
    parent_to_children: dict[str, list[tuple[str, str]]] = {}
    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type in REVOLUTE_SKIP_TYPES:
            continue
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        name = joint.get("name")
        parent_to_children.setdefault(parent, []).append((name, child))
    for children in parent_to_children.values():
        children.sort(key=lambda item: item[0])

    order: list[str] = []
    current_links = [root_link]
    while current_links:
        level_joints: list[tuple[str, str]] = []
        for link in current_links:
            level_joints.extend(parent_to_children.get(link, []))
        level_joints.sort(key=lambda item: item[0])
        next_links: list[str] = []
        for joint_name, child_link in level_joints:
            order.append(joint_name)
            next_links.append(child_link)
        current_links = next_links
    return order


def assert_isaac_joint_order(urdf_path: str | Path, root_link: str = "base_link") -> list[str]:
    """Return Isaac joint order; warn if naive URDF BFS disagrees (EDU3 USD order differs)."""
    derived = parse_urdf_bfs_joint_order(urdf_path, root_link=root_link)
    if derived != ISAAC_JOINT_NAMES:
        print(
            "NOTE: URDF alphabetical BFS != Isaac joint_names; using Isaac order from Lab.\n"
            f"  BFS: {derived}\n"
            f"  Isaac: {ISAAC_JOINT_NAMES}"
        )
    if sorted(derived) != sorted(ISAAC_JOINT_NAMES):
        raise ValueError(
            "Joint name sets differ between URDF and ISAAC_JOINT_NAMES.\n"
            f"  only in URDF: {sorted(set(derived) - set(ISAAC_JOINT_NAMES))}\n"
            f"  only in Isaac: {sorted(set(ISAAC_JOINT_NAMES) - set(derived))}"
        )
    return ISAAC_JOINT_NAMES
