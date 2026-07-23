"""Startup gate for the solver-visible legacy PhysX DOF friction field."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def set_legacy_joint_friction_checked(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    value: float = 0.0,
    atol: float = 1.0e-7,
) -> None:
    """Write the legacy solver field, read it back, and fail closed on drift."""
    asset = env.scene[asset_cfg.name]
    view = asset.root_physx_view
    before = view.get_dof_friction_coefficients()
    target = before.clone()
    selected_envs = (
        torch.arange(target.shape[0], device=target.device)
        if env_ids is None
        else env_ids.to(device=target.device, dtype=torch.long)
    )
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or isinstance(joint_ids, slice):
        joint_ids = torch.arange(target.shape[1], device=target.device)
    else:
        joint_ids = torch.as_tensor(joint_ids, device=target.device, dtype=torch.long)
    target[selected_envs[:, None], joint_ids[None, :]] = value
    physx_indices = torch.arange(target.shape[0], dtype=torch.int32, device="cpu")
    view.set_dof_friction_coefficients(target, physx_indices)
    after = view.get_dof_friction_coefficients()
    checked = after[selected_envs[:, None], joint_ids[None, :]]
    max_error = float(torch.max(torch.abs(checked - value)).item())
    if not torch.isfinite(checked).all() or max_error > atol:
        raise RuntimeError(
            "EDU3 legacy friction gate FAILED: "
            f"requested={value}, max_error={max_error:.9g}, "
            f"before_range=({before.min().item():.6g},{before.max().item():.6g}), "
            f"after_range=({after.min().item():.6g},{after.max().item():.6g})"
        )
    print(
        "EDU3_LEGACY_FRICTION_GATE=PASS "
        f"envs={selected_envs.numel()} joints={joint_ids.numel()} "
        f"requested={value:.6g} before_min={before.min().item():.6g} "
        f"before_max={before.max().item():.6g} after_min={after.min().item():.6g} "
        f"after_max={after.max().item():.6g}",
        flush=True,
    )

