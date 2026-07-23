# Edu3-Flat 训练说明

相位纯 RL（MiniFlat 风格），**无** WBC / `joint_reference_tracking`。  
任务代码在本包 `edu3_nqj13_flat/`；经 `roboparty_train` 的 bridge 注册为 gym id **`Edu3-Flat`**。

## 开训前

```bash
# 资产门（可选但建议）
python tools/verify_trainable_edu3_nqj13.py --package .
# Isaac smoke（需 Isaac Lab 环境）
# python tools/isaaclab_16env_smoke.py
```

## 训练

在 `roboparty_train/robolab` 下（与 Mini-Flat 相同入口）：

```bash
cd /home/joyin/roboparty_train/robolab
python scripts/rsl_rl/train.py --task=Edu3-Flat --headless --num_envs=4096
```

日志目录：`logs/rsl_rl/edu3_flat_phase_rl-knee-upright-20260720/`

## 任务要点

| 项 | 值 |
|---|---|
| 机器人 | `EDU3_NQJ13_TRAINABLE_CFG`（21 DoF） |
| 观测 | 74 维（含 sin/cos 相位）× history 10 |
| 命令 | `vx ∈ [-0.4, 0.4]`，vy=0，ωz=0，15% standing |
| 相位 | period=0.5 s，duty=0.615 |
| 对称增广 | 关闭（EDU3 关节布局与 Mini 不同） |
| 摩擦门 | `Edu3FlatEnv` 启动时强制 `EDU3_LEGACY_FRICTION_GATE=PASS` |

奖励表见 `edu3_nqj13_flat/REWARDS.md`。
