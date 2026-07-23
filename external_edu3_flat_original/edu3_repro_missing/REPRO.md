# Edu3-Flat 复现补丁包（对方此前缺失的部分）

本包是对已发送资产任务包 `edu3_nqj13_trainable_fullbody_v1` 的**补齐**，不是另起一套训练框架。

目标：先用**同一套 base / 奖励 / 动作处理 / MuJoCo 脚本 / 已训好的 policy** 复现视频效果；再单独对比你们本地的适配改动。

## 环境版本（本机训练/出视频时）

| 项 | 版本 |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2 |
| MuJoCo | 3.3.3 |
| 任务 gym id | `Edu3-Flat` |
| 训练入口 | `robolab/scripts/rsl_rl/train.py` |

## 包内容

| 目录 | 作用 |
|---|---|
| `02_robolab_base/` | 训练时实际使用的 `robolab/tasks/direct/base/`（含相位观测与奖励） |
| `03_bridge/` | `Edu3-Flat` 注册桥（`tasks/direct/edu3/__init__.py`） |
| `04_checkpoint/` | 视频对应模型 `exported/policy.pt` + `params/env.yaml` + `agent.yaml` |
| `05_mujoco_sim2sim/` | 与训练对齐的 MuJoCo 运行代码 + 参考视频 `simulation.mp4` |
| `01_asset_task/` | 占位说明：请继续使用你们已收到的资产任务包 |

资产/任务配置仍以先前发送的 `edu3_nqj13_trainable_fullbody_v1` 为准（含 `edu3_nqj13_flat/`）。

---

## 替换怎么做（对应此前说的 1–4）

结论：**可以靠文件替换完成**，不需要你们手写再改一遍逻辑。  
前提：用本包文件**整目录覆盖**你们为跑通而改过的对应位置，并暂时停用本地兼容补丁。

### 1）不要对目标角做关节限位平滑

- **正确行为**（本包）：`clip(action, ±1) * 0.25 + default_joint_pos`，然后直接 `set_joint_position_target` / MuJoCo PD。
- **替换**：用 `02_robolab_base/.../base/base_env.py` 覆盖你们的 `BaseEnv`；MuJoCo 用 `05_mujoco_sim2sim/.../sim2sim_edu3.py`。
- **不要保留**：你们加的「按关节 ROM 再平滑限制目标角」逻辑。

### 2）物理步长用 5ms×4，不要改成 1ms

- **正确行为**：`sim.dt = 0.005`，`decimation = 4` → 策略 20ms（与 `04_checkpoint/params/env.yaml` 一致）。
- **替换**：直接跑本包 MuJoCo 脚本即可（默认已是 5ms×4）。
- **不要保留**：MuJoCo `dt=0.001` 的改法。

### 3）保留 sim2sim 的 1 个策略步目标延迟

- **正确行为**：`target_history` 长度 2，PD 跟踪的是上一策略步目标（不是「零延迟」）。
- **替换**：用本包 `sim2sim_edu3.py`，不要删掉 `target_history` / `delayed_target_pos`。

### 4）奖励用本包 `base/mdp/rewards.py`，不要本地兼容实现

- **正确行为**：`lin_vel_x_stall_penalty`、`feet_phase_contact_xnor`、`feet_swing_*` 等均在本包 `mdp/rewards.py`。
- **替换**：整目录覆盖 `02_robolab_base/.../base/`（尤其 `mdp/rewards.py`）。
- **不要保留**：你们为跑通而补充/覆盖的那 9 个函数实现。

### 相位 74 维（你们缺 sin/cos 的根因）

本包 `base_env.py` 在 `gait_phase_period > 0` 时自动拼接 `sin/cos`（74 = 3+3+3+2+21+21+21）。  
任务配置里已设 `gait_phase_period = 0.5`。覆盖 `base/` 后，**不必再手写相位补丁**。

---

## 安装步骤（建议顺序）

在你们的 `robolab` 安装树中（路径按实际调整）：

```bash
# A. 覆盖 base（含奖励）——先备份你们改过的版本
cp -a <robolab>/robolab/tasks/direct/base <robolab>/robolab/tasks/direct/base.bak_local
rm -rf <robolab>/robolab/tasks/direct/base
cp -a 02_robolab_base/robolab/tasks/direct/base <robolab>/robolab/tasks/direct/base

# B. 安装 Edu3 注册桥
mkdir -p <robolab>/robolab/tasks/direct/edu3
cp -a 03_bridge/robolab/tasks/direct/edu3/__init__.py \
      <robolab>/robolab/tasks/direct/edu3/__init__.py

# C. 指向已收到的资产包
export EDU3_ASSET_ROOT=/path/to/edu3_nqj13_trainable_fullbody_v1

# D. 把 MuJoCo 脚本放进资产包（推荐，MJCF 相对路径最稳）
mkdir -p "$EDU3_ASSET_ROOT/scripts/mujoco"
cp -a 05_mujoco_sim2sim/scripts/mujoco/edu3_sim "$EDU3_ASSET_ROOT/scripts/mujoco/"
cp -a 04_checkpoint/exported/policy.pt "$EDU3_ASSET_ROOT/scripts/mujoco/edu3_sim/"
```

确认 `Edu3-Flat` 能被 import（Isaac 环境内）：

```bash
python -c "import robolab.tasks.direct.edu3; import gymnasium as gym; print(gym.spec('Edu3-Flat'))"
```

---

## 先复现视频（不要改奖励/动作/物理）

```bash
cd "$EDU3_ASSET_ROOT"
python scripts/mujoco/edu3_sim/sim2sim_edu3.py \
  --load_model scripts/mujoco/edu3_sim/policy.pt \
  --headless --vx 0.2 --duration 8 \
  --video_path /tmp/edu3_repro.mp4
```

期望：与本包 `05_.../simulation.mp4` 同类稳定行走（非立刻摔倒）。

关键默认（与训练一致）：

- 观测 74 × history 10
- `action_scale=0.25`，`clip_actions=1.0`，无 ROM 平滑限位
- `dt=0.005`，`decimation=4`
- 初始高度 0.40 m，全身 21 关节，机身不固定
- 默认命令视频用 **vx=0.2**（训练范围 `[-0.4, 0.4]`；你们之前的 0.30 可后测，但复现请先用 0.2）

---

## 训练命令（需要重训时）

```bash
export EDU3_ASSET_ROOT=/path/to/edu3_nqj13_trainable_fullbody_v1
cd <robolab>
python scripts/rsl_rl/train.py --task=Edu3-Flat --headless --num_envs=4096
```

最终环境/算法参数以 `04_checkpoint/params/env.yaml` 与 `agent.yaml` 为准。

---

## 建议对比顺序

1. **只跑本包 policy + 本包 MuJoCo**（验证部署对齐）
2. 再 **用本包 base 重训**（验证训练对齐）
3. 最后才单独打开你们的「ROM 限位 / 1ms 物理 / 自写奖励」做消融

第 1 步不过关时，不要先怀疑模型；优先查关节顺序、默认角、相位、动作缩放是否与本包一致。
