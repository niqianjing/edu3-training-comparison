# EDU3 nqj13 可训练全身资产（最终一致性包）

本包唯一 CAD/URDF 来源：

`output/edu3_nqj13/edu3_nqj05.SLDASM/urdf/edu3_nqj05.SLDASM.urdf`

源文件 SHA256：`E3A444C38E1A975D5FDF66D5121EAB74B7E5858DA714AFF48965AECDFE0439E9`。

注册表中的加工后 URDF 只用于验证比较和碰撞代理对照，不是生成源；本包没有使用“抽象小熊”“抽象学生”或旧 nqj09 资产。

## 已完成的训练化加工

- 22 links、21 个语义关节、全身质量 9.2699999944 kg；惯量和可视网格来自上述 SolidWorks 导出。
- 原始 21 个全零 joint limit 已替换为可训练的 ROM、effort、velocity 合同。
- 能力探针力矩：髋 pitch/膝 50 Nm；髋 roll/yaw/腰 20 Nm；踝 pitch/roll 各 40 Nm；手臂 10 Nm。
- 速度上限：踝 12 rad/s，其余 24 rad/s。
- 实测输出侧动力学：25Nm 模组 `Tc=0.51 Nm, b=0.0432 Nm·s/rad`；10Nm 模组 `Tc=0.146 Nm, b=0.0306 Nm·s/rad`。
- MuJoCo 直接使用 MJCF 的 `frictionloss/damping`。
- Isaac/PhysX 的旧 joint friction 字段是无量纲系数，不能填入 Nm 数值。最终 USD 已把 `physxJoint:jointFriction` 和导入器 drive damping 对全部 21 关节覆盖为 0；实测 SI 值保存在 `edu3:measured*` 属性和 manifest 中，并由显式执行器施加：`tau_f=-Tc*tanh(qd/0.01)-b*qd`。
- Isaac 训练配置直接加载本包已经验收的 USD，禁止再次从 URDF 临时转换后直接训练。
- 启动训练仍必须调用 `edu3_legacy_friction_gate.py` 做逐位写0和回读；日志必须同时出现 `EDU3_LEGACY_FRICTION_GATE=PASS` 与 `EDU3_MEASURED_FRICTION_MODEL=PASS`。
- 地面接触合同：静摩擦1.0、动摩擦1.0、恢复系数0。

## 文件说明

- `urdf/edu3_nqj13_trainable_fullbody.urdf`：完整可训练 URDF，保留实测 SI dynamics。
- `mjcf/edu3_nqj13_trainable_fullbody.xml`：MuJoCo 21执行器模型。
- `usd/edu3_nqj13_trainable_fullbody.usd`：全新转换并做摩擦语义覆盖后的 Isaac USD；`configuration/` 子层必须一同保留。
- `meshes/`：来自指定 nqj13 SolidWorks 导出的22个 STL，使用相对路径。
- `edu3_robot/edu3_nqj13_trainable_cfg.py`：直接加载最终 USD，从 manifest 读取合同。
- `edu3_robot/edu3_legacy_friction_gate.py`：旧 PhysX 摩擦字段的强制清零、逐位回读门。
- `edu3_robot/measured_friction_actuator.py`：显式施加实测 SI 摩擦和阻尼。
- （原目录名 `isaaclab/` 已改为 `edu3_robot/`，避免遮盖 Isaac Lab 包。）
- `asset_manifest.json`：源 hash、生成物 hash、21关节合同和名称映射。
- `tools/`：生成、跨格式验证、USD摩擦覆盖、USD严格验证和16环境 smoke 脚本。
- `PACKAGE_SHA256SUMS`：最终包内每个文件的 SHA256 清单。
- `FINAL_VERIFICATION.md`：最终验收结果。

## 训练任务（Edu3-Flat）

本包新增相位纯 RL 任务（参考 MiniFlat，无参考步态）：见 `edu3_nqj13_flat/` 与 `TRAIN_EDU3_FLAT.md`。  
gym id：`Edu3-Flat`（经 `roboparty_train` bridge 注册）。

## 训练前强制门

1. 运行 `tools/verify_trainable_edu3_nqj13.py`，必须返回 PASS。
2. 在 Isaac Lab 5.1 环境运行 `tools/isaaclab_16env_smoke.py`。
3. 日志必须同时出现：
   - `EDU3_MEASURED_FRICTION_MODEL=PASS`
   - `EDU3_LEGACY_FRICTION_GATE=PASS envs=16 joints=21 ... after_min=0 after_max=0`
   - `EDU3_ISAACLAB_16ENV_SMOKE=PASS envs=16 joints=21`
4. 缺任一道门，禁止正式训练。

## 边界说明

40Nm 踝 pitch/roll 是仿真结构能力探针上限，不代表双电机差动机构能在任意方向同时输出 `40+40 Nm`。真机部署前必须改为电机空间钳位/菱形包络，并通过连续热账；本包用于先验证结构在仿真中的运动可行性，不代表已经满足真机毕业条件。
