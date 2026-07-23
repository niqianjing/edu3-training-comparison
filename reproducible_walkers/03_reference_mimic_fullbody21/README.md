# 03 小海参考动作迁移：学生全身 21 关节

## 方案

- 学生 9.27 kg 全身，21 个关节主动控制，机身自由。
- 将小海走路参考动作重定向到学生的 21 个关节；策略同时看机器人状态和参考进度，学习跟随动作并保持平衡。
- 每帧 75 项观察，连续 10 帧，共 750 项；输出 21 项动作。
- 髋前后摆、髋侧摆、膝关节 25 Nm；腰、髋扭转、双踝和双臂 10 Nm。
- Isaac 物理步长 5 ms，MuJoCo 物理步长 1 ms，策略周期均为 20 ms。

## 关键文件

- `project/mimic_real/`：隔离的实际训练工程。
- `assets/edu3_official/`：URDF、USD、由合同生成的 MJCF、网格和资产清单。
- `contract_and_tools/edu3_reference_contract_v1.json`：单一训练/转换合同。
- `contract_and_tools/edu3_walk_from_xiaohai_v1.json`：重定向后的参考动作。
- `contract_and_tools/start_edu3_reference_dual.sh`：双种子训练启动脚本。
- `contract_and_tools/eval_edu3_reference_isaac.py`：Isaac 严格评估。
- `contract_and_tools/eval_edu3_reference_mujoco_velocity_ab.py`：最终修正后的 MuJoCo 闭环评估。
- `models/seed42|seed43/model_1000.pt`：训练检查点。
- `models/seed42|seed43/exported_policy.pt`：MuJoCo 使用的导出策略。
- `eval/seed42|seed43/`：Isaac 和最终 RootOmegaFix MuJoCo 视频、摘要与逐步数据。

## 最终 MuJoCo 对齐要点

1. 自由根角速度使用 `data.qvel[3:6]`，不能用坐标语义不同的 `mj_objectVelocity()` 结果代替。
2. 30 Hz、245 帧的参考动作重采样到 50 Hz、406 帧，与 20 ms 策略周期一致。
3. 单帧 75 项：根角速度 3、投影重力 3、关节位置 21、关节速度 21、上次实际执行动作 21、相位 6；十帧合计 750。
4. 动作按“参考目标 + 原始动作 × 0.25”生成，再按真实关节限位处理，经过显式 PD、摩擦和力矩限制；写入历史的是实际执行动作。

最终 MuJoCo 8.1 秒：seed42 前进约 2.104 m、侧移约 0.267 m；seed43 前进约 2.138 m、侧移约 0.303 m，均未摔倒且实际执行目标越界为 0。仍存在肩侧摆贴近限位和侧移，属于仿真成功基线，不是直接真机部署版。

