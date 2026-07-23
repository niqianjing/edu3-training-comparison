# 02 行止方法：学生全身 21 关节

## 方案

- 学生 9.27 kg 全身，21 个关节主动控制，机身自由。
- 不依赖外部动作文件；策略根据机器人状态、速度命令和步态相位自行形成动作。
- 每帧 74 项观察，连续 10 帧，共 740 项；输出 21 项动作。
- 原训练能力探针为 50/40/20/10 Nm 分组，另保存冻结 seed43 模型的真实候选 25/10 Nm 降力矩考试。
- seed42/43 的 `model_5999.pt` 及 TorchScript/ONNX 导出模型均已保存。

## 关键文件

- `original_package/external_edu3_flat_20260721/`：行止交付包的 URDF、USD、MJCF、mesh、任务、奖励、执行器和工具。
- `framework_overlay/base/`：训练机上实际配套的基础环境代码。
- `framework_overlay/train.py`：训练入口快照。
- `models/seed42|seed43/`：检查点、导出策略和参数快照。
- `eval/`：原合同 Isaac/MuJoCo 证据和 seed43 25/10 Nm 冻结模型考试。

优先使用交付包中的 `FINAL_VERIFICATION.md`、`integration/launch_external_probe_dual_gpu.sh`、`integration/external_play_eval.py` 与 `integration/external_mujoco_eval.py` 还原运行方式。

## 注意

- 740 项不是 740 种不同传感器，而是每帧 74 项状态连续保存 10 帧。
- 训练机适配代码与原交付包并列保存，不能混称为“原文件逐字复现”。
- 仿真能走与硬件可用是两道门；真机仍需验证真实电流、温升、电池压降、通信延迟和关节标定。

