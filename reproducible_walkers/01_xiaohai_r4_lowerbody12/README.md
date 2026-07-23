# 01 小海方法迁移：学生下半身 R4

## 方案

- 学生 9.27 kg 原身，上半身固定。
- 12 个腿关节主动控制。
- 髋前后摆、髋侧摆、膝关节 25 Nm；髋扭转和双踝 10 Nm。
- PhysX 物理步长 2.5 ms，策略周期 20 ms。
- 完整步态周期 30 个策略步；半步 15 个策略步，即 0.30 s。
- seed42 与 seed43 是同一训练合同的两个随机种子。

## 关键文件

- `training_snapshot/files/gym/envs/edu3_12/`：环境、控制器和配置。
- `training_snapshot/files/gym/scripts/train_edu3_timing.py`：训练入口。
- `training_snapshot/files/gym/scripts/play_edu3_strict.py`：Isaac Gym 严格评估入口。
- `eval/edu3_mujoco_timing_eval.py`：MuJoCo 数值复考。
- `eval/edu3_mujoco_timing_eval_visual.py`：MuJoCo 实体网格视频。
- `assets/edu3_xiaohai12/`：实际训练用 URDF 与全部网格。
- `models/seed42|seed43/model_1000.pt`：两个检查点。

训练任务名：

- `edu3_xiaohai_roll25_timing20ms_r4_seed42`
- `edu3_xiaohai_roll25_timing20ms_r4_seed43`

运行时应在 `training_snapshot` 对应的独立 Isaac Gym 环境中调用训练入口，并通过 `--task` 选择上述任务；具体通用参数以脚本 `--help` 为准。

## 已保存结果与限制

- seed42 Isaac：8 秒前进约 1.183 m，侧移约 0.323 m，转向约 -1.87°。
- seed42 MuJoCo：8 秒前进约 1.470 m，侧移约 0.121 m，转向约 -0.15°。
- 保存结果仍有原始目标越过真实关节范围的问题，不能直接当作真机部署包。
- 此路线直接由 Isaac Gym 导入 URDF。`assets/fullbody_usd_reference_not_used_by_r4/` 只用于与学生全身 USD 对照，不参与 R4 训练。

