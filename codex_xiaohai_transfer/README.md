# Codex 小海步态迁移代码快照

`server_snapshot/` 是 2026-07-21 从 `zero@192.168.28.127` 实际训练目录读取的版本，而不是根据聊天记录重新拼写。

主要入口：

- `server_snapshot/gym/scripts/train_edu3_timing.py`
- `server_snapshot/gym/envs/edu3_12/edu3_controller_config.py`
- `server_snapshot/gym/envs/edu3_12/edu3_tasks.py`
- `server_snapshot/gym/scripts/play_edu3_strict.py`
- `server_snapshot/eval/edu3_mujoco_timing_eval_visual.py`

R4 使用学生 9.27 kg 原身，策略周期 20 ms、物理步长 2.5 ms、半步 15 个控制周期即 0.30 s；髋侧摆改为 25 Nm，其余电机配置不变。seed42/43 均只训练到 1000 轮后停止并做双引擎复考。

`local_working_snapshots/` 保留构建与审计过程中使用的本地文件，便于比较本地工作稿与服务器正式运行快照。正式复现应以 `server_snapshot/` 为准。

学生 STL 网格没有在此目录重复存放；它们与相邻 `01_外部Edu3Flat原始包/meshes/` 来自同一学生 nqj13 资产族。两份实际运行 URDF 已保留在 `server_snapshot/resources/urdf/`。

