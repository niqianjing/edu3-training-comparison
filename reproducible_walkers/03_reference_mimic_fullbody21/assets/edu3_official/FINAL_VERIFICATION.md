# EDU3 nqj13 最终资产验收

验收时间：2026-07-16（Asia/Shanghai）

## 来源门

- 唯一原始 URDF：`edu3_nqj05.SLDASM.urdf`
- 原始 SHA256：`e3a444c38e1a975d5fdf66d5121eab74b7e5858da714aff48965aecdfe0439e9`
- 注册表加工后 URDF：仅比较验证，不作为生成源。

## 静态与运行时结果

- URDF/MJCF/manifest 严格验证：PASS，错误数0。
- MuJoCo 真实编译：PASS；`nq=28, nv=27, nu=21`，质量 `9.2699999944 kg`。
- MuJoCo 实测动力学回读：25Nm组 `frictionloss=0.51, damping=0.0432`；10Nm组 `frictionloss=0.146, damping=0.0306`。
- 全新 USD 转换：PASS；未复用旧 USD 缓存。
- USD 严格回读：PASS；22刚体、21转动关节、质量 `9.2700000628829 kg`。
- USD 21关节 ROM/effort/velocity：PASS。
- USD 旧 `physxJoint:jointFriction`：21/21 为0。
- USD 导入器 drive damping：21/21 为0，避免与显式 SI 模型重复计算。
- USD 专用实测 SI 属性：21/21 与 manifest 一致。
- mesh 来源/hash：22/22 与指定 nqj13 原始来源一致。

## Isaac Lab 16环境门

直接加载最终 USD，16环境、21关节、4物理步：PASS。

关键日志：

```text
EDU3_MEASURED_FRICTION_MODEL=PASS ...
EDU3_LEGACY_FRICTION_GATE=PASS envs=16 joints=21 requested=0 before_min=0 before_max=0 after_min=0 after_max=0
EDU3_ISAACLAB_16ENV_SMOKE=PASS envs=16 joints=21 steps=4
```

## 同期结构能力探针

`Edu3-RealBody-FricZero-Torque2x-Retrain` 已完成4096env×6000轮并生成 `model_5999.pt`。末100轮均值：

- feet_air：`0.002621876`（旧50/20/40手刹炉 `0.000625464`，约4.19倍）
- track_lin：`0.581108`
- mean_episode_length：`884.8714`
- mean_reward：`19.22227`

结论：清除旧 PhysX 隐形手刹显著恢复抬脚权限，证明旧摩擦字段是重要根因；但该单变量炉的跟踪和总奖励没有超过旧炉，不能把“摩擦修复”写成“已经走好”。最终策略质量仍需结合视频和后续奖励/控制结构继续优化。
