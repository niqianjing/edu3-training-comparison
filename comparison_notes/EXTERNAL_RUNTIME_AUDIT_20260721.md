# 外部 Edu3-Flat 运行审计（2026-07-21）

## 已通过

- 原交付包 `PACKAGE_SHA256SUMS`：全部通过。
- 原始 URDF SHA256：`E3A444C38E1A975D5FDF66D5121EAB74B7E5858DA714AFF48965AECDFE0439E9`，与清单一致。
- 加工后 URDF SHA256：`F989ED31778C14587C43491B52B9F427A05CA268DDC53D77BEB91589A168C0F0`，与登记参考和交付包一致。
- 交付包自带严格校验：`PASS`；质量、惯量、碰撞、21 关节映射、限位、URDF/MJCF 合同一致。

## 当前不能直接训练的原因

交付包不是自包含工程。`edu3_flat_env_cfg.py` 依赖新版
`robolab.tasks.direct.base.mdp` 的下列函数，但交付包没有携带，墨造现有
RoboLab 基础层及其备份中也不存在：

1. `lin_vel_x_stall_penalty`
2. `lin_vel_x_overspeed_penalty`
3. `feet_phase_contact_xnor`
4. `feet_swing_contact_penalty`
5. `feet_swing_height_penalty`
6. `feet_contact_no_vel`
7. `hip_pos_l2`
8. `single_foot_stance_without_cmd`

实际启动在解析奖励配置时停止，首个错误为：

```text
AttributeError: module 'robolab.tasks.direct.base.mdp'
has no attribute 'lin_vel_x_stall_penalty'
```

因此当前没有开始有效训练，也没有产生可比较策略。

## 还发现的说明错误

README 给出的验证命令只写了 `--package .`，但脚本实际还强制要求
`--raw`、`--reference` 和 `--report`。服务器默认 `python` 也不是正确的
Isaac Lab Python 3.11。按脚本真实参数和正确解释器运行后，静态严格校验通过。

## 给代码提供方/Fable的明确请求

请补交与此任务同一提交版本的 `roboparty_train/robolab` 基础层，至少包含
上述八个函数及其单元、返回形状和命令掩码语义；最好直接给出完整提交号。
在准确实现到齐之前，不以同名猜测公式，也不使用返回零的占位函数训练，
否则结果不再代表“别人训练的代码”。

## 对比纪律

该外部任务使用 21 自由度全身和能力探针力矩（髋前后/膝 50 Nm、髋侧摆/
髋扭转/腰 20 Nm、踝 40 Nm、手臂 10 Nm），不是 Codex 小海迁移版的
真实学生电机合同。即使后续走好，也只能先证明此外部算法和能力探针配置，
不能直接证明真实学生硬件可部署。
