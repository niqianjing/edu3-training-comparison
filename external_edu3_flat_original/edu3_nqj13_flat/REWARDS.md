# Edu3-Flat 奖励表（相位纯 RL，无参考步态）

针对：髋内旋扫腿、高速上半身后倾。

| 项 | 权重 | 说明 |
|---|---|---|
| `track_lin_vel_xy_exp` | +2.0 | 线速度跟踪，std=0.20；cmd \(v_x\in[-0.4,0.4]\) |
| `lin_vel_x_stall` | −2.0 | 有命令但速度不足 |
| `lin_vel_x_overspeed` | −2.0 | 超速（抑制后倾刹车） |
| `swing_knee_flexion` | **+2.0** | 摆动相屈膝 → ~0.55 rad |
| `swing_hip_yaw` | **−4.0** | 摆动相额外罚 yaw（双边） |
| `joint_deviation_hip_yaw` | **−6.0** | 死区 1° |
| `feet_swing_height` | −35 | 抬脚净空（略降，让膝主导） |
| `flat_orientation_l2` | **−1.5** | 基座姿态平（含侧倾） |
| `base_pitch` | **−4.0** | 前后俯仰都罚，保持上半身直立 |
| `upward` | **+0.8** | 保持朝上 |
| `feet_phase_contact` | +1.5 | 相位触地 |
| `termination_penalty` | −200 | 摔倒 |

相位：`period=0.5s`，`duty=0.615`。左右膝/yaw 均用同一套正则，双边同时约束。
