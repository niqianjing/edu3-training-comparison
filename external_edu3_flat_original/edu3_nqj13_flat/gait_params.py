"""Gait / command constants for Edu3-Flat (phase RL, no reference gait).

Adapted from MiniFlat ``wbc_gait_params``; WBC stride math is intentionally omitted.
Isaac control rate: step_dt = 0.005 * 4 = 0.02 s (50 Hz).
"""

import math

# --- Gait clock ---
CONTROL_FREQUENCY = 50.0  # Hz
STEP_CYCLE_PERIOD = 0.5  # s (matches prior edu/minic phase clocks; retrain if changed)
DOUBLE_SUPPORT_FRACTION = 0.23
GAIT_PHASE_OFFSET = 0.5
GAIT_STANCE_DUTY = 0.5 + DOUBLE_SUPPORT_FRACTION / 2.0  # 0.615

# --- Foot trajectory ---
DEFAULT_WALK_FOOT_LIFT = 0.020  # m
# EDU3 ankle pitch/roll are coincident; sole contact spheres sit ~3.3 cm below roll frame.
ANKLE_HEIGHT_OFFSET = 0.033  # m

# --- Velocity command (URDF-scale walk ~0.4 m/s) ---
LIN_VEL_X_MIN = -0.4  # m/s
LIN_VEL_X_MAX = 0.4  # m/s
CMD_ACTIVE_THRESHOLD = 0.02  # m/s
LIN_VEL_TRACK_STD = 0.20  # m/s
LIN_VEL_SUCCESS_ERROR = 0.08  # m/s (= 20% of LIN_VEL_X_MAX)
LIN_VEL_STALL_MIN_RATIO = 0.5

FEET_SWING_HEIGHT_SUCCESS_MIN = 0.018  # m

# Penalize hip_yaw only beyond this deadzone (tighter to block inward sweep).
HIP_YAW_DEADZONE = math.radians(1.0)

# Swing-leg knee flexion target (~31°); encourages sagittal lift over yaw sweep.
SWING_KNEE_TARGET_RAD = 0.55
SWING_KNEE_STD_RAD = 0.22
