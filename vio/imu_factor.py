"""Standard preintegrated IMU residuals, using estimator state only (no GT)."""
from dataclasses import dataclass
import numpy as np
from .math_utils import so3_log
from .preintegration import PreintegratedIMU


@dataclass
class NavState:
    T_WB: np.ndarray
    velocity_W: np.ndarray


def residual(T_WB0, v0, T_WB1, v1, preint: PreintegratedIMU, gravity_W):
    T_WB0 = np.asarray(T_WB0, dtype=np.float64)
    T_WB1 = np.asarray(T_WB1, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    g = np.asarray(gravity_W, dtype=np.float64)

    R0, p0 = T_WB0[:3, :3], T_WB0[:3, 3]
    R1, p1 = T_WB1[:3, :3], T_WB1[:3, 3]
    dt = preint.dt

    r_R = so3_log(preint.delta_R.T @ (R0.T @ R1))
    r_v = R0.T @ (v1 - v0 - g * dt) - preint.delta_v
    r_p = R0.T @ (p1 - p0 - v0 * dt - 0.5 * g * dt * dt) - preint.delta_p
    return np.concatenate([r_R, r_v, r_p])
