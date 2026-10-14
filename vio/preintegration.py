"""IMU preintegration without ground-truth inputs.

Implements the chapter's ΔR, Δv, Δp preintegration and a 9x9 covariance over
[rotation, velocity, position]. Phase C uses fixed sensor-noise covariance;
Phase D can supply AirIMU per-sample learned covariance.
"""
from dataclasses import dataclass
import numpy as np
from .math_utils import so3_exp, skew

EUROC_GYRO_NOISE_DENSITY = 1.6968e-4   # rad/s/sqrt(Hz)
EUROC_ACCEL_NOISE_DENSITY = 2.0e-3     # m/s^2/sqrt(Hz)


@dataclass
class PreintegratedIMU:
    delta_R: np.ndarray
    delta_v: np.ndarray
    delta_p: np.ndarray
    covariance: np.ndarray
    dt: float


def _as_cov3(x, default_var):
    if x is None:
        return np.eye(3) * float(default_var)
    x = np.asarray(x, dtype=np.float64)
    if x.shape == (3,):
        return np.diag(x)
    if x.shape == (3, 3):
        return x
    raise ValueError(f"Expected covariance shape (3,) or (3,3), got {x.shape}")


class IMUPreintegrator:
    def __init__(self, gyro_noise_density=EUROC_GYRO_NOISE_DENSITY,
                 accel_noise_density=EUROC_ACCEL_NOISE_DENSITY):
        self.gyro_noise_density = float(gyro_noise_density)
        self.accel_noise_density = float(accel_noise_density)

    def integrate(self, timestamps_ns, acc, gyro, gyro_bias=None, accel_bias=None,
                  acc_cov=None, gyro_cov=None,
                  covariance_floor=(1e-8, 1e-10)):
        ts = np.asarray(timestamps_ns, dtype=np.int64)
        acc = np.asarray(acc, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        if len(ts) < 2:
            raise ValueError("At least two IMU samples are required")
        if len(acc) != len(ts) or len(gyro) != len(ts):
            raise ValueError("timestamps, acc and gyro must have equal length")

        bg = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)
        ba = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=np.float64)
        acc = acc - ba
        gyro = gyro - bg

        dR = np.eye(3, dtype=np.float64)
        dv = np.zeros(3, dtype=np.float64)
        dp = np.zeros(3, dtype=np.float64)
        P = np.zeros((9, 9), dtype=np.float64)
        total_dt = 0.0

        dt_all = np.diff(ts.astype(np.float64)) * 1e-9
        dt_all = np.clip(dt_all, 1e-5, 0.1)

        for k, dt in enumerate(dt_all):
            Rk = dR.copy()
            a = acc[k]
            w = gyro[k]
            dv_prev = dv.copy()

            # Chapter / Forster-style preintegration in frame B_k.
            dp += dv_prev * dt + 0.5 * (Rk @ a) * dt * dt
            dv += (Rk @ a) * dt
            dR = dR @ so3_exp(w * dt)

            # First-order covariance propagation for x=[dtheta,dv,dp].
            F = np.eye(9, dtype=np.float64)
            F[3:6, 0:3] = -Rk @ skew(a) * dt
            F[6:9, 0:3] = -0.5 * Rk @ skew(a) * dt * dt
            F[6:9, 3:6] = np.eye(3) * dt

            if gyro_cov is None:
                qg = np.eye(3) * (self.gyro_noise_density ** 2 / max(dt, 1e-6))
            else:
                qg = _as_cov3(np.asarray(gyro_cov)[k], self.gyro_noise_density ** 2)
            if acc_cov is None:
                qa = np.eye(3) * (self.accel_noise_density ** 2 / max(dt, 1e-6))
            else:
                qa = _as_cov3(np.asarray(acc_cov)[k], self.accel_noise_density ** 2)

            # Numerical floor only; does not use ground truth.
            qg = qg + np.eye(3) * float(covariance_floor[1])
            qa = qa + np.eye(3) * float(covariance_floor[0])

            G = np.zeros((9, 6), dtype=np.float64)
            G[0:3, 0:3] = np.eye(3) * dt
            G[3:6, 3:6] = Rk * dt
            G[6:9, 3:6] = 0.5 * Rk * dt * dt
            Q = np.block([[qg, np.zeros((3, 3))], [np.zeros((3, 3)), qa]])
            P = F @ P @ F.T + G @ Q @ G.T
            P = 0.5 * (P + P.T)
            total_dt += float(dt)

        return PreintegratedIMU(dR, dv, dp, P, total_dt)
