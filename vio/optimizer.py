"""Two-frame covariance-weighted nonlinear optimiser for MAC-VARIO.

Important correction relative to the student's earlier runner:
  * no ground-truth pose, velocity, orientation, or bias enters optimisation;
  * the visual term is an actual C_k -> C_k+1 correspondence residual;
  * the IMU term uses R_k from the estimated previous state;
  * GT is reserved for evaluation only.
"""
from dataclasses import dataclass
import numpy as np
from .math_utils import perturb_pose_left
from .visual_factor import residual_and_covariance, VisualFactor
from .imu_factor import residual as imu_residual, NavState
from .preintegration import PreintegratedIMU


@dataclass
class OptimResult:
    state_k_adjusted: NavState
    state_k1: NavState
    success: bool
    iterations: int
    final_cost: float
    cost_log: list


def _safe_cholesky(C, eps=1e-9):
    C = 0.5 * (C + C.T)
    for s in (eps, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        try:
            return np.linalg.cholesky(C + np.eye(C.shape[0]) * s)
        except np.linalg.LinAlgError:
            pass
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-8, None)
    return np.linalg.cholesky(vecs @ np.diag(vals) @ vecs.T)


class TwoFrameVIOOptimizer:
    def __init__(self, gravity_W=(0.0, 0.0, -9.81), imu_weight=1.0,
                 huber_delta=2.5, lm_lambda=1e-4, max_iter=8,
                 tol=1e-6, fd_eps=1e-6, max_features=None):
        self.gravity_W = np.asarray(gravity_W, dtype=np.float64)
        self.imu_weight = float(imu_weight)
        self.huber_delta = float(huber_delta)
        self.lm_lambda = float(lm_lambda)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)
        self.max_features = max_features

    def _select_features(self, vis):
        if self.max_features is None or len(vis.points_c0) <= self.max_features:
            return vis
        # Deterministic uncertainty-based selection: retain most informative features.
        score = np.trace(vis.cov_c0, axis1=1, axis2=2) + np.trace(vis.cov_c1, axis1=1, axis2=2)
        idx = np.argsort(score)[:self.max_features]
        return VisualFactor(vis.points_c0[idx], vis.points_c1[idx],
                            vis.cov_c0[idx], vis.cov_c1[idx], vis.T_BC)

    def _decode(self, x, T1_init, v0_init, v1_init):
        T1 = perturb_pose_left(T1_init, x[:6])
        v0 = v0_init + x[6:9]
        v1 = v1_init + x[9:12]
        return T1, v0, v1

    def _weighted_residual(self, x, state0, T1_init, v1_init, vis, imu):
        T1, v0, v1 = self._decode(x, T1_init, state0.velocity_W, v1_init)
        chunks = []

        rv, Sv = residual_and_covariance(state0.T_WB, T1, vis)
        for i in range(len(rv)):
            L = _safe_cholesky(Sv[i])
            z = np.linalg.solve(L, rv[i])
            n = np.linalg.norm(z)
            # IRLS Huber on Mahalanobis norm.
            w = 1.0 if n <= self.huber_delta else self.huber_delta / max(n, 1e-12)
            chunks.append(np.sqrt(w) * z)

        if imu is not None and self.imu_weight > 0.0:
            ri = imu_residual(state0.T_WB, v0, T1, v1, imu, self.gravity_W)
            Li = _safe_cholesky(imu.covariance)
            zi = np.linalg.solve(Li, ri)
            chunks.append(np.sqrt(self.imu_weight) * zi)
        return np.concatenate(chunks)

    def optimize(self, state0: NavState, T_WB1_init, v1_init,
                 visual: VisualFactor, imu: PreintegratedIMU | None):
        vis = self._select_features(visual)
        x = np.zeros(12, dtype=np.float64)
        lam = self.lm_lambda
        cost_log = []
        success = False

        for it in range(self.max_iter):
            r = self._weighted_residual(x, state0, T_WB1_init, v1_init, vis, imu)
            cost = 0.5 * float(r @ r)
            cost_log.append(cost)

            J = np.empty((len(r), 12), dtype=np.float64)
            for j in range(12):
                xp = x.copy(); xm = x.copy()
                xp[j] += self.fd_eps; xm[j] -= self.fd_eps
                rp = self._weighted_residual(xp, state0, T_WB1_init, v1_init, vis, imu)
                rm = self._weighted_residual(xm, state0, T_WB1_init, v1_init, vis, imu)
                J[:, j] = (rp - rm) / (2.0 * self.fd_eps)

            H = J.T @ J + lam * np.eye(12)
            g = J.T @ r
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dx)):
                break

            x_try = x + dx
            r_try = self._weighted_residual(x_try, state0, T_WB1_init, v1_init, vis, imu)
            cost_try = 0.5 * float(r_try @ r_try)
            if cost_try < cost:
                x = x_try
                lam = max(lam * 0.5, 1e-9)
                if np.linalg.norm(dx) < self.tol or abs(cost - cost_try) < self.tol:
                    success = True
                    cost_log.append(cost_try)
                    break
            else:
                lam = min(lam * 10.0, 1e8)
        else:
            success = True

        T1, v0, v1 = self._decode(x, T_WB1_init, state0.velocity_W, v1_init)
        final_r = self._weighted_residual(x, state0, T_WB1_init, v1_init, vis, imu)
        return OptimResult(
            state_k_adjusted=NavState(state0.T_WB.copy(), v0),
            state_k1=NavState(T1, v1),
            success=success,
            iterations=len(cost_log),
            final_cost=0.5 * float(final_r @ final_r),
            cost_log=cost_log,
        )
