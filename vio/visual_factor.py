"""Covariance-weighted two-frame visual factor matching Chapter 10 Sec. 4.3."""
from dataclasses import dataclass
import numpy as np
from .math_utils import inv_T, transform_points


@dataclass
class VisualFactor:
    points_c0: np.ndarray      # N x 3, measurement in C_k
    points_c1: np.ndarray      # N x 3, measurement in C_k+1
    cov_c0: np.ndarray         # N x 3 x 3
    cov_c1: np.ndarray         # N x 3 x 3
    T_BC: np.ndarray           # body <- camera

    def validate(self):
        n = len(self.points_c0)
        if n < 3:
            raise ValueError("At least three two-frame visual correspondences are required")
        for a in (self.points_c1, self.cov_c0, self.cov_c1):
            if len(a) != n:
                raise ValueError("Visual factor arrays must have the same number of features")


def residual_and_covariance(T_WB0, T_WB1, factor: VisualFactor, eps=1e-9):
    """Return r_j = p_C1,pred - p_C1,meas and Sigma_j.

    T_C1_C0 = inv(T_WC1) T_WC0, where T_WC = T_WB T_BC.
    Sigma = R_C1_C0 Sigma_C0 R^T + Sigma_C1.
    """
    factor.validate()
    T_WC0 = np.asarray(T_WB0) @ np.asarray(factor.T_BC)
    T_WC1 = np.asarray(T_WB1) @ np.asarray(factor.T_BC)
    T_C1_C0 = inv_T(T_WC1) @ T_WC0
    pred = transform_points(T_C1_C0, factor.points_c0)
    r = pred - np.asarray(factor.points_c1)
    R = T_C1_C0[:3, :3]
    cov = np.einsum('ij,njk,lk->nil', R, factor.cov_c0, R) + factor.cov_c1
    cov = cov + np.eye(3)[None, :, :] * eps
    return r, cov
