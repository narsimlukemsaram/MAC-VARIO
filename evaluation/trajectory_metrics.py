import numpy as np
from vio.math_utils import so3_log


def umeyama_align(est, gt, with_scale=True):
    est = np.asarray(est, dtype=np.float64); gt = np.asarray(gt, dtype=np.float64)
    mu_e, mu_g = est.mean(0), gt.mean(0)
    X, Y = est-mu_e, gt-mu_g
    C = (Y.T @ X) / len(est)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[-1,-1] = -1
    Rm = U @ S @ Vt
    var = np.mean(np.sum(X*X, axis=1))
    s = float(np.sum(D*np.diag(S))/var) if with_scale and var > 1e-12 else 1.0
    t = mu_g - s*Rm@mu_e
    return Rm,t,s


def ate_rmse(est_pos, gt_pos, align=True):
    if align:
        Rm,t,s = umeyama_align(est_pos, gt_pos, True)
        est_pos = (s*(Rm@np.asarray(est_pos).T)).T+t
    e = np.linalg.norm(np.asarray(est_pos)-np.asarray(gt_pos), axis=1)
    return float(np.sqrt(np.mean(e*e)))


def rpe(est_T, gt_T):
    re, te = [], []
    for i in range(len(est_T)-1):
        Er = np.linalg.inv(gt_T[i]) @ gt_T[i+1]
        Ee = np.linalg.inv(est_T[i]) @ est_T[i+1]
        E = np.linalg.inv(Er) @ Ee
        re.append(np.degrees(np.linalg.norm(so3_log(E[:3,:3]))))
        te.append(np.linalg.norm(E[:3,3]))
    return float(np.mean(re)), float(np.mean(te))
