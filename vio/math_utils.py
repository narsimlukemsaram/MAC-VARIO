import numpy as np


def skew(v):
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def so3_exp(phi):
    """SO(3) exponential map using Rodrigues' formula (NumPy only)."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    I = np.eye(3, dtype=np.float64)
    if theta < 1e-8:
        # sin(theta)/theta and (1-cos(theta))/theta^2 Taylor expansions.
        A = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        B = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
    return I + A * K + B * (K @ K)


def matrix_to_xyzw(Rm):
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    Rm = np.asarray(Rm, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(Rm))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (Rm[2, 1] - Rm[1, 2]) / s
        y = (Rm[0, 2] - Rm[2, 0]) / s
        z = (Rm[1, 0] - Rm[0, 1]) / s
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        s = np.sqrt(max(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2], 0.0)) * 2.0
        s = max(s, 1e-15)
        w = (Rm[2, 1] - Rm[1, 2]) / s
        x = 0.25 * s
        y = (Rm[0, 1] + Rm[1, 0]) / s
        z = (Rm[0, 2] + Rm[2, 0]) / s
    elif Rm[1, 1] > Rm[2, 2]:
        s = np.sqrt(max(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2], 0.0)) * 2.0
        s = max(s, 1e-15)
        w = (Rm[0, 2] - Rm[2, 0]) / s
        x = (Rm[0, 1] + Rm[1, 0]) / s
        y = 0.25 * s
        z = (Rm[1, 2] + Rm[2, 1]) / s
    else:
        s = np.sqrt(max(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1], 0.0)) * 2.0
        s = max(s, 1e-15)
        w = (Rm[1, 0] - Rm[0, 1]) / s
        x = (Rm[0, 2] + Rm[2, 0]) / s
        y = (Rm[1, 2] + Rm[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_xyzw_to_matrix(q):
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=np.float64)


def quat_slerp(q0, q1, u):
    """Spherical interpolation of xyzw quaternions."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / max(np.linalg.norm(q0), 1e-15)
    q1 = q1 / max(np.linalg.norm(q1), 1e-15)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    u = float(np.clip(u, 0.0, 1.0))
    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-15)
    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    s0 = np.sin((1.0 - u) * theta0) / sin_theta0
    s1 = np.sin(u * theta0) / sin_theta0
    q = s0 * q0 + s1 * q1
    return q / max(np.linalg.norm(q), 1e-15)


def so3_log(Rm):
    """SO(3) logarithm map returning an axis-angle rotation vector."""
    q = matrix_to_xyzw(Rm)
    # q and -q encode the same rotation; choose shortest representation.
    if q[3] < 0.0:
        q = -q
    v = q[:3]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return 2.0 * v
    angle = 2.0 * np.arctan2(nv, float(q[3]))
    return (angle / nv) * v


def make_T(Rm=None, t=None):
    T = np.eye(4, dtype=np.float64)
    if Rm is not None:
        T[:3, :3] = np.asarray(Rm, dtype=np.float64)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def inv_T(T):
    T = np.asarray(T, dtype=np.float64)
    Rm = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def transform_points(T, pts):
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def perturb_pose_left(T, dpose):
    """Left perturbation in world coordinates: translation + SO(3) rotation."""
    dpose = np.asarray(dpose, dtype=np.float64)
    out = np.asarray(T, dtype=np.float64).copy()
    out[:3, :3] = so3_exp(dpose[3:6]) @ out[:3, :3]
    out[:3, 3] = out[:3, 3] + dpose[:3]
    return out
