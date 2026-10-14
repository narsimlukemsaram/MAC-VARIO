import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

from datasets.euroc import load_imu, load_gt, estimate_initial_imu_calibration
from vio.math_utils import inv_T, quat_xyzw_to_matrix


def body_from_camera(T_WC, T_BC):
    return T_WC @ inv_T(T_BC)


def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def read_gt_biases(mav0, query_time_ns):
    path = Path(mav0) / "state_groundtruth_estimate0" / "data.csv"

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            x = np.asarray(
                [float(v) for v in line.strip().split(",")],
                dtype=float,
            )

            rows.append(x)

    a = np.asarray(rows)

    ts = a[:, 0].astype(np.int64)

    idx = int(np.argmin(np.abs(ts - query_time_ns)))

    # EuRoC state layout:
    # timestamp,
    # p(3),
    # q_wxyz(4),
    # v(3),
    # gyro_bias(3),
    # accel_bias(3)

    q_wxyz = a[idx, 4:8]

    q_xyzw = np.array(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    )

    bg = a[idx, 11:14]
    ba = a[idx, 14:17]

    return q_xyzw, bg, ba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mav0 = Path(cfg["dataset"]["mav0"])

    with open(cfg["macvo"]["cache"], "rb") as f:
        pdict = pickle.load(f)

    pairs = [
        pdict[k]
        for k in sorted(pdict)
        if pdict[k].valid
    ]

    skip_s = float(
        cfg["dataset"].get("skip_initial_seconds", 0.0)
    )

    if skip_s > 0.0:
        tcut = pairs[0].time0_ns + int(skip_s * 1e9)

        pairs = [
            p for p in pairs
            if p.time0_ns >= tcut
        ]

    first = pairs[0]

    T_WB0 = body_from_camera(
        first.T_WC0,
        first.T_BC,
    )

    imu_ts, acc, gyro = load_imu(mav0)

    g_est, bg_est, ba_est = estimate_initial_imu_calibration(
        imu_ts,
        acc,
        gyro,
        T_WB0,
        first.time0_ns,
        cfg["imu"].get(
            "initial_stationary_window_s",
            1.0,
        ),
    )

    q_gt, bg_gt, ba_gt = read_gt_biases(
        mav0,
        first.time0_ns,
    )

    R_GT_B = quat_xyzw_to_matrix(q_gt)
    R_VIS_B = T_WB0[:3, :3]

    # Map the known gravity vector from EuRoC world
    # into the MAC-VO visual world.
    R_VIS_GT = R_VIS_B @ R_GT_B.T

    g_gt_world = np.array([0.0, 0.0, -9.81])

    g_true_visual = R_VIS_GT @ g_gt_world

    print("\nSequence:", cfg["dataset"]["sequence"])

    print("\n--- Gravity ---")
    print("estimated gravity:", g_est)
    print("GT-mapped gravity:", g_true_visual)

    print(
        "gravity direction error [deg]:",
        angle_deg(g_est, g_true_visual),
    )

    print("\n--- Gyro bias ---")
    print("estimated bg:", bg_est)
    print("GT bg:", bg_gt)
    print(
        "bias error norm:",
        np.linalg.norm(bg_est - bg_gt),
    )

    print("\n--- Accel bias ---")
    print("estimated ba:", ba_est)
    print("GT ba:", ba_gt)
    print(
        "bias error norm:",
        np.linalg.norm(ba_est - ba_gt),
    )


if __name__ == "__main__":
    main()