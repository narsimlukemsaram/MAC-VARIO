from pathlib import Path
import argparse
import pickle
import numpy as np
import yaml

from datasets.euroc import load_imu, estimate_initial_imu_calibration
from vio.math_utils import inv_T


def body_from_camera(T_WC, T_BC):
    return T_WC @ inv_T(T_BC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mav0 = Path(cfg["dataset"]["mav0"])

    with open(cfg["macvo"]["cache"], "rb") as f:
        pair_dict = pickle.load(f)

    pairs = [
        pair_dict[k]
        for k in sorted(pair_dict)
        if pair_dict[k].valid
    ]

    skip_s = float(
        cfg["dataset"].get("skip_initial_seconds", 0.0)
    )

    if skip_s > 0:
        tcut = pairs[0].time0_ns + int(skip_s * 1e9)
        pairs = [p for p in pairs if p.time0_ns >= tcut]

    first = pairs[0]

    imu_ts, acc, gyro = load_imu(mav0)

    T0 = body_from_camera(first.T_WC0, first.T_BC)

    gravity, bg, ba = estimate_initial_imu_calibration(
        imu_ts,
        acc,
        gyro,
        T0,
        first.time0_ns,
        cfg["imu"].get("initial_stationary_window_s", 1.0),
    )

    window_s = cfg["imu"].get(
        "initial_stationary_window_s", 1.0
    )

    start_ns = first.time0_ns - int(window_s * 1e9)

    mask = (
        (imu_ts >= start_ns)
        & (imu_ts <= first.time0_ns)
    )

    A = acc[mask]
    G = gyro[mask]

    print("\nSequence:", cfg["dataset"]["sequence"])
    print("Initialisation samples:", len(A))
    print("Initial frame time:", first.time0_ns)

    print("\n--- Gyroscope ---")
    print("mean [rad/s]:", np.mean(G, axis=0))
    print("mean norm:", np.linalg.norm(np.mean(G, axis=0)))
    print("std [rad/s]:", np.std(G, axis=0))
    print("std norm:", np.linalg.norm(np.std(G, axis=0)))

    print("\n--- Accelerometer ---")
    print("mean [m/s^2]:", np.mean(A, axis=0))
    print("mean norm:", np.linalg.norm(np.mean(A, axis=0)))
    print("std [m/s^2]:", np.std(A, axis=0))
    print("std norm:", np.linalg.norm(np.std(A, axis=0)))

    print("\n--- Estimated initialisation ---")
    print("gravity_W:", gravity)
    print("|gravity|:", np.linalg.norm(gravity))
    print("gyro bias bg:", bg)
    print("|bg|:", np.linalg.norm(bg))
    print("accel bias ba:", ba)
    print("|ba|:", np.linalg.norm(ba))


if __name__ == "__main__":
    main()