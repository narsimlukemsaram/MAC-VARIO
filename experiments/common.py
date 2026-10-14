from pathlib import Path
import json
import numpy as np

from datasets.euroc import (
    load_imu,
    load_gt,
    interpolate_gt,
    estimate_initial_imu_calibration,
)
from macvo.adapter import MACVOAdapter, build_two_frame_visual
from airimu.adapter import AirIMUAdapter
from vio.math_utils import inv_T, quat_xyzw_to_matrix
from vio.preintegration import IMUPreintegrator
from vio.imu_factor import NavState
from vio.optimizer import TwoFrameVIOOptimizer
from evaluation.trajectory_metrics import ate_rmse, rpe


def _rotation_angle_deg(R1, R2):
    """Return the geodesic rotation difference between two rotation matrices."""
    R = np.asarray(R1, dtype=np.float64).T @ np.asarray(R2, dtype=np.float64)
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _load_cfg(path):
    import yaml

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _T_from_pos_quat(p, q):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_matrix(q)
    T[:3, 3] = p
    return T


def _body_from_camera(T_WC, T_BC):
    return T_WC @ inv_T(T_BC)


def _pair_visual_kinematics(pair):
    """Return MAC-VO body poses and finite-difference velocity for one pair."""
    T0_vis = _body_from_camera(pair.T_WC0, pair.T_BC)
    T1_vis = _body_from_camera(pair.T_WC1, pair.T_BC)
    dt = max((pair.time1_ns - pair.time0_ns) * 1e-9, 1e-3)
    v_pair = (T1_vis[:3, 3] - T0_vis[:3, 3]) / dt
    return T0_vis, T1_vis, dt, v_pair


def _apply_preintegrated_covariance_floor(preint, floor_cfg):
    """Apply estimator-level floors to the 9x9 [R, v, p] covariance."""
    if preint is None or not floor_cfg:
        return preint

    P = np.asarray(preint.covariance, dtype=np.float64).copy()
    for start, key in ((0, "rotation"), (3, "velocity"), (6, "position")):
        floor = float(floor_cfg.get(key, 0.0))
        if floor <= 0.0:
            continue
        idx = np.arange(start, start + 3)
        P[idx, idx] = np.maximum(P[idx, idx], floor)

    preint.covariance = 0.5 * (P + P.T)
    return preint


def _valid_nav_state(state):
    """Check that an optimiser output contains finite pose and velocity values."""
    return (
        state is not None
        and np.all(np.isfinite(state.T_WB))
        and np.all(np.isfinite(state.velocity_W))
    )


def run_phase(phase, config_path, gamma=1.0):
    cfg = _load_cfg(config_path)
    phase = phase.upper()
    if phase not in ("B", "C", "D"):
        raise ValueError(f"Unsupported phase: {phase}. Expected B, C, or D.")

    mav0 = Path(cfg["dataset"]["mav0"])
    out = Path(cfg["output"]["root"]) / f"phase_{phase}" / cfg["dataset"]["sequence"]
    if gamma != 1.0:
        out = out / f"gamma_{gamma:g}"
    out.mkdir(parents=True, exist_ok=True)

    imu_ts, raw_acc, raw_gyro = load_imu(mav0)

    # MAC-VO pair factors are cached so B/C/D use identical visual observations.
    cache = Path(cfg["macvo"]["cache"])
    if cache.exists():
        pairs_dict = MACVOAdapter.load_cache(cache)
    else:
        mac = MACVOAdapter(
            cfg["macvo"]["root"],
            cfg["macvo"]["config"],
            mav0,
            cfg["runtime"]["device"],
        ).build()
        pairs_dict = mac.run(cfg["runtime"].get("max_frames"))
        mac.save_cache(cache)

    pairs = [pairs_dict[k] for k in sorted(pairs_dict) if pairs_dict[k].valid]

    skip_s = float(cfg["dataset"].get("skip_initial_seconds", 0.0))
    if pairs and skip_s > 0.0:
        t0_cut = pairs[0].time0_ns + int(skip_s * 1e9)
        pairs = [p for p in pairs if p.time0_ns >= t0_cut]

    if not pairs:
        raise RuntimeError("Need at least one valid MAC-VO two-frame factor")

    min_features = int(cfg["optimizer"].get("min_common_features", 20))
    diag_every = int(cfg.get("runtime", {}).get("drift_diagnostics_every", 100))

    # Initial estimator state comes only from the visual estimate and IMU data.
    first = pairs[0]
    T0, _, _, v0 = _pair_visual_kinematics(first)

    gravity, bg, ba = estimate_initial_imu_calibration(
        imu_ts,
        raw_acc,
        raw_gyro,
        T0,
        first.time0_ns,
        cfg["imu"].get("initial_stationary_window_s", 1.0),
    )

    # Optional calibrated values may override the no-GT initial estimate.
    if "gravity_W" in cfg["imu"]:
        gravity = np.asarray(cfg["imu"]["gravity_W"], dtype=np.float64)
    if "gyro_bias" in cfg["imu"]:
        bg = np.asarray(cfg["imu"]["gyro_bias"], dtype=np.float64)
    if "accel_bias" in cfg["imu"]:
        ba = np.asarray(cfg["imu"]["accel_bias"], dtype=np.float64)

    state = NavState(T_WB=T0.copy(), velocity_W=v0.copy())

    air = None
    if phase == "D":
        air = AirIMUAdapter().load_precomputed(
            cfg["airimu"]["pickle"],
            cfg["dataset"]["sequence"],
        )
        off, score = air.align_to_raw(
            raw_acc,
            raw_gyro,
            max_offset=cfg["airimu"].get("alignment_max_offset"),
            window=cfg["airimu"].get("alignment_window", 3000),
        )
        print(
            f"[Phase D] AirIMU/raw-IMU sample offset={off}, "
            f"alignment_score={score:.6g}"
        )

    pre = IMUPreintegrator(
        cfg["imu"].get("gyro_noise_density", 1.6968e-4),
        cfg["imu"].get("accel_noise_density", 2.0e-3),
    )

    opt = TwoFrameVIOOptimizer(
        gravity_W=gravity,
        imu_weight=0.0 if phase == "B" else cfg["optimizer"].get("imu_weight", 1.0),
        huber_delta=cfg["optimizer"].get("huber_delta", 2.5),
        lm_lambda=cfg["optimizer"].get("lm_lambda", 1e-4),
        max_iter=cfg["optimizer"].get("max_iter", 8),
        max_features=cfg["optimizer"].get("max_features"),
    )

    Ts = [T0.copy()]
    times = [first.time0_ns]
    previous_pair = None

    optimized_pairs = 0
    visual_fallback_pairs = 0
    optimizer_fallback_pairs = 0
    gap_reanchors = 0

    for k, pair in enumerate(pairs):
        T0_vis, T1_vis, dt, v1_init = _pair_visual_kinematics(pair)

        contiguous = (
            previous_pair is None
            or pair.frame0_idx == previous_pair.frame1_idx
        )

        # If MAC-VO dropped one or more pair intervals, explicitly add the current
        # pair's first visual pose so the trajectory timestamps remain synchronized.
        if previous_pair is not None and not contiguous:
            state = NavState(T_WB=T0_vis.copy(), velocity_W=v1_init.copy())
            gap_reanchors += 1

            if not times or times[-1] != pair.time0_ns:
                Ts.append(T0_vis.copy())
                times.append(pair.time0_ns)

            print(
                f"[{phase} pair {k}] re-anchor after MAC-VO pair gap: "
                f"{previous_pair.frame1_idx} -> {pair.frame0_idx}"
            )

        # Build the covariance-weighted visual factor. If the pair has too few
        # usable correspondences, advance with MAC-VO only rather than leaving the
        # estimator state at the previous timestamp.
        try:
            vf = build_two_frame_visual(pair, min_features)
        except RuntimeError as exc:
            state = NavState(T_WB=T1_vis.copy(), velocity_W=v1_init.copy())
            visual_fallback_pairs += 1

            if not times or times[-1] != pair.time1_ns:
                Ts.append(T1_vis.copy())
                times.append(pair.time1_ns)

            print(
                f"[{phase} pair {k}] visual-factor fallback: {exc}; "
                f"state advanced using MAC-VO pose"
            )

            previous_pair = pair
            continue

        # -------------------------------------------------------------
        # IMU preintegration for the current camera interval.
        # -------------------------------------------------------------
        im = None
        if phase in ("C", "D"):
            i0 = int(np.searchsorted(imu_ts, pair.time0_ns))
            i1 = int(np.searchsorted(imu_ts, pair.time1_ns, side="right"))

            if i1 - i0 >= 2:
                if phase == "C":
                    im = pre.integrate(
                        imu_ts[i0:i1],
                        raw_acc[i0:i1],
                        raw_gyro[i0:i1],
                        bg,
                        ba,
                    )
                    im = _apply_preintegrated_covariance_floor(
                        im,
                        cfg["imu"].get("preintegrated_covariance_floor"),
                    )

                else:
                    seg = air.segment_raw_indices(i0, i1)
                    if seg is None:
                        print(
                            f"[D pair {k}] AirIMU coverage does not include raw IMU "
                            f"indices [{i0},{i1}); using visual factor only"
                        )
                    else:
                        n = min(i1 - i0, len(seg["corrected_acc"]))
                        if n >= 2:
                            floors = cfg["airimu"].get(
                                "covariance_floor",
                                {"acc": 1e-6, "gyro": 1e-8},
                            )
                            ac = np.maximum(
                                seg["acc_cov"][:n],
                                float(floors["acc"]),
                            )
                            gc = np.maximum(
                                seg["gyro_cov"][:n],
                                float(floors["gyro"]),
                            )

                            im = pre.integrate(
                                imu_ts[i0:i0 + n],
                                seg["corrected_acc"][:n],
                                seg["corrected_gyro"][:n],
                                np.zeros(3),
                                np.zeros(3),
                                ac,
                                gc,
                            )
                            im = _apply_preintegrated_covariance_floor(
                                im,
                                cfg["imu"].get("preintegrated_covariance_floor"),
                            )

        # -------------------------------------------------------------
        # Diagnostic only: compare the previously estimated state against
        # the current MAC-VO anchor before re-anchoring this local problem.
        # -------------------------------------------------------------
        if diag_every > 0 and k % diag_every == 0:
            anchor_position_error = np.linalg.norm(
                state.T_WB[:3, 3] - T0_vis[:3, 3]
            )
            anchor_rotation_error = _rotation_angle_deg(
                state.T_WB[:3, :3],
                T0_vis[:3, :3],
            )
            anchor_velocity_error = np.linalg.norm(
                state.velocity_W - v1_init
            )
            print(
                f"[{phase} drift {k}] "
                f"pos={anchor_position_error:.4f} m, "
                f"rot={anchor_rotation_error:.3f} deg, "
                f"vel={anchor_velocity_error:.4f} m/s"
            )

        # -------------------------------------------------------------
        # Visual-anchored two-frame state.
        #
        # T_k is anchored to MAC-VO for each local optimisation. The start
        # velocity v_k is also derived from visual motion, avoiding recursive
        # propagation of an unconstrained/incomplete long-horizon inertial state.
        # -------------------------------------------------------------
        if previous_pair is not None and contiguous:
            T_prev0_vis = _body_from_camera(
                previous_pair.T_WC0,
                previous_pair.T_BC,
            )
            dt_prev = max(
                (pair.time0_ns - previous_pair.time0_ns) * 1e-9,
                1e-3,
            )
            v0_local = (
                T0_vis[:3, 3] - T_prev0_vis[:3, 3]
            ) / dt_prev
        else:
            # First pair or a discontinuity: current-pair finite difference is
            # the most conservative no-GT local velocity estimate available.
            v0_local = v1_init.copy()

        state_local = NavState(
            T_WB=T0_vis.copy(),
            velocity_W=v0_local.copy(),
        )

        # T1 and v1 are initialised from MAC-VO for this pair. The optimiser
        # adjusts them using the visual covariance factor and, in C/D, the IMU factor.
        T1_init = T1_vis.copy()

        try:
            res = opt.optimize(
                state_local,
                T1_init,
                v1_init,
                vf,
                im,
            )
            valid_result = bool(res.success) and _valid_nav_state(res.state_k1)
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            res = None
            valid_result = False
            print(f"[{phase} pair {k}] optimiser exception: {exc}")

        if valid_result:
            optimized_pairs += 1
            state = res.state_k1
            cost_text = f"{res.final_cost:.3f}"
            ok_text = "True"
        else:
            # A failed/non-finite local solve must not contaminate subsequent pairs.
            # Fall back to the current MAC-VO endpoint; GT is never used here.
            optimizer_fallback_pairs += 1
            state = NavState(
                T_WB=T1_vis.copy(),
                velocity_W=v1_init.copy(),
            )
            cost_text = "nan" if res is None else f"{res.final_cost:.3f}"
            ok_text = "False"
            print(
                f"[{phase} pair {k}] optimisation fallback: "
                f"cost={cost_text}; state advanced using MAC-VO pose"
            )

        if not times or times[-1] != pair.time1_ns:
            Ts.append(state.T_WB.copy())
            times.append(pair.time1_ns)

        print(
            f"{phase} {k + 1}/{len(pairs)}: "
            f"nvis={len(vf.points_c0)}, cost={cost_text}, ok={ok_text}"
        )

        previous_pair = pair

    Ts = np.asarray(Ts, dtype=np.float64)
    times = np.asarray(times, dtype=np.int64)

    if len(Ts) < 2:
        raise RuntimeError(
            "No usable two-frame estimates were produced. "
            "Check MAC-VO pair extraction and min_common_features."
        )

    np.savez(
        out / "trajectory.npz",
        timestamps_ns=times,
        T_WB=Ts,
    )

    # Ground truth is loaded only after estimation, for evaluation metrics.
    gt_ts, gt_pos, gt_q = load_gt(mav0)
    gp, gq = interpolate_gt(gt_ts, gt_pos, gt_q, times)
    gtT = np.asarray([_T_from_pos_quat(p, q) for p, q in zip(gp, gq)])

    ate = ate_rmse(Ts[:, :3, 3], gp, align=True)
    rr, tr = rpe(Ts, gtT)

    metrics = {
        "phase": phase,
        "sequence": cfg["dataset"]["sequence"],
        "ATE_RMSE_m": ate,
        "rrel_deg_per_frame": rr,
        "trel_m_per_frame": tr,
        "frames": len(Ts),
        # Keep the legacy key while making the accounting explicit.
        "visual_pairs": optimized_pairs,
        "optimized_pairs": optimized_pairs,
        "candidate_visual_pairs": len(pairs),
        "visual_fallback_pairs": visual_fallback_pairs,
        "optimizer_fallback_pairs": optimizer_fallback_pairs,
        "gap_reanchors": gap_reanchors,
    }

    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics
