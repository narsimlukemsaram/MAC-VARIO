"""Adapter around MAC-VO for MAC-VARIO.

The key design point is that MAC-VO's PointNode indices are *pair-local map
points*, not persistent landmark IDs across successive frame pairs.  Therefore
MAC-VARIO must consume each MAC-VO MatchObs directly as a two-frame factor
(pixel1/depth1 -> pixel2/depth2 with endpoint metric covariances), rather than
intersect PointNode IDs between neighbouring frames.

Ground-truth poses are not exposed to MAC-VO or to the VIO estimator.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import os
import pickle
import sys

import numpy as np
import torch


@dataclass
class VisualPairResult:
    frame0_idx: int
    frame1_idx: int
    time0_ns: int
    time1_ns: int
    T_WC0: np.ndarray
    T_WC1: np.ndarray
    T_BC: np.ndarray
    points_c0: np.ndarray
    points_c1: np.ndarray
    cov_c0: np.ndarray
    cov_c1: np.ndarray

    @property
    def valid(self):
        return (
            self.points_c0 is not None
            and self.points_c1 is not None
            and len(self.points_c0) >= 3
            and len(self.points_c0) == len(self.points_c1)
        )


class MACVOAdapter:
    def __init__(self, macvo_root, config_path, mav0, device="cuda:0"):
        self.root = Path(macvo_root).resolve()
        self.cfg = config_path
        self.mav0 = str(mav0)
        self.device = device
        self.results = {}
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

    def build(self):
        from Utility.Config import load_config
        from DataLoader import SequenceBase, StereoFrame, smart_transform
        from Odometry.MACVO import MACVO

        old = os.getcwd()
        os.chdir(self.root)
        try:
            cfg, _ = load_config(self.root / self.cfg)
            # Do not expose EuRoC ground-truth poses to the visual estimator.
            seq_cfg = SimpleNamespace(root=self.mav0, gt_pose=False)
            raw = SequenceBase[StereoFrame].instantiate("EuRoC_NoIMU", seq_cfg)
            self.sequence = smart_transform(raw, cfg.Preprocess)
            self.system = MACVO[StereoFrame].from_config(cfg)
        finally:
            os.chdir(old)

        # The hook fires after write-back of the previous optimisation and before
        # the next pair is inserted.  At that instant, prev_keyframe is frame k
        # and graph.get_frame2match(frame k) contains the incoming pair k-1 -> k.
        self.system.register_on_optimize_finish(self._hook)
        return self

    def _hook(self, system):
        if system.prev_keyframe is None:
            return

        graph = system.graph
        idx1 = int(system.prev_keyframe[1])

        try:
            frame1 = graph.frames[torch.tensor([idx1], dtype=torch.long)]
            matches_all = graph.get_frame2match(frame1)
            if len(matches_all) == 0:
                return

            # frame2match links each match to both endpoint frames.  Select only
            # matches for which the current frame is endpoint 2, i.e. the incoming
            # consecutive pair frame0 -> frame1.
            match_idx_all = matches_all.index
            frame2_ids = graph.match2frame2.mapping[match_idx_all]
            incoming = frame2_ids == idx1
            if int(incoming.sum().item()) == 0:
                return
            matches = matches_all[incoming]

            frame0_ids = graph.match2frame1.mapping[matches.index]
            uniq0 = torch.unique(frame0_ids)
            if len(uniq0) != 1:
                # This should not occur for a two-frame MAC-VO factor.  If it does,
                # keep the dominant predecessor rather than mixing frame pairs.
                values, counts = torch.unique(frame0_ids, return_counts=True)
                idx0 = int(values[torch.argmax(counts)].item())
                keep = frame0_ids == idx0
                matches = matches[keep]
            else:
                idx0 = int(uniq0[0].item())

            if len(matches) == 0:
                return

            frame0 = graph.frames[torch.tensor([idx0], dtype=torch.long)]
            frame1 = graph.frames[torch.tensor([idx1], dtype=torch.long)]

            from Utility.Point import pixel2point_NED
            import pypose as pp

            uv0 = matches.data["pixel1_uv"].float()
            uv1 = matches.data["pixel2_uv"].float()
            d0 = matches.data["pixel1_d"].float().reshape(-1)
            d1 = matches.data["pixel2_d"].float().reshape(-1)
            K0 = frame0.data["K"][0].float()
            K1 = frame1.data["K"][0].float()

            p0 = pixel2point_NED(uv0, d0, K0).double().cpu().numpy()
            p1 = pixel2point_NED(uv1, d1, K1).double().cpu().numpy()
            c0 = matches.data["obs1_covTc"].double().cpu().numpy()
            c1 = matches.data["obs2_covTc"].double().cpu().numpy()

            # Reject invalid measurements before they reach the nonlinear solver.
            good = (
                np.isfinite(p0).all(axis=1)
                & np.isfinite(p1).all(axis=1)
                & np.isfinite(c0).all(axis=(1, 2))
                & np.isfinite(c1).all(axis=(1, 2))
                & (d0.cpu().numpy() > 0)
                & (d1.cpu().numpy() > 0)
            )
            p0, p1, c0, c1 = p0[good], p1[good], c0[good], c1[good]
            if len(p0) == 0:
                return

            T_WC0 = pp.SE3(frame0.data["pose"][0].double().cpu()).matrix().squeeze().numpy()
            T_WC1 = pp.SE3(frame1.data["pose"][0].double().cpu()).matrix().squeeze().numpy()
            T_BC = pp.SE3(frame1.data["T_BS"][0].double().cpu()).matrix().squeeze().numpy()
            time0_ns = int(frame0.data["time_ns"][0].item())
            time1_ns = int(frame1.data["time_ns"][0].item())

            self.results[idx1] = VisualPairResult(
                frame0_idx=idx0,
                frame1_idx=idx1,
                time0_ns=time0_ns,
                time1_ns=time1_ns,
                T_WC0=T_WC0,
                T_WC1=T_WC1,
                T_BC=T_BC,
                points_c0=p0,
                points_c1=p1,
                cov_c0=c0,
                cov_c1=c1,
            )
        except Exception as exc:
            print(f"[MACVOAdapter] pair ending at frame {idx1} capture warning: {exc}")

    def run(self, max_frames=None):
        from Utility.Sandbox import Sandbox

        seq = self.sequence
        if max_frames is not None:
            class Limited:
                def __init__(self, s, n):
                    self.s = s
                    self.n = n

                def __len__(self):
                    return min(len(self.s), self.n)

                def __iter__(self):
                    for i, x in enumerate(self.s):
                        if i >= self.n:
                            break
                        yield x

                def __getattr__(self, name):
                    return getattr(self.s, name)

            seq = Limited(seq, max_frames)

        sb = Sandbox(Path("macvario_macvo_tmp").resolve())
        self.system.receive_frames(seq, sb)

        # register_on_optimize_finish is called before insertion of the next pair,
        # so the final pair has not been captured when the sequence ends.  Capture
        # it once explicitly after MAC-VO termination/write-back.
        self._hook(self.system)
        return self.results

    def save_cache(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.results, f)

    @staticmethod
    def load_cache(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Fail clearly if an obsolete cache from the point-ID implementation is used.
        if data:
            sample = next(iter(data.values()))
            if not hasattr(sample, "points_c0"):
                raise RuntimeError(
                    "MAC-VO cache was produced by the old point-ID adapter. "
                    "Delete the cache and rerun Phase B."
                )
        return data


def build_two_frame_visual(pair: VisualPairResult, min_points=20):
    """Build the chapter's direct C_k -> C_k+1 covariance-weighted factor."""
    from vio.visual_factor import VisualFactor

    n = len(pair.points_c0)
    if n < min_points:
        raise RuntimeError(f"Only {n} two-frame correspondences; need >= {min_points}")

    return VisualFactor(
        pair.points_c0,
        pair.points_c1,
        pair.cov_c0,
        pair.cov_c1,
        pair.T_BC,
    )
