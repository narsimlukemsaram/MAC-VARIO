"""AirIMU correction/covariance adapter with no GT-state inputs.

The precomputed AirIMU EuRoC bundle is generated on the IMU interval that overlaps
EuRoC ground truth, whereas MAC-VARIO reads the full raw IMU CSV. Therefore the
precomputed arrays can have a non-zero sample offset relative to the raw CSV.
This adapter estimates that offset from the IMU signals themselves (not GT pose,
velocity, orientation, or bias) and then maps raw-IMU indices to AirIMU indices.
"""
from pathlib import Path
import pickle, sys
import numpy as np
import torch


class AirIMUAdapter:
    def __init__(self):
        self.data = None
        self.raw_index_offset = 0
        self.alignment_score = None

    def load_precomputed(self, path, sequence):
        with open(path, 'rb') as f:
            full = pickle.load(f)
        if sequence not in full:
            cand = [k for k in full if sequence in k or k in sequence]
            if not cand:
                raise KeyError(f'{sequence} not found; keys={list(full)[:20]}')
            sequence = cand[0]

        d = full[sequence]
        self.data = {}
        for k in ('corrected_acc', 'corrected_gyro', 'acc_cov', 'gyro_cov',
                  'correction_acc', 'correction_gyro', 'dt'):
            if k not in d:
                continue
            v = d[k]
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 3 and v.shape[0] == 1:
                v = v[0]
            self.data[k] = v
        return self

    def align_to_raw(self, raw_acc, raw_gyro, max_offset=None, window=3000):
        """Estimate where AirIMU sample 0 lies in the full raw IMU arrays.

        AirIMU corrected measurements equal raw measurements plus relatively small
        learned corrections.  We therefore find the offset that minimizes a robust,
        scale-normalized discrepancy between the AirIMU corrected signals and the
        raw IMU signals over an early window.  This uses only sensor data.
        """
        if self.data is None:
            raise RuntimeError('load_precomputed() must be called first')

        raw_acc = np.asarray(raw_acc, dtype=np.float64)
        raw_gyro = np.asarray(raw_gyro, dtype=np.float64)
        ca = np.asarray(self.data['corrected_acc'], dtype=np.float64)
        cg = np.asarray(self.data['corrected_gyro'], dtype=np.float64)
        n_air = min(len(ca), len(cg))
        n_raw = min(len(raw_acc), len(raw_gyro))
        if n_air < 2 or n_raw < n_air:
            raise ValueError(f'Cannot align AirIMU length {n_air} to raw IMU length {n_raw}')

        feasible = n_raw - n_air
        if max_offset is None:
            max_offset = feasible
        max_offset = int(max(0, min(feasible, max_offset)))
        m = int(min(window, n_air))
        # Subsample for speed while retaining motion structure.
        stride = max(1, m // 800)
        aa = ca[:m:stride]
        gg = cg[:m:stride]

        # Robust scales prevent accelerometer magnitude from dominating gyro.
        sa = max(float(np.median(np.abs(aa - np.median(aa, axis=0)))), 1e-3)
        sg = max(float(np.median(np.abs(gg - np.median(gg, axis=0)))), 1e-4)

        best_offset, best_score = 0, np.inf
        for off in range(max_offset + 1):
            ra = raw_acc[off:off + m:stride]
            rg = raw_gyro[off:off + m:stride]
            if len(ra) != len(aa) or len(rg) != len(gg):
                continue
            # Median squared discrepancy is robust to short aggressive motions.
            ea = np.median(np.sum(((aa - ra) / sa) ** 2, axis=1))
            eg = np.median(np.sum(((gg - rg) / sg) ** 2, axis=1))
            score = float(0.25 * ea + eg)
            if score < best_score:
                best_score = score
                best_offset = off

        self.raw_index_offset = int(best_offset)
        self.alignment_score = float(best_score)
        return self.raw_index_offset, self.alignment_score

    def segment_raw_indices(self, raw_start, raw_end):
        """Return AirIMU samples corresponding to raw IMU indices [start,end)."""
        if self.data is None:
            raise RuntimeError('load_precomputed() must be called first')
        a0 = int(raw_start) - self.raw_index_offset
        a1 = int(raw_end) - self.raw_index_offset
        if a0 < 0 or a1 > len(self.data['corrected_acc']):
            return None
        return {k: v[a0:a1] for k, v in self.data.items() if k != 'dt'}

    def segment(self, start, end):
        # Kept for backwards compatibility; these are AirIMU-local indices.
        return {k: v[start:end] for k, v in self.data.items() if k != 'dt'}

    @classmethod
    def from_network(cls, airimu_root, conf_path, ckpt_path, device='cuda:0'):
        """Load CodeNet for correction/covariance inference only.

        We call network.inference(), not network.integrate(), so EuRoC GT orientation
        is never used for gravity compensation inside AirIMU.
        """
        sys.path.insert(0, str(Path(airimu_root).resolve()))
        from pyhocon import ConfigFactory
        from model.net import net_dict
        conf = ConfigFactory.parse_file(str(conf_path)); train_cfg = conf.train
        net = net_dict['codenet'](train_cfg).to(device).double()
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck['model_state_dict']); net.eval()
        obj = cls(); obj.net = net; obj.device = device
        return obj

    def infer_segment(self, acc, gyro):
        acc_t = torch.tensor(acc, dtype=torch.float64, device=self.device)[None]
        gyro_t = torch.tensor(gyro, dtype=torch.float64, device=self.device)[None]
        with torch.no_grad():
            out = self.net.inference({'acc': acc_t, 'gyro': gyro_t})
        corr_a = out['correction_acc'][0].cpu().numpy()
        corr_g = out['correction_gyro'][0].cpu().numpy()
        ac = out['cov_state']['acc_cov'][0].cpu().numpy()
        gc = out['cov_state']['gyro_cov'][0].cpu().numpy()
        return {
            'corrected_acc': np.asarray(acc) + corr_a,
            'corrected_gyro': np.asarray(gyro) + corr_g,
            'acc_cov': ac,
            'gyro_cov': gc,
        }
