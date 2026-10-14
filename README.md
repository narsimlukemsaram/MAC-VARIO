# MAC-VARIO: Robust Visual-Inertial Odometry with Learned Metrics-Aware Covariance Fusion for Autonomous Aerial Vehicles in GPS-Denied Environments

This repository contains MAC-VARIO code to match the concept below:
- MAC-VO visual correspondences with metric 3-D covariance
- AirIMU corrected measurements and learned uncertainty
- IMU preintegration with 9x9 covariance
- true consecutive-frame visual residuals `C_k -> C_{k+1}`
- covariance-weighted visual + inertial two-frame optimisation
- **no EuRoC ground-truth pose, orientation, velocity or bias inside the estimator**
- ground truth is loaded only after estimation for ATE/RPE evaluation.

## Repository layout

```text
MAC_VARIO/
├── macvo/
│   └── adapter.py
├── airimu/
│   └── adapter.py
├── vio/
│   ├── preintegration.py
│   ├── visual_factor.py
│   ├── imu_factor.py
│   └── optimizer.py
├── datasets/
│   └── euroc.py
├── experiments/
│   ├── phase_b.py
│   ├── phase_c.py
│   ├── phase_d.py
│   ├── covariance_floor.py
│   └── visual_degradation.py
├── evaluation/
│   └── trajectory_metrics.py
├── configs/
├── checkpoints/
└── results/
```

## Important methodological corrections

1. `GT velocity` is **not** used for initialization. Initial velocity comes from the first two MAC-VO body poses.
2. `GT orientation` is **not** used for gravity projection. Gravity and biases are estimated from an initial IMU window, or supplied from calibration.
3. Phase C does **not** use EuRoC ground-truth IMU biases.
4. The visual factor intersects MAC-VO landmark IDs in consecutive frames and uses an actual two-frame point residual.
5. Ground truth is imported only in `experiments/common.py` after trajectory estimation to compute metrics.
6. The provided AirIMU adapter supports network **inference** of corrections/covariance without calling AirIMU's GT-orientation-assisted integration.

## Windows setup

Recommended environment: Python 3.10/3.11, NVIDIA CUDA build of PyTorch if using GPU.

```powershell
cd C:\GitHub\MAC_VARIO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Keep the upstream repositories separate, for example:

```text
C:\GitHub\MAC-VO-main
C:\GitHub\AirIMU-main
C:\GitHub\MAC_VARIO
```

Install MAC-VO and AirIMU dependencies according to their upstream READMEs. Copy the required pretrained weights into those upstream repositories, and do not commit large checkpoints to this repository.

## Configure EuRoC

Edit `configs/euroc_mh02.yaml` and set:

- `dataset.mav0`
- `macvo.root`
- `macvo.config`
- `airimu.pickle` for Phase D

For a quick CPU syntax/integration check, set `runtime.device: cpu` and `runtime.max_frames: 50`.

## Run

From the repository root:

```powershell
python -m experiments.phase_b --config configs/euroc_mh02.yaml
python -m experiments.phase_c --config configs/euroc_mh02.yaml
python -m experiments.phase_d --config configs/euroc_mh02.yaml
```

Phases B/C/D share a MAC-VO cache, so the visual front end stays constant.

## Phase definitions

- **B — MAC-VO:** visual-only (`imu_weight=0`).
- **C — MAC-VIO:** raw IMU, fixed manufacturer/noise-density covariance, no GT bias.
- **D — MAC-VARIO:** AirIMU corrected IMU + learned covariance + the same MAC-VO visual factors.

## Reproducing the paper ablations

```powershell
python -m experiments.covariance_floor --config configs/euroc_mh02.yaml
python -m experiments.visual_degradation --config configs/euroc_v201.yaml
```

The degradation script creates gamma-degraded copies of EuRoC images. Each degraded copy must have a unique MAC-VO cache and then be rerun through B/C/D.

## Scientific-use note

Rerun the corrected pipeline and update the paper from the newly generated `metrics.json` files. This repository intentionally contains no hard-coded target metrics.


## Windows note: SciPy/App Control

This corrected v2 repository does not require SciPy. Earlier versions imported
`scipy.spatial.transform`, which can trigger Windows Application Control when a
compiled SciPy DLL such as `_ckdtree.pyd` is blocked. The required SO(3),
quaternion, and SLERP operations are now implemented with NumPy in
`vio/math_utils.py`.

If upgrading an existing environment, remove SciPy if desired:

```powershell
python -m pip uninstall -y scipy
python -m pip install -r requirements.txt
```
