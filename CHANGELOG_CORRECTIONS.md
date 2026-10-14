# Corrections made from the student's implementation

- Removed per-frame EuRoC GT rotation from IMU residuals.
- Removed GT velocity initialisation.
- Removed GT IMU biases from Phase C.
- Replaced map/world-point visual residual with true two-frame camera-frame correspondence residual.
- Added point-ID intersection between consecutive MAC-VO frames.
- Endpoint covariance is propagated as `R Sigma_k R^T + Sigma_k+1`.
- AirIMU network adapter uses correction/covariance inference only; it does not request GT orientation for integration.
- Added standalone covariance-floor and gamma-degradation experiment scripts.
- Separated evaluation so GT is used only after estimation.

- Removed SciPy dependency to avoid Windows Application Control blocking compiled `_ckdtree.pyd`; added NumPy-only SO(3), quaternion, and SLERP utilities.
