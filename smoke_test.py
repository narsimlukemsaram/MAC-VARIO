"""Core synthetic smoke test; does not require MAC-VO/AirIMU datasets."""
import numpy as np
from vio.visual_factor import VisualFactor
from vio.imu_factor import NavState
from vio.preintegration import PreintegratedIMU
from vio.optimizer import TwoFrameVIOOptimizer

T0=np.eye(4); T1=np.eye(4); T1[0,3]=0.10
p0=np.array([[0,0,3],[1,0,4],[-1,.2,5],[.4,-.3,2.5]],float)
p1=p0.copy(); p1[:,0]-=.10
C=np.tile(np.eye(3)[None]*1e-3,(len(p0),1,1))
vis=VisualFactor(p0,p1,C,C,np.eye(4))
pre=PreintegratedIMU(np.eye(3),np.array([0,0,.981]),np.array([.10,0,.04905]),np.eye(9)*1e-2,.1)
state=NavState(T0,np.zeros(3))
opt=TwoFrameVIOOptimizer(gravity_W=[0,0,-9.81],max_iter=2,max_features=None)
r=opt.optimize(state,T1,np.array([1,0,0.0]),vis,pre)
assert np.isfinite(r.final_cost)
print('smoke test passed; final cost=',r.final_cost)
