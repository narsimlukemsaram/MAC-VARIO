"""Run a reproducible covariance-floor sweep for Phase D.
Edit the YAML between runs or use this script as a template for the paper ablation.
"""
import argparse, copy, tempfile, yaml
from experiments.common import run_phase
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/euroc_mh02.yaml')
a=p.parse_args()
with open(a.config) as f: base=yaml.safe_load(f)
settings=[(1e-8,1e-10),(1e-6,1e-8),(1e-4,1e-6),(1e-3,1e-5)]
for ac,gc in settings:
    cfg=copy.deepcopy(base); cfg['airimu']['covariance_floor']={'acc':ac,'gyro':gc}
    with tempfile.NamedTemporaryFile('w',suffix='.yaml',delete=False) as f:
        yaml.safe_dump(cfg,f); path=f.name
    print(f'\n=== acc floor={ac:g}, gyro floor={gc:g} ===')
    run_phase('D',path)
