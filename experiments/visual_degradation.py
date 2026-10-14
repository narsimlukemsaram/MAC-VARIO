"""Image-degradation helper matching the chapter's gamma experiment.

MAC-VO must be rerun on degraded stereo images; this script creates a degraded
copy of EuRoC cam0/cam1 images so each gamma value gets its own visual cache.
"""
import argparse, shutil
from pathlib import Path
import cv2, numpy as np, yaml

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/euroc_mh02.yaml')
p.add_argument('--gammas',nargs='+',type=float,default=[1.0,1.5,2.5,3.5])
a=p.parse_args()
with open(a.config) as f: cfg=yaml.safe_load(f)
src=Path(cfg['dataset']['mav0'])
base=Path(cfg['output']['root'])/'degraded_datasets'/cfg['dataset']['sequence']
for gamma in a.gammas:
    dst=base/f'gamma_{gamma:g}'/'mav0'
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst)
    if gamma!=1.0:
        for cam in ('cam0','cam1'):
            for fp in (dst/cam/'data').glob('*.png'):
                im=cv2.imread(str(fp),cv2.IMREAD_UNCHANGED)
                x=np.clip(im.astype(np.float32)/255.0,0,1)
                y=np.power(x,gamma)
                cv2.imwrite(str(fp),np.clip(y*255,0,255).astype(np.uint8))
    print(f'Created {dst}')
print('Now duplicate the YAML per gamma, set dataset.mav0 to each generated mav0, use a unique macvo.cache, and run phases B/C/D.')
