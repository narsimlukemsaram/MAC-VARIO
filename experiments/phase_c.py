import argparse
import multiprocessing as mp
from experiments.common import run_phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/euroc_mh02.yaml')
    args = parser.parse_args()
    run_phase('C', args.config)


if __name__ == '__main__':
    mp.freeze_support()
    main()
