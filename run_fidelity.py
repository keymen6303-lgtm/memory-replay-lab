"""CPU entry point for the stage-9 experiment and Chinese report."""
import argparse
from threadpoolctl import threadpool_limits
from src.fidelity import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(resume=args.resume)
        from src.fidelity_report import generate
        generate()
