"""CPU entry point for stage 8; see readout_protocol.md."""
import argparse
from threadpoolctl import threadpool_limits
from src.readout import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(resume=args.resume)
        from src.readout_report import generate
        generate()
