"""CPU entry point for stage 7; see energy_protocol.md."""
import argparse
from threadpoolctl import threadpool_limits
from src.energy import run, replicate

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--replication-only',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.replication_only: replicate()
        else:
            run(resume=args.resume)
            from src.energy_report import generate
            generate()
