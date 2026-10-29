"""Train/validate a paper arm, then evaluate its frozen state separately."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from batt_rul.paper_pipeline import ARMS, train, evaluate

parser=argparse.ArgumentParser(description=__doc__)
sub=parser.add_subparsers(dest="command",required=True)
fit=sub.add_parser("train")
fit.add_argument("--dataset",choices=("hust","xjtu"),required=True)
fit.add_argument("--arm",choices=ARMS,required=True)
fit.add_argument("--seed",type=int)
fit.add_argument("--updates",type=int,default=800)
fit.add_argument("--feature-weight",type=float,default=.1)
fit.add_argument("--eta",type=float,choices=(.01,.1,.3))
fit.add_argument("--smoke",action="store_true",help="Mark a shortened engineering check, not a paper result")
test=sub.add_parser("evaluate")
test.add_argument("--run",type=Path,required=True)
test.add_argument("--allow-smoke",action="store_true")
for command in (fit,test):
    command.add_argument("--cache",type=Path,required=True)
    command.add_argument("--splits",type=Path,required=True)
    command.add_argument("--author-root",type=Path,default=Path("third_party/SAITS"))
    command.add_argument("--output",type=Path,required=True)
    command.add_argument("--device",default="cuda:0")
args=parser.parse_args()
if args.command=="train":
    if args.seed is None: args.seed=20261008 if args.arm=="t1" else 20261005
    if args.arm.startswith("brits-") and args.feature_weight!=.1:
        parser.error("Paper BRITS arms use feature weight 0.1")
with threadpool_limits(limits=1):
    (train if args.command=="train" else evaluate)(args)
