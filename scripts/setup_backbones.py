"""Download exact upstream backbones without redistributing their checkouts."""
import argparse
from pathlib import Path
import subprocess
from batt_rul.saits_task_adapter import AUTHOR_COMMIT as SAITS_COMMIT, author_receipt as saits_receipt
from batt_rul.brits_cycle_imputer import AUTHOR_COMMIT as BRITS_COMMIT, author_receipt as brits_receipt
from batt_rul.t1_cycle_adapter import AUTHOR_COMMIT as T1_COMMIT, _load_author

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("backbone",choices=("saits","brits","t1"))
parser.add_argument("--destination",type=Path)
args=parser.parse_args()
config={"saits":("WenjieDu/SAITS",SAITS_COMMIT,"SAITS"),
        "brits":("caow13/BRITS",BRITS_COMMIT,"BRITS"),
        "t1":("Oppenheimerdinger/T1",T1_COMMIT,"T1")}
repo,commit,name=config[args.backbone]
destination=args.destination or Path("third_party")/name
if not destination.exists():
    destination.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["git","clone","https://github.com/"+repo+".git",str(destination)],check=True)
    subprocess.run(["git","-C",str(destination),"checkout","--detach",commit],check=True)
if args.backbone=="saits": receipt=saits_receipt(destination)
elif args.backbone=="brits": receipt=brits_receipt(destination)
else: _,receipt=_load_author(destination)
print(receipt)
