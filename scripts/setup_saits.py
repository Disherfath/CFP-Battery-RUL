"""Fetch the exact MIT-licensed SAITS checkout used by the paper."""
import argparse
from pathlib import Path
import subprocess
from batt_rul.saits_task_adapter import AUTHOR_COMMIT, author_receipt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination", type=Path, default=Path("third_party/SAITS"))
args = parser.parse_args()
if args.destination.exists():
    print(author_receipt(args.destination))
else:
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/WenjieDu/SAITS.git", str(args.destination)], check=True)
    subprocess.run(["git", "-C", str(args.destination), "checkout", "--detach", AUTHOR_COMMIT], check=True)
    print(author_receipt(args.destination))
