"""Select Joint-SAITS eta using original-initialization validation results."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs",type=Path,nargs=3,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
rows=[]
contract=None
for run in args.runs:
    config=json.loads((run/"config.json").read_text())
    status=json.loads((run/"status.json").read_text())
    metrics=json.loads((run/"joint_validation_metrics.json").read_text())
    if config["arm"]!="joint" or config["seed"]!=20261005 or status["status"]!="complete" or config["updates"]!=800:
        raise ValueError("Use three completed original-initialization Joint runs")
    current={k:config[k] for k in ("dataset","fit_ids","split_manifest_sha256","cache_manifest_sha256")}
    if contract is None: contract=current
    if current!=contract: raise ValueError("Joint candidate scopes differ")
    rows.append({"eta":config["eta"],"validation_mape_pct":metrics["missing_mape_pct"]})
if {row["eta"] for row in rows}!={.01,.1,.3}: raise ValueError("Candidates must cover eta=0.01,0.1,0.3")
selected=min(rows,key=lambda row:(row["validation_mape_pct"],row["eta"]))
if args.output.exists(): raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps({"dataset":contract["dataset"],"candidates":rows,"selected":selected,
    "test_used":False,"apply_selected_eta_to_other_initializations":True},indent=2),encoding="utf-8")
print(json.dumps(selected))
