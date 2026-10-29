"""Portable orchestration of the paper's frozen numerical building blocks.

Model adapters, feature losses, update functions, masks and backend fitting are
copied from the experiment implementation. This module replaces historical
run-directory dependencies with explicit cache, role and output arguments.
"""
from __future__ import annotations
import json
import hashlib
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

from . import saits_training, joint_training, brits_training, strong_training
from .backends import fit_backend
from .completion import ChannelStats, complete_queries
from .joint_rul_head import JointRULHead, fit_joint_reference
from .life_sensitive_features import FrozenLifeMetric, fit_life_metric
from .observed_cycle_completion import complete_observed_cycles
from .observed_only_features import observed_viq18
from .protocol import CONDITIONS, VIEW_CONDITION, VIEW_REPLICATE, build_views, schedule, sampling_schedule
from .reference_data import viq18
from .saits_task_adapter import AuthorSAITSAdapter
from .brits_cycle_imputer import BRITSCycleImputer
from .t1_within_cycle_adapter import T1WithinCycleAdapter
from .torch_data import load_cell_records

COUNTS = {"hust": {"train":47,"val":8,"test":22}, "xjtu":{"train":29,"val":7,"test":19}}
ARMS = ("base", "cfp", "direction", "joint", "saits64", "t1", "brits-base", "brits-masked", "brits-cfp", "brits-direction", "linear", "c0", "observed-only")
VIEW_WEIGHTS = np.array([.25] + [1./12.] * 9)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_role(cache, dataset, split_manifest, role):
    records = sorted(load_cell_records(Path(cache), dataset=dataset, history=50,
        split_manifest=Path(split_manifest), include_roles=(role,)), key=lambda r:r.cell_id)
    if len(records) != COUNTS[dataset][role]:
        raise ValueError(f"Expected {COUNTS[dataset][role]} {dataset} {role} cells")
    ids = [r.cell_id for r in records]
    clean = np.stack([r.curves[..., :3] for r in records]).astype(np.float64)
    banks, masks = [], []
    for record in records:
        views = build_views(record.curves, record.cell_id)
        mask = np.stack([v["mask"] for v in views])
        bank = np.stack([v["filled"] for v in views]).astype(np.float64)
        # Same observation-preservation repair used by every final paper run.
        bank = np.where(mask, record.curves[None, ..., :3], bank)
        banks.append(bank); masks.append(mask)
    return {"ids":ids, "clean":clean, "linear":np.stack(banks), "masks":np.stack(masks),
        "lifetime":np.asarray([r.lifetime for r in records], dtype=np.float64)}


def feature_bank(bank):
    return np.stack([np.stack([viq18(view) for view in cell]) for cell in bank])


def rule_features(role, arm):
    if arm == "observed-only":
        return np.stack([np.stack([observed_viq18(cell, mask) for mask in masks])
            for cell, masks in zip(role["clean"], role["masks"], strict=True)])
    bank = role["linear"].copy()
    if arm == "c0":
        for i in range(len(bank)):
            for view in range(1, 10):
                bank[i, view] = complete_observed_cycles(bank[i, view], role["masks"][i, view], mode="nearest4_offset").values
    return feature_bank(bank)


def decode(loglife):
    with np.errstate(over="raise", invalid="raise"):
        result = np.maximum(np.expm1(loglife) - 50., 0.)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite RUL prediction; no finite-subset scoring")
    return result


def score(loglife, lifetime):
    pred = decode(loglife)
    target = lifetime[:, None] - 50.
    error = pred - target
    result = {}
    for name, selection in (("missing",slice(1,10)), ("all",slice(None)), ("clean",slice(0,1))):
        e = error[:, selection]
        # Missing: each of 3 conditions has exactly 3 repetitions.
        # All: aggregate repetitions inside each condition first.
        w = VIEW_WEIGHTS if name == "all" else np.full(e.shape[1],1./e.shape[1])
        result[name+"_mape_pct"] = float(np.mean(np.sum(np.abs(e)/np.maximum(target,1.)*w,axis=1))*100)
        result[name+"_mae"] = float(np.mean(np.sum(np.abs(e)*w,axis=1)))
        result[name+"_rmse"] = float(np.sqrt(np.mean(np.sum(e*e*w,axis=1))))
    return result


def select_backends(train_features, val_features, train, val, output):
    """Fit to training cells, select by equal-missing-condition validation MAPE."""
    output.mkdir(exist_ok=True)
    raw = train_features.reshape(-1,18)
    weights = np.tile(VIEW_WEIGHTS, len(train["ids"]))
    y = np.repeat(np.log1p(train["lifetime"]),10)
    scaler = StandardScaler().fit(raw, sample_weight=weights)
    target_mean = float(np.average(y,weights=weights))
    head = {"x":scaler.transform(raw), "query":scaler.transform(val_features.reshape(-1,18)),
        "y":y, "weights":weights, "target_mean":target_mean}
    all_rows, selected = [], {}
    for family in ("ridge","rbf"):
        head["family"] = family
        grid = [(a,None) for a in (.01,.1,1.,10.,100.)] if family=="ridge" else [
            (a,g) for a in (.001,.01,.1,1.,10.) for g in (1./144.,1./36.,1./9.)]
        best = None
        for index,(alpha,gamma) in enumerate(grid):
            try:
                fitted, pred = fit_backend(head,alpha,gamma)
                metrics = score(pred.reshape(len(val["ids"]),10),val["lifetime"])
                row = dict(family=family,grid_index=index,alpha=alpha,gamma=gamma,status="finite",**metrics)
                if best is None or metrics["missing_mape_pct"] < best[0]:
                    best = (metrics["missing_mape_pct"],index,fitted,alpha,gamma)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                row = dict(family=family,grid_index=index,alpha=alpha,gamma=gamma,status="failed",error=type(exc).__name__)
            all_rows.append(row)
        if best is None:
            raise RuntimeError(f"All {family} validation candidates failed")
        loss,index,fitted,alpha,gamma = best
        state = dict(mean=scaler.mean_,scale=scaler.scale_,family=np.array(family),alpha=np.array(alpha),
            target_mean=np.array(target_mean),fit_ids=np.asarray(train["ids"]))
        if family=="ridge":
            state.update(coef=fitted.coef_,intercept=np.array(fitted.intercept_))
        else:
            state.update(x=head["x"],dual_coef=fitted.dual_coef_,gamma=np.array(gamma))
        np.savez_compressed(output/(family+".npz"),**state)
        selected[family]={"alpha":alpha,"gamma":gamma,"validation_mape_pct":loss,"grid_index":index}
    pd.DataFrame(all_rows).to_csv(output/"validation_grid.csv",index=False)
    dump(output/"selection.json",selected)
    return selected


def predict_backend(path, features):
    with np.load(path,allow_pickle=False) as s:
        z=(features.reshape(-1,18)-s["mean"])/s["scale"]
        if str(s["family"])=="ridge":
            values=z@s["coef"]+s["intercept"]
        else:
            values=rbf_kernel(z,s["x"],gamma=float(s["gamma"]))@s["dual_coef"]+s["target_mean"]
    return values.reshape(features.shape[:2])


def make_model(arm, author_root, stats, seed):
    if arm=="t1":
        return T1WithinCycleAdapter(author_root,torch.as_tensor(stats.mean,dtype=torch.float64),
            torch.as_tensor(stats.std,dtype=torch.float64),seed)
    if arm.startswith("brits-"):
        return BRITSCycleImputer(author_root,seed)
    return AuthorSAITSAdapter(author_root,seed)


def configure_rng(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)


def neural_completion(model, role, stats, device, arm):
    if arm=="t1":
        return strong_training.complete_queries("t1_within",model,role["linear"],role["masks"],stats,device)
    return complete_queries(model,role["linear"],role["masks"],stats,device)


def train(args):
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    if args.updates != 800 and not args.smoke:
        raise ValueError("Paper runs use 800 updates; shortened checks require --smoke")
    if not 1 <= args.updates <= 800:
        raise ValueError("updates must be between 1 and 800")
    configure_rng(args.seed)
    train_role=load_role(args.cache,args.dataset,args.splits,"train")
    stats=ChannelStats.fit(train_role["clean"],train_role["ids"])
    args.output.mkdir(parents=True)
    config={"schema":"cfp-public-run-v1","dataset":args.dataset,"arm":args.arm,"seed":args.seed,
        "updates":args.updates,"smoke":args.smoke,"feature_weight":args.feature_weight,
        "eta":args.eta if args.eta is not None else {"hust":.3,"xjtu":.1}[args.dataset],
        "fit_ids":train_role["ids"],"channel_stats":stats.receipt(),"test_opened":False,
        "split_manifest_sha256":file_sha256(args.splits),
        "cache_manifest_sha256":file_sha256(args.cache/"manifest.csv")}
    dump(args.output/"config.json",config)
    if args.arm in ("linear","c0","observed-only"):
        config["updates"]=0
        dump(args.output/"config.json",config)
        train_features=rule_features(train_role,args.arm)
    else:
        model=make_model(args.arm,args.author_root,stats,args.seed).to(args.device)
        network_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected_parameters=575177 if args.arm=="t1" else (37856 if args.arm.startswith("brits-") else 68126)
        if network_parameters!=expected_parameters:
            raise RuntimeError("Backbone parameter count differs from the paper")
        config["inference_network_parameters"]=network_parameters
        config["joint_head_parameters"]=641 if args.arm=="joint" else 0
        config["upstream_commit"]=model.receipt.get("commit")
        dump(args.output/"config.json",config)
        if args.arm=="saits64":
            # All three initializations use the same original reference schedule.
            plan=sampling_schedule(len(train_role["ids"]),800,20261005)
        else:
            plan=schedule(len(train_role["ids"]))
        np.savez_compressed(args.output/"schedule.npz",**plan)
        metric_state=fit_life_metric(train_role["clean"],train_role["linear"],train_role["lifetime"],
            train_role["ids"],alpha={"hust":10.,"xjtu":1.}[args.dataset])
        metric=FrozenLifeMetric(metric_state).to(args.device)
        dump(args.output/"feature_metric.json",metric_state)
        joint_head=None
        head_ref=None
        if args.arm=="joint":
            head_ref=fit_joint_reference(train_role["clean"],train_role["linear"],train_role["lifetime"],train_role["ids"])
            head_seed={20261005:20261018,20261018:20271018,20261019:20271019}.get(args.seed,args.seed+10000)
            joint_head=JointRULHead(head_ref,head_seed).to(args.device)
        parameters=list(model.parameters())+(list(joint_head.parameters()) if joint_head is not None else [])
        optimizer=torch.optim.Adam(parameters,lr=.001)
        raw=torch.as_tensor(train_role["clean"],dtype=torch.float64,device=args.device)
        masks=torch.as_tensor(train_role["masks"],dtype=torch.bool,device=args.device)
        mean,std=(torch.as_tensor(x,dtype=torch.float64,device=args.device) for x in (stats.mean,stats.std))
        target_z=torch.as_tensor(stats.transform(train_role["clean"]),device=args.device)
        history=[]
        configure_rng(args.seed)
        for step in range(args.updates):
            if args.arm in ("saits64","t1"):
                row=strong_training.train_step("saits64" if args.arm=="saits64" else "t1_within",
                    model,optimizer,raw,masks,plan,step,mean,std,target_fit_z=target_z)
            else:
                i,v=int(plan["cell"][step]),int(plan["view"][step])
                x,m=raw[i:i+1],masks[i:i+1,v]
                if args.arm=="joint":
                    labels=torch.as_tensor(train_role["lifetime"][i:i+1],dtype=torch.float64,device=args.device)
                    row=joint_training.fit_step(model,joint_head,optimizer,x,m,mean,std,labels,config["eta"])
                elif args.arm.startswith("brits-"):
                    arm={"cfp":"uniform","direction":"task"}.get(args.arm[6:],args.arm[6:])
                    row=brits_training.fit_step(model,optimizer,x,m,mean,std,metric,arm)
                else:
                    arm={"cfp":"uniform","direction":"task"}.get(args.arm,args.arm)
                    row=saits_training.fit_step(model,optimizer,x,m,mean,std,metric,arm,auxiliary_weight=args.feature_weight)
            history.append(dict(step=step+1,**row))
            if (step+1)%50==0 or step+1==args.updates:
                print(f"{args.dataset} {args.arm} {step+1}/{args.updates} loss={row['loss']:.6g}",flush=True)
        pd.DataFrame(history).to_csv(args.output/"training.csv",index=False)
        checkpoint={"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":config,
            "head_reference":head_ref,"head":{k:v.detach().cpu() for k,v in joint_head.state_dict().items()} if joint_head else None}
        torch.save(checkpoint,args.output/"model.pt")
        completed=neural_completion(model,train_role,stats,args.device,args.arm)
        train_features=feature_bank(completed)
    # Validation arrays are opened only after model training is finished.
    val=load_role(args.cache,args.dataset,args.splits,"val")
    if set(train_role["ids"]) & set(val["ids"]):
        raise ValueError("Training and validation identities overlap")
    if args.arm in ("linear","c0","observed-only"):
        val_features=rule_features(val,args.arm)
    else:
        completed=neural_completion(model,val,stats,args.device,args.arm)
        val_features=feature_bank(completed)
        if joint_head is not None:
            dump(args.output/"joint_validation_metrics.json",score(joint_training.predict_joint(joint_head,completed,args.device),val["lifetime"]))
    np.savez_compressed(args.output/"descriptor_banks.npz",train=train_features,val=val_features,
        train_ids=np.asarray(train_role["ids"]),val_ids=np.asarray(val["ids"]))
    selected=select_backends(train_features,val_features,train_role,val,args.output/"backends")
    dump(args.output/"status.json",{"status":"smoke_complete" if args.smoke else "complete",
        "completed_updates":config["updates"],"test_opened":False,"selected":selected})


def evaluate(args):
    config=json.loads((args.run/"config.json").read_text())
    # Match the original frozen-test numerical settings in a fresh process.
    # In particular, cuDNN TF32 defaults can otherwise alter T1 convolutions.
    configure_rng(config["seed"])
    status=json.loads((args.run/"status.json").read_text())
    if status["status"]!="complete" and not args.allow_smoke:
        raise ValueError("Complete a formal training/validation run first")
    if file_sha256(args.splits)!=config["split_manifest_sha256"] or file_sha256(args.cache/"manifest.csv")!=config["cache_manifest_sha256"]:
        raise ValueError("Training and test must use the same frozen cache and split manifests")
    output=args.output
    if output.exists():
        raise FileExistsError("Choose a new test-output directory")
    stats_state=config["channel_stats"]
    stats=ChannelStats(tuple(config["fit_ids"]),np.asarray(stats_state["channel_mean"]),np.asarray(stats_state["channel_std"]))
    test=load_role(args.cache,config["dataset"],args.splits,"test")
    if set(test["ids"]) & set(config["fit_ids"]):
        raise ValueError("Test identities overlap with fitted cells")
    output.mkdir(parents=True)
    results={}
    arm=config["arm"]
    if arm in ("linear","c0","observed-only"):
        features=rule_features(test,arm)
    else:
        model=make_model(arm,args.author_root,stats,config["seed"]).to(args.device)
        checkpoint=torch.load(args.run/"model.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["model"],strict=True)
        # Frozen-test fastpath matches the original T1 test evaluator.
        model.requires_grad_(False)
        completed=neural_completion(model,test,stats,args.device,arm)
        features=feature_bank(completed)
        if arm=="joint":
            head=JointRULHead(checkpoint["head_reference"]).to(args.device)
            head.load_state_dict(checkpoint["head"],strict=True)
            results["joint_mlp"]=joint_training.predict_joint(head,completed,args.device)
    for family in ("ridge","rbf"):
        results[family]=predict_backend(args.run/"backends"/(family+".npz"),features)
    summary,rows={},[]
    for family,loglife in results.items():
        summary[family]=score(loglife,test["lifetime"])
        predicted=decode(loglife)
        for i,cell in enumerate(test["ids"]):
            for v in range(10):
                rows.append(dict(dataset=config["dataset"],arm=arm,seed=config["seed"],family=family,
                    cell_id=cell,condition=CONDITIONS[VIEW_CONDITION[v]],replicate=int(VIEW_REPLICATE[v]),
                    target=float(test["lifetime"][i]-50),prediction_loglife=float(loglife[i,v]),prediction=float(predicted[i,v])))
    pd.DataFrame(rows).to_csv(output/"predictions.csv",index=False)
    dump(output/"metrics.json",summary)
    dump(output/"receipt.json",{"dataset":config["dataset"],"test_ids":test["ids"],"training_seed":config["seed"],
        "smoke":config["smoke"],"network_training":False,"backend_fitting":False})
