"""Portable orchestration tests; execute on the research server."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from batt_rul.paper_pipeline import score, select_backends, predict_backend, load_role, COUNTS
from batt_rul.protocol import build_views, schedule, sampling_schedule
from batt_rul import paper_pipeline


def test_condition_equal_scoring():
    # One clean view and three views per missing condition. Conditions, not
    # the ten repeated rows, receive equal weight in the all-condition metric.
    predicted=np.array([[1000.,1010.,1010.,1010.,1020.,1020.,1020.,1030.,1030.,1030.]])
    metrics=score(np.log1p(predicted+50.),np.array([1050.]))
    assert metrics["missing_mae"]==pytest.approx(20.)
    assert metrics["all_mae"]==pytest.approx(15.)
    assert metrics["missing_mape_pct"]==pytest.approx(2.)
    assert metrics["all_mape_pct"]==pytest.approx(1.5)
    assert metrics["missing_rmse"]==pytest.approx(np.sqrt(1400./3))


def test_views_have_exact_missing_amount_and_order():
    raw=np.zeros((50,128,4),dtype=np.float32)
    raw[...,0]=3.6
    raw[...,1]=-1.
    raw[...,2]=np.linspace(0.,1.,128)
    raw[...,3]=np.linspace(0.,100.,128)
    views=build_views(raw,"synthetic-cell")
    assert len(views)==10
    assert views[0]["mask"].all()
    assert [v["condition"] for v in views]==["clean"]+["mcar_time40"]*3+["block40"]*3+["window40"]*3
    for view in views[1:]:
        assert np.all((~view["mask"]).sum(axis=1)==51)
    repeated=build_views(raw,"synthetic-cell")
    for a,b in zip(views,repeated,strict=True):
        np.testing.assert_array_equal(a["mask"],b["mask"])


def test_schedule_contract():
    a,b=schedule(47),schedule(47)
    assert a["cell"].shape==(800,)
    assert a["view"].min()>=1 and a["view"].max()<=9
    np.testing.assert_array_equal(a["cell"],b["cell"])
    q=sampling_schedule(47,800,20261005)
    assert q["cell"].shape==(800,64)
    assert q["cycle"].min()>=0 and q["cycle"].max()<50


def test_backend_grid_and_frozen_prediction(tmp_path):
    rng=np.random.default_rng(12)
    x=rng.normal(size=(4,10,18))
    q=rng.normal(size=(2,10,18))
    train={"ids":["a","b","c","d"],"lifetime":np.array([800.,1000.,1200.,1600.])}
    val={"ids":["e","f"],"lifetime":np.array([900.,1400.])}
    chosen=select_backends(x,q,train,val,tmp_path/"backends")
    grid=pd.read_csv(tmp_path/"backends/validation_grid.csv")
    assert len(grid)==20
    assert grid.family.value_counts().to_dict()=={"rbf":15,"ridge":5}
    for family in ("ridge","rbf"):
        pred=predict_backend(tmp_path/"backends"/(family+".npz"),q)
        assert pred.shape==(2,10)
        assert score(pred,val["lifetime"])["missing_mape_pct"]==pytest.approx(chosen[family]["validation_mape_pct"],rel=1e-10)
        with np.load(tmp_path/"backends"/(family+".npz"),allow_pickle=False) as state:
            assert state["fit_ids"].tolist()==train["ids"]


def test_loader_opens_only_requested_role(tmp_path,monkeypatch):
    manifest=[]
    for cell,role in (("a","train"),("b","train"),("c","val"),("d","test")):
        # Only training files exist. A test/validation access would fail.
        filename=cell+".npz"
        if role=="train":
            raw=np.ones((50,128,4),dtype=np.float32)
            raw[...,3]=np.linspace(0.,100.,128)
            np.savez_compressed(tmp_path/filename,curves=raw)
        manifest.append(dict(cell_id=cell,cache_file=filename,rul_h50=1000.))
    pd.DataFrame(manifest).to_csv(tmp_path/"manifest.csv",index=False)
    pd.DataFrame([dict(cell_id=x,role=y) for x,y in (("a","train"),("b","train"),("c","val"),("d","test"))]).to_csv(tmp_path/"roles.csv",index=False)
    monkeypatch.setitem(COUNTS,"hust",{"train":2,"val":1,"test":1})
    context=load_role(tmp_path,"hust",tmp_path/"roles.csv","train")
    assert context["ids"]==["a","b"]
    truth=np.broadcast_to(context["clean"][:,None],context["linear"].shape)
    np.testing.assert_array_equal(context["linear"][context["masks"]],truth[context["masks"]])


def test_published_splits_have_expected_counts():
    config=Path(__file__).resolve().parents[1]/"configs"
    for name,expected in (("hust",{"train":47,"val":8,"test":22}),("xjtu",{"train":29,"val":7,"test":19})):
        roles=pd.read_csv(config/(name+"_roles.csv"),dtype={"cell_id":str})
        assert not roles.cell_id.duplicated().any()
        assert roles.role.value_counts().to_dict()==expected


def test_overflow_is_not_silently_discarded():
    with pytest.raises(FloatingPointError):
        score(np.full((1,10),1000.),np.array([1000.]))


def test_frozen_evaluation_configures_numeric_mode_before_data_access(tmp_path, monkeypatch):
    run=tmp_path/"run"
    run.mkdir()
    cache=tmp_path/"cache"
    cache.mkdir()
    (cache/"manifest.csv").write_text("cell_id,cache_file\n",encoding="utf-8")
    split=tmp_path/"roles.csv"
    split.write_text("cell_id,role\n",encoding="utf-8")
    config={"seed":20261005,"dataset":"hust","fit_ids":["train-cell"],
        "channel_stats":{"channel_mean":[0.,0.,0.],"channel_std":[1.,1.,1.]},
        "cache_manifest_sha256":paper_pipeline.file_sha256(cache/"manifest.csv"),
        "split_manifest_sha256":paper_pipeline.file_sha256(split)}
    (run/"config.json").write_text(json.dumps(config),encoding="utf-8")
    (run/"status.json").write_text(json.dumps({"status":"complete"}),encoding="utf-8")
    events=[]
    monkeypatch.setattr(paper_pipeline,"configure_rng",lambda seed:events.append(("numeric_mode",seed)))
    class StopAfterScopeCheck(Exception):
        pass
    def load(*args,**kwargs):
        assert events==[("numeric_mode",20261005)]
        raise StopAfterScopeCheck
    monkeypatch.setattr(paper_pipeline,"load_role",load)
    args=SimpleNamespace(run=run,cache=cache,splits=split,output=tmp_path/"evaluation",allow_smoke=False)
    with pytest.raises(StopAfterScopeCheck):
        paper_pipeline.evaluate(args)
