"""Numerical tests are executed only on the research server."""
from pathlib import Path
import os

import pytest
import torch

from batt_rul.saits_task_adapter import (
    AUTHOR_COMMIT, AuthorSAITSAdapter, author_receipt, cycle_inputs, restore_observed,
)

AUTHOR = Path(os.environ.get("SAITS_ROOT", Path(__file__).resolve().parents[1] / "third_party" / "SAITS"))
pytestmark = pytest.mark.skipif(not (AUTHOR / "modeling/saits.py").is_file(), reason="Run scripts/setup_saits.py or set SAITS_ROOT")


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    return AuthorSAITSAdapter(AUTHOR, 20261005)


@pytest.fixture
def data():
    generator = torch.Generator().manual_seed(17)
    target = torch.randn((2, 128, 3), generator=generator)
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[:, 37:88, :] = False
    return target, observed


def test_author_receipt():
    assert author_receipt(AUTHOR)["commit"] == AUTHOR_COMMIT


def test_init_preserves_rng_cpu_cuda():
    cpu = torch.random.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    first, second = AuthorSAITSAdapter(AUTHOR, 2), AuthorSAITSAdapter(AUTHOR, 2)
    assert torch.equal(cpu, torch.random.get_rng_state())
    if cuda:
        assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))


def test_zero_hidden_and_target_loss_only(data):
    target, mask = data
    hidden_extreme = torch.where(mask, target, torch.full_like(target, 1e20))
    inputs = cycle_inputs(hidden_extreme, mask, target, stage="train")
    assert (inputs["X"][~mask] == 0).all()
    assert torch.equal(inputs["X"][mask], target[mask])
    assert torch.equal(inputs["indicating_mask"].bool(), ~mask)
    assert torch.equal(inputs["X_holdout"], target)
    assert "X_holdout" not in cycle_inputs(target, mask, stage="test")


def test_native_losses_forward_grad_and_observed_unchanged(model, data):
    target, mask = data
    model.train()
    model.zero_grad(set_to_none=True)
    result = model(target, mask, target, stage="train")
    direct = model.core(cycle_inputs(target, mask, target, stage="train"), "train")
    assert torch.equal(result["imputed_data"][mask], target[mask])
    assert torch.equal(result["reconstruction_loss"], direct["reconstruction_loss"])
    assert torch.equal(result["imputation_loss"], direct["imputation_loss"])
    assert torch.equal(result["loss"], direct["reconstruction_loss"] + direct["imputation_loss"])
    result["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any((g != 0).any() for g in grads)


def test_hidden_values_do_not_change_inference(model, data):
    target, mask = data
    model.eval()
    with torch.no_grad():
        first = model(target, mask)["imputed_data"]
        second = model(torch.where(mask, target, torch.full_like(target, -1e20)), mask)["imputed_data"]
    assert torch.equal(first, second)


def test_target_values_never_change_completion(model, data):
    target, mask = data
    model.eval()
    with torch.no_grad():
        first = model(target, mask, target, stage="val")
        second = model(target, mask, target + 10., stage="val")
    assert torch.equal(first["imputed_data"], second["imputed_data"])
    assert not torch.equal(first["imputation_loss"], second["imputation_loss"])


def test_restore_raw_observations_exact(data):
    target, mask = data
    raw = target.double() * 0.13 + torch.tensor([3., -0.1, 1.1], dtype=torch.float64)
    completed = restore_observed(target + 7., raw, mask, torch.zeros(3), torch.ones(3))
    assert torch.equal(completed[mask], raw[mask])
    assert torch.equal(completed[~mask], (target + 7.).double()[~mask])


@pytest.mark.parametrize("case", ["shape", "empty", "mask_shape", "mask_type", "nonfinite", "integer", "stage", "target", "all_visible", "all_hidden"])
def test_invalid_contract(data, case):
    target, mask = data
    args = [target, mask, target]
    stage = "train"
    if case == "shape": args[0] = target[:, :127]
    if case == "empty": args = [value[:0] for value in args]
    if case == "mask_shape": args[1] = mask[:, :, :2]
    if case == "mask_type": args[1] = mask.float()
    if case == "nonfinite": args[0] = target * float("nan")
    if case == "integer": args[0] = target.long()
    if case == "stage": stage = "bad"
    if case == "target": args[2] = None
    if case == "all_visible": args[1] = torch.ones_like(mask)
    if case == "all_hidden": args[1] = torch.zeros_like(mask)
    with pytest.raises(ValueError):
        cycle_inputs(*args, stage=stage)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_attention_device_and_restoration(data):
    model = AuthorSAITSAdapter(AUTHOR, 3).cuda()
    target, mask = (value.cuda() for value in data)
    result = model(target, mask, target, stage="train")
    result["loss"].backward()
    assert torch.equal(result["imputed_data"][mask], target[mask])
    assert all(module.device == target.device for module in model.core.modules() if hasattr(module, "device"))
