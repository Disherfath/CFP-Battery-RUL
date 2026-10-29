"""Synthetic numerical checks; execute on the research server, not locally."""
import json

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge

from batt_rul.reference_data import FEATURE_NAMES, viq18
from batt_rul.life_sensitive_features import FrozenLifeMetric, fit_life_metric, torch_viq18


@pytest.fixture
def training():
    generator = torch.Generator().manual_seed(823)
    raw = torch.randn((6, 50, 128, 3), generator=generator, dtype=torch.float64)
    raw = raw * torch.tensor([.12, .3, .18], dtype=torch.float64) + torch.tensor([3.5, -1., .8], dtype=torch.float64)
    raw += torch.arange(6, dtype=torch.float64)[:, None, None, None] * .07
    clean_features = torch_viq18(raw)
    features = clean_features[:, None].repeat(1, 10, 1)
    features[:, 1:] += torch.randn((6, 9, 18), generator=generator, dtype=torch.float64) * .03
    lifetime = torch.tensor([280., 310., 440., 520., 630., 760.], dtype=torch.float64)
    return raw, features, lifetime, [f"fit{i}" for i in range(6)]


@pytest.fixture
def state(training):
    raw, bank, life, ids = training
    return fit_life_metric(raw, bank, life, ids, alpha=.7)


def test_torch_features_match_reference_numpy_all18(training):
    raw = training[0]
    actual = torch_viq18(raw).numpy()
    expected = np.stack([viq18(cell) for cell in raw.numpy()])
    assert actual.shape == (6, 18) and actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
    assert list(FEATURE_NAMES[:6]) == ["V_mean", "V_std", "V_min", "V_max",
                                     "V_last5_minus_first5", "V_cycle_mean_slope_t01"]


def test_channel_major_order_and_normalized_cycle_slope():
    t = torch.linspace(0, 1, 50, dtype=torch.float64)[None, :, None, None]
    p = torch.linspace(0, 1, 128, dtype=torch.float64)[None, None, :, None]
    factors = torch.tensor([1., 2., 3.], dtype=torch.float64)[None, None, None]
    raw = factors * t + p + 10*factors
    features = torch_viq18(raw)
    torch.testing.assert_close(features[0, [5, 11, 17]], torch.tensor([1., 2., 3.], dtype=torch.float64))
    torch.testing.assert_close(features[0, [4, 10, 16]], torch.tensor([1., 2., 3.], dtype=torch.float64)*45/49)
    torch.testing.assert_close(features[0, [0, 6, 12]], torch.tensor([11., 21.5, 32.], dtype=torch.float64))


def test_constant_std_is_exact_zero_and_gradient_finite():
    raw = torch.ones((2, 50, 128, 3), dtype=torch.float64, requires_grad=True)
    features = torch_viq18(raw)
    assert torch.count_nonzero(features[:, [1, 7, 13]]) == 0
    features[:, [1, 7, 13]].sum().backward()
    assert torch.isfinite(raw.grad).all() and torch.count_nonzero(raw.grad) == 0
    raw.grad = None
    torch_viq18(raw).sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_channel_shift_changes_only_location_statistics(training):
    raw = training[0]
    shift = torch.tensor([.4, -2., 3.], dtype=torch.float64)
    before, after = torch_viq18(raw), torch_viq18(raw + shift)
    expected = before.reshape(-1, 3, 6).clone()
    expected[:, :, [0, 2, 3]] += shift[None, :, None]
    torch.testing.assert_close(after, expected.reshape(-1, 18), rtol=2e-12, atol=2e-12)


def test_scale_uses_li_weighted_views_not_clean_only(training, state):
    raw, bank, _, _ = training
    array = bank.numpy().reshape(-1, 18)
    weights = np.tile([.25] + [1/12]*9, len(raw))
    mean = np.average(array, axis=0, weights=weights)
    std = np.sqrt(np.average((array-mean)**2, axis=0, weights=weights))
    std[std < 1e-8] = 1.
    np.testing.assert_allclose(state["feature_mean"], mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state["feature_std"], std, rtol=1e-12, atol=1e-12)
    assert not np.allclose(mean, bank[:, 0].mean(dim=0).numpy())


def test_teacher_matches_sklearn_ridge_and_intercept(training, state):
    raw, _, lifetime, _ = training
    clean = np.stack([viq18(cell) for cell in raw.numpy()])
    x = (clean - np.asarray(state["feature_mean"])) / np.asarray(state["feature_std"])
    reference = Ridge(alpha=.7, fit_intercept=True, solver="cholesky").fit(x, np.log1p(lifetime.numpy()))
    np.testing.assert_allclose(state["coefficient"], reference.coef_, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(state["intercept"], reference.intercept_, rtol=2e-10, atol=2e-11)
    actual = FrozenLifeMetric(state).predict_loglife(raw).detach().numpy()
    np.testing.assert_allclose(actual, reference.predict(x), rtol=2e-10, atol=2e-11)


def test_direction_and_uniform_quadratics_have_unit_trace(state):
    metric = FrozenLifeMetric(state)
    assert not state["direction_degenerate"]
    torch.testing.assert_close(metric.direction.square().sum(), torch.tensor(1., dtype=torch.float64))
    task = metric.direction[:, None] * metric.direction[None, :]
    uniform = torch.eye(18, dtype=torch.float64)/18
    torch.testing.assert_close(task.trace(), uniform.trace())
    assert state["task_quadratic_trace"] == pytest.approx(1.)


@pytest.mark.parametrize("kind", ["uniform", "task"])
def test_losses_match_explicit_quadratic_and_no_target_teacher_gradient(training, state, kind):
    raw = training[0]
    complete = (raw + .025 * torch.linspace(0, 1, 50, dtype=torch.float64)[None, :, None, None]).requires_grad_()
    target = raw.clone().requires_grad_()
    metric = FrozenLifeMetric(state)
    assert list(metric.parameters()) == []
    assert all(not value.requires_grad for value in metric.buffers())
    delta = metric.discrepancy(complete, target)
    matrix = torch.eye(18, dtype=torch.float64)/18 if kind == "uniform" else torch.outer(metric.direction, metric.direction)
    expected = torch.einsum("bi,ij,bj->b", delta, matrix, delta).mean()
    actual = metric.loss(complete, target, kind)
    assert actual.dtype == torch.float64 and actual.ndim == 0
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert complete.grad is not None and torch.isfinite(complete.grad).all()
    assert torch.count_nonzero(complete.grad) > 0
    assert target.grad is None
    assert all(value.grad is None for value in metric.buffers())


@pytest.mark.parametrize("kind", ["uniform", "task"])
def test_feature_loss_gradient_matches_finite_difference(training, state, kind):
    clean = training[0]
    completion = (clean + .03).requires_grad_()
    metric = FrozenLifeMetric(state)
    loss = metric.loss(completion, clean, kind)
    gradient = torch.autograd.grad(loss, completion)[0]
    index, epsilon = (0, 17, 57, 0), 1e-5
    plus, minus = completion.detach().clone(), completion.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    finite_difference = (metric.loss(plus, clean, kind)-metric.loss(minus, clean, kind))/(2*epsilon)
    torch.testing.assert_close(gradient[index], finite_difference, rtol=2e-3, atol=2e-8)


def test_raw_li_bank_and_precomputed_feature_bank_fit_equivalently(training):
    clean, _, lifetime, ids = training
    raw_bank = clean[:, None].repeat(1, 10, 1, 1, 1)
    raw_bank[:, 1:] += torch.arange(1, 10, dtype=torch.float64)[None, :, None, None, None] * .001
    feature_bank = torch_viq18(raw_bank.reshape(-1, 50, 128, 3)).reshape(len(clean), 10, 18)
    raw_state = fit_life_metric(clean, raw_bank, lifetime, ids, alpha=1.)
    feature_state = fit_life_metric(clean, feature_bank, lifetime, ids, alpha=1.)
    for key in ("feature_mean", "feature_std", "coefficient", "intercept", "direction",
                "label_sha256", "scale_sha256", "li_feature_bank_sha256"):
        assert raw_state[key] == feature_state[key]


def test_degenerate_teacher_and_scale_floor_are_explicit():
    clean = torch.ones((3, 50, 128, 3), dtype=torch.float64)
    bank = clean[:, None].repeat(1, 10, 1, 1, 1)
    state = fit_life_metric(clean, bank, [100., 200., 300.], ["a", "b", "c"], alpha=1.)
    metric = FrozenLifeMetric(state)
    assert state["direction_degenerate"] and state["task_quadratic_trace"] == 0
    assert torch.equal(metric.feature_std, torch.ones(18, dtype=torch.float64))
    assert torch.count_nonzero(metric.direction) == 0
    completion = (clean + .1).requires_grad_()
    loss = metric.loss(completion, clean, "task")
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(completion.grad).all()


def test_fitting_detaches_inputs_and_state_is_json_serializable(training):
    clean, bank, lifetime, ids = training
    clean.requires_grad_(); bank.requires_grad_(); lifetime.requires_grad_()
    state = fit_life_metric(clean, bank, lifetime, ids, alpha=.7)
    restored = json.loads(json.dumps(state))
    metric = FrozenLifeMetric(restored)
    assert metric.to_dict() == restored
    assert restored["fit_ids"] == ids
    assert len(restored["label_sha256"]) == len(restored["scale_sha256"]) == 64
    assert clean.grad is None and bank.grad is None and lifetime.grad is None
    assert all(not value.requires_grad for value in metric.state_dict().values())


def test_teacher_diagnostics_match_loss_and_projection(training, state):
    clean = training[0]
    completion = clean + .01
    metric = FrozenLifeMetric(state)
    d = metric.diagnostics(completion, clean)
    assert all(not value.requires_grad for value in d.values())
    torch.testing.assert_close(d["uniform_loss_per_cell"].mean(), metric.loss(completion, clean, "uniform"))
    torch.testing.assert_close(d["task_loss_per_cell"].mean(), metric.loss(completion, clean, "task"))
    torch.testing.assert_close(d["teacher_loglife_difference"], d["direction_difference"]*torch.linalg.vector_norm(metric.coefficient),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("fault", ["clean_shape", "bank_shape", "bank_clean", "ids", "labels", "alpha", "nan"])
def test_invalid_fit_inputs_refused(training, fault):
    clean, bank, lifetime, ids = training
    alpha = 1.
    if fault == "clean_shape": clean = clean[:, :49]
    elif fault == "bank_shape": bank = bank[:, :9]
    elif fault == "bank_clean": bank[:, 0, 0] += 1
    elif fault == "ids": ids[0] = ids[1]
    elif fault == "labels": lifetime[0] = -1
    elif fault == "alpha": alpha = 0
    elif fault == "nan": bank[0, 1, 0] = float("nan")
    with pytest.raises(ValueError):
        fit_life_metric(clean, bank, lifetime, ids, alpha=alpha)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_double_buffers_finite_gradient_and_target_detachment(training, state):
    clean = training[0].cuda().requires_grad_()
    completion = (clean.detach() + .02).requires_grad_()
    metric = FrozenLifeMetric(state).cuda()
    loss = metric.loss(completion, clean, "task")
    loss.backward()
    assert loss.dtype == torch.float64 and torch.isfinite(loss)
    assert torch.isfinite(completion.grad).all() and clean.grad is None
    assert all(value.dtype == torch.float64 for value in metric.buffers())
