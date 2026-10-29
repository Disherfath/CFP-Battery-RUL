"""Frozen fit-only lifetime direction for differentiable VIQ feature losses.

The caller supplies training cells explicitly and restores true observations
before calling the loss. This module has no dataset loader, split discovery,
checkpoint selection, or access to validation/test data. It adds no trainable
parameters and is needed only while training the imputer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Sequence

import torch
from torch import nn

from .reference_data import FEATURE_NAMES


def _raw_tensor(raw: torch.Tensor, name: str = "raw") -> None:
    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.float64:
        raise ValueError(f"{name} must be a float64 torch tensor")
    if raw.ndim != 4 or raw.shape[1:] != (50, 128, 3) or len(raw) == 0:
        raise ValueError(f"{name} must be nonempty [B,50,128,3]")
    if not torch.isfinite(raw).all():
        raise ValueError(f"{name} must be finite; completion must be explicit")


def torch_viq18(raw: torch.Tensor) -> torch.Tensor:
    """Differentiable [B,18], matching reference_data.viq18 definitions/order.

    Features are channel-major V/I/Q, with mean, population standard deviation,
    minimum, maximum, last-five minus first-five cycle mean, and cycle-mean
    slope on t=linspace(0,1,50). No epsilon is added to the population variance.
    At exactly zero variance the std is exactly zero and uses a zero derivative,
    avoiding the undefined derivative of sqrt(0).
    """
    _raw_tensor(raw)
    mean = raw.mean(dim=(1, 2))
    centered = raw - mean[:, None, None, :]
    variance = centered.square().mean(dim=(1, 2))
    positive = variance > 0
    safe_variance = torch.where(positive, variance, torch.ones_like(variance))
    std = torch.where(positive, torch.sqrt(safe_variance), torch.zeros_like(variance))
    minimum = raw.amin(dim=(1, 2))
    maximum = raw.amax(dim=(1, 2))
    drift = raw[:, -5:].mean(dim=(1, 2)) - raw[:, :5].mean(dim=(1, 2))
    t = torch.linspace(0., 1., 50, dtype=torch.float64, device=raw.device)
    centered_t = t - t.mean()
    cycle_means = raw.mean(dim=2)
    slope = (cycle_means * centered_t[None, :, None]).sum(dim=1) / centered_t.square().sum()
    features = torch.stack((mean, std, minimum, maximum, drift, slope), dim=-1).reshape(len(raw), 18)
    if not torch.isfinite(features).all():
        raise ValueError("VIQ features overflowed float64")
    return features


def _fit_tensor(value, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.dtype != torch.float64:
        raise ValueError(f"{name} tensor must be float64")
    result = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _array_hash(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    header = json.dumps({"shape": list(array.shape), "dtype": "float64"}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()


@torch.no_grad()
def fit_life_metric(clean_fit, li_fit_bank, lifetimes, fit_ids: Sequence[str], *,
                    alpha: float) -> dict:
    """Return JSON-serializable state from supplied fit cells only.

    clean_fit: raw [N,50,128,3].
    li_fit_bank: raw [N,10,50,128,3] or VIQ feature bank [N,10,18].
      View 0 must be clean. Its weight is 1/4; each other view has weight 1/12;
      physical cells are equally weighted. No other views are accepted.
    lifetimes: [N], converted here to log1p(lifetime).
    alpha: strictly positive Ridge penalty for sum-of-squared-errors fitting,
      equivalent to sklearn Ridge(alpha=alpha, fit_intercept=True).

    The feature scale is fitted to the LI views. The teacher is fitted to CLEAN
    fit features using that same scale, with an unpenalized intercept. Target and
    X centering handle the intercept only; there is no extra scale fit on clean
    data. u is the unit teacher coefficient direction, or zero if degenerate.
    """
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("alpha must be finite and strictly positive")
    clean = _fit_tensor(clean_fit, "clean_fit")
    _raw_tensor(clean, "clean_fit")
    ids = list(fit_ids)
    if len(ids) != len(clean) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("fit_ids must be unique nonempty strings in supplied cell order")
    lifetime = _fit_tensor(lifetimes, "lifetimes")
    if lifetime.shape != (len(clean),) or (lifetime < 0).any():
        raise ValueError("lifetimes must be nonnegative [N] in fit-cell order")
    bank = _fit_tensor(li_fit_bank, "li_fit_bank")
    if bank.shape == (len(clean), 10, 50, 128, 3):
        features = torch_viq18(bank.reshape(-1, 50, 128, 3)).reshape(len(clean), 10, 18)
        source_kind = "raw LI completion bank"
    elif bank.shape == (len(clean), 10, 18):
        features = bank
        source_kind = "LI VIQ18 feature bank"
    else:
        raise ValueError("li_fit_bank must be [N,10,50,128,3] or [N,10,18]")
    clean_features = torch_viq18(clean)
    if not torch.allclose(features[:, 0], clean_features, rtol=1e-10, atol=1e-10):
        raise ValueError("LI bank view 0 must be the same clean fit-cell features")
    view_weights = torch.tensor([0.25] + [1. / 12] * 9, dtype=torch.float64)
    weights = view_weights[None, :].expand(len(clean), -1).reshape(-1).clone()
    weights /= weights.sum()
    flat = features.reshape(-1, 18)
    feature_mean = (flat * weights[:, None]).sum(dim=0)
    variance = ((flat - feature_mean).square() * weights[:, None]).sum(dim=0)
    raw_std = torch.sqrt(variance.clamp_min(0))
    feature_std = torch.where(raw_std < 1e-8, torch.ones_like(raw_std), raw_std)
    x = (clean_features - feature_mean) / feature_std
    y = torch.log1p(lifetime)
    x_mean, y_mean = x.mean(dim=0), y.mean()
    centered_x, centered_y = x - x_mean, y - y_mean
    normal = centered_x.T @ centered_x + float(alpha) * torch.eye(18, dtype=torch.float64)
    coefficient = torch.linalg.solve(normal, centered_x.T @ centered_y)
    intercept = y_mean - x_mean @ coefficient
    coefficient_norm = torch.linalg.vector_norm(coefficient)
    degenerate = coefficient_norm.item() <= 1e-12
    direction = torch.zeros_like(coefficient) if degenerate else coefficient / coefficient_norm
    if not all(torch.isfinite(v).all() for v in (feature_mean, feature_std, coefficient, intercept, direction)):
        raise ValueError("nonfinite fit-only task metric")
    return {
        "schema": "frozen-life-sensitive-viq18-v1", "fit_ids": ids,
        "feature_names": list(FEATURE_NAMES), "history": 50, "points": 128,
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "coefficient": coefficient.tolist(), "intercept": intercept.item(),
        "direction": direction.tolist(), "coefficient_norm": coefficient_norm.item(),
        "direction_degenerate": degenerate, "ridge_alpha": float(alpha),
        "teacher_target": "log1p(lifetime)", "teacher_source": "clean fit cells only",
        "scale_source": source_kind, "view_weights": view_weights.tolist(),
        "view_order": "clean first; nine missing views", "physical_cell_weight": "equal",
        "scale_std_floor_rule": "population std below 1e-8 replaced by 1",
        "label_sha256": _array_hash(y), "lifetime_sha256": _array_hash(lifetime),
        "scale_sha256": _array_hash(torch.stack((feature_mean, feature_std))),
        "li_feature_bank_sha256": _array_hash(features),
        "clean_feature_sha256": _array_hash(clean_features),
        "uniform_quadratic_trace": 1.0, "task_quadratic_trace": direction.square().sum().item(),
        "degenerate_rule": "zero task direction and zero task loss; report explicitly",
    }


class FrozenLifeMetric(nn.Module):
    """Frozen double buffers, differentiable only through the completed curves.

    uniform = mean(delta_z**2), corresponding to I/18 (trace 1).
    task = mean((delta_z @ u)**2), corresponding to uu^T (trace 1 when
    nondegenerate). A degenerate fitted teacher has u=0 and task trace/loss 0.
    """

    def __init__(self, state: dict):
        super().__init__()
        if state.get("schema") != "frozen-life-sensitive-viq18-v1":
            raise ValueError("unsupported frozen life metric schema")
        if state.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("VIQ18 feature order differs")
        self.metadata = copy.deepcopy(state)
        for key in ("feature_mean", "feature_std", "coefficient", "direction"):
            value = torch.tensor(state[key], dtype=torch.float64).detach().clone()
            if value.shape != (18,) or not torch.isfinite(value).all():
                raise ValueError(f"{key} must be finite [18]")
            self.register_buffer(key, value)
        if not (self.feature_std > 0).all():
            raise ValueError("feature_std must be strictly positive")
        intercept = torch.tensor(state["intercept"], dtype=torch.float64)
        if intercept.ndim != 0 or not torch.isfinite(intercept):
            raise ValueError("intercept must be finite scalar")
        self.register_buffer("intercept", intercept.detach().clone())
        norm = torch.linalg.vector_norm(self.direction).item()
        if state["direction_degenerate"]:
            if norm != 0:
                raise ValueError("degenerate teacher must use zero direction")
        elif not math.isclose(norm, 1., rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("nondegenerate teacher direction must be unit length")

    def _features(self, raw: torch.Tensor) -> torch.Tensor:
        if any(value.dtype != torch.float64 or value.device != raw.device for value in self.buffers()):
            raise ValueError("frozen metric buffers must remain float64 on the input device")
        return torch_viq18(raw)

    def standardized_features(self, raw: torch.Tensor) -> torch.Tensor:
        return (self._features(raw) - self.feature_mean.detach()) / self.feature_std.detach()

    def predict_loglife(self, raw: torch.Tensor) -> torch.Tensor:
        """Teacher prediction in log1p(lifetime) units; not decoded RUL."""
        return self.standardized_features(raw) @ self.coefficient.detach() + self.intercept.detach()

    def discrepancy(self, raw_completion: torch.Tensor, raw_clean: torch.Tensor) -> torch.Tensor:
        """Standardized completion-minus-clean features [B,18]; target detached."""
        _raw_tensor(raw_completion, "raw_completion")
        _raw_tensor(raw_clean, "raw_clean")
        if raw_completion.shape != raw_clean.shape or raw_completion.device != raw_clean.device:
            raise ValueError("completion and clean target must have the same shape/device")
        completed = self.standardized_features(raw_completion)
        target = self.standardized_features(raw_clean.detach())
        return completed - target

    def loss(self, raw_completion: torch.Tensor, raw_clean: torch.Tensor,
             kind: str = "task") -> torch.Tensor:
        if kind not in {"uniform", "task"}:
            raise ValueError("kind must be 'uniform' or 'task'")
        delta = self.discrepancy(raw_completion, raw_clean)
        if kind == "uniform":
            return delta.square().mean()
        return (delta @ self.direction.detach()).square().mean()

    @torch.no_grad()
    def diagnostics(self, raw_completion: torch.Tensor, raw_clean: torch.Tensor) -> dict[str, torch.Tensor]:
        delta = self.discrepancy(raw_completion, raw_clean)
        completed = self.predict_loglife(raw_completion)
        clean = self.predict_loglife(raw_clean)
        return {"completed_teacher_loglife": completed, "clean_teacher_loglife": clean,
                "teacher_loglife_difference": completed - clean,
                "standardized_feature_difference": delta,
                "direction_difference": delta @ self.direction,
                "uniform_loss_per_cell": delta.square().mean(dim=-1),
                "task_loss_per_cell": (delta @ self.direction).square()}

    def to_dict(self) -> dict:
        state = copy.deepcopy(self.metadata)
        for key in ("feature_mean", "feature_std", "coefficient", "direction", "intercept"):
            state[key] = getattr(self, key).detach().cpu().tolist()
        return state
