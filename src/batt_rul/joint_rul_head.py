"""Trainable VIQ18-to-lifetime head for an end-to-end supervised comparator.

Unlike the frozen Direction teacher, this head is optimized jointly with the
imputer. This module has no data loader or split discovery.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn

from .life_sensitive_features import torch_viq18, _fit_tensor, _array_hash, _raw_tensor
from .reference_data import FEATURE_NAMES


@torch.no_grad()
def fit_joint_reference(clean_fit, li_fit_bank, lifetime, fit_ids):
    """Fit only supplied training cells; LI weights match the CFP reference."""
    clean = _fit_tensor(clean_fit, "clean_fit")
    _raw_tensor(clean)
    ids = list(fit_ids)
    if len(ids) != len(clean) or len(ids) < 2 or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("unique nonempty training cell IDs required")
    bank = _fit_tensor(li_fit_bank, "li_fit_bank")
    if bank.shape != (len(clean), 10, 50, 128, 3):
        raise ValueError("canonical ten-view LI fit bank required")
    if not torch.equal(bank[:, 0], clean):
        raise ValueError("LI clean view differs")
    y = _fit_tensor(lifetime, "lifetime")
    if y.shape != (len(clean),) or (y <= 50).any():
        raise ValueError("aligned lifetime endpoints above H50 required")
    f = torch_viq18(bank.reshape(-1, 50, 128, 3)).reshape(len(clean), 10, 18)
    vw = torch.tensor([.25] + [1. / 12.] * 9, dtype=torch.float64)
    w = vw[None].expand(len(clean), -1).reshape(-1).clone()
    w /= w.sum()
    flat = f.reshape(-1, 18)
    mean = (flat * w[:, None]).sum(0)
    raw_std = ((flat - mean).square() * w[:, None]).sum(0).sqrt()
    std = torch.where(raw_std < 1e-8, torch.ones_like(raw_std), raw_std)
    target = torch.log1p(y)
    target_mean, target_std = target.mean(), target.std(unbiased=False)
    if target_std < 1e-8:
        target_std = torch.ones_like(target_std)
    return {"schema": "joint-rul-viq18-reference-v1", "fit_ids": ids,
        "feature_names": list(FEATURE_NAMES), "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "target_mean": float(target_mean), "target_std": float(target_std),
        "target": "log1p(lifetime endpoint N); population standardization on training cells",
        "view_weights": vw.tolist(), "feature_scale_source": "training LI ten-view bank",
        "scale_sha256": _array_hash(torch.stack((mean, std))),
        "target_sha256": _array_hash(target), "li_feature_bank_sha256": _array_hash(f),
        "hidden_width": 32, "activation": "GELU", "head_parameters": 641}


class JointRULHead(nn.Module):
    def __init__(self, reference, seed=20261018):
        super().__init__()
        if reference.get("schema") != "joint-rul-viq18-reference-v1" or reference["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("unsupported joint-head reference")
        self.metadata = copy.deepcopy(reference)
        for name in ("feature_mean", "feature_std"):
            value = torch.tensor(reference[name], dtype=torch.float64)
            if value.shape != (18,) or not torch.isfinite(value).all():
                raise ValueError("finite VIQ18 reference required")
            self.register_buffer(name, value)
        if not (self.feature_std > 0).all():
            raise ValueError("positive feature scales required")
        for name in ("target_mean", "target_std"):
            value = float(reference[name])
            if not math.isfinite(value) or (name == "target_std" and value <= 0):
                raise ValueError("finite target reference and positive scale required")
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        # CPU initialization has its own RNG, leaving SAITS/sampling RNG intact.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(18, 32, dtype=torch.float64), nn.GELU(),
                                         nn.Linear(32, 1, dtype=torch.float64))

    def forward(self, completed_raw):
        standardized = (torch_viq18(completed_raw) - self.feature_mean) / self.feature_std
        return self.network(standardized).squeeze(-1)

    def standardized_target(self, lifetime):
        if lifetime.ndim != 1 or not torch.isfinite(lifetime).all() or (lifetime <= 50).any():
            raise ValueError("finite lifetime endpoints [B] above H50 required")
        return (torch.log1p(lifetime) - self.target_mean) / self.target_std

    def predict_loglife(self, completed_raw):
        return self(completed_raw) * self.target_std + self.target_mean
