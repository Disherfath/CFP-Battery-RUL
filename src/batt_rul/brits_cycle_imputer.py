"""BRITS author-core battery adaptation, not a numerical paper reproduction.

Derived from caow13/BRITS commit fc0a3a472a6d99a6934e471d40c25ed0b029b501.
MIT License, Copyright (c) 2018 Dawn90.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Mechanical changes: feature/sequence dimensions and device-safe zero states.
New battery input adapter uses previous-mask deltas, not author preprocessing.
Author bare Linear fusion, estimate MAE accumulation and consistency are kept.
The original allocated classifier is frozen and not evaluated; no MIT/RUL loss.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

AUTHOR_COMMIT = "fc0a3a472a6d99a6934e471d40c25ed0b029b501"
AUTHOR_FILES = {
    "models/rits.py": "d625f20b3e259261616b3c7b0da599cac938ff3e05b6e014053e66aafc084515",
    "models/brits.py": "a94373ff64b08dc69ea742867dcdb2ad059565668db24c8a5186d9cfd3e7babe",
    "input_process.py": "59e7e4b4947e470efab345dcb4aac2f0c79da4a860793b2f48d44fe9a02b33b8",
    "LICENSE": "354dd922300104666b04eb92d54bc38af0c71a27265db9caf80b047722f82355",
}
MODEL_CONFIG = {"features": 3, "sequence_length": 128, "hidden_size": 64,
    "impute_weight": 3., "label_weight": 0., "consistency_weight": .1,
    "fusion": "author bare Linear; no sigmoid", "MIT": False,
    "delta": "previous observed mask; first point zero; unit progress-index gap"}


def author_receipt(repo: Path) -> dict:
    repo = Path(repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != AUTHOR_COMMIT:
        raise RuntimeError("author BRITS commit changed")
    hashes = {}
    for name, expected in AUTHOR_FILES.items():
        digest = hashlib.sha256((repo / name).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"author BRITS file changed: {name}")
        hashes[name] = digest
    return {"repo": str(repo), "commit": commit, "license": "MIT", "sha256": hashes}


def directional_delta(observed: torch.Tensor) -> torch.Tensor:
    """Recompute separately after reversing mask; never reverse forward deltas."""
    if observed.ndim != 3 or observed.dtype != torch.bool or min(observed.shape) < 1:
        raise ValueError("nonempty [batch,time,feature] boolean mask required")
    delta = torch.zeros(observed.shape, dtype=torch.float32, device=observed.device)
    for t in range(1, observed.shape[1]):
        delta[:, t] = 1. + (~observed[:, t - 1]).float() * delta[:, t - 1]
    return delta


def cycle_inputs(visible_z: torch.Tensor, observed: torch.Tensor, *, stage="test") -> dict:
    if stage not in ("train", "val", "test"):
        raise ValueError("unknown stage")
    if visible_z.ndim != 3 or tuple(visible_z.shape[1:]) != (128, 3) or len(visible_z) == 0:
        raise ValueError("nonempty [batch,128,3] independent cycles required")
    if not visible_z.is_floating_point() or observed.dtype != torch.bool or observed.shape != visible_z.shape:
        raise ValueError("floating values and same-shape boolean mask required")
    if observed.device != visible_z.device or not torch.isfinite(visible_z[observed]).all():
        raise ValueError("same-device masks and finite actually observed values required")
    if stage == "train":
        counts = observed.flatten(1).sum(1)
        if (counts == 0).any() or (counts == observed[0].numel()).any():
            raise ValueError("each training cycle needs visible and artificial missing points")
    safe = torch.where(observed, visible_z, torch.zeros_like(visible_z))
    mask_f = observed.to(visible_z.dtype)
    backward_mask = observed.flip(1)
    return {"forward": {"values": safe, "masks": mask_f,
                "deltas": directional_delta(observed).to(visible_z.dtype)},
            "backward": {"values": safe.flip(1), "masks": mask_f.flip(1),
                "deltas": directional_delta(backward_mask).to(visible_z.dtype)}}


class FeatureRegression(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(features, features))
        self.b = nn.Parameter(torch.Tensor(features))
        self.register_buffer("m", torch.ones(features, features) - torch.eye(features))
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(x, self.W * self.m, self.b)


class TemporalDecay(nn.Module):
    def __init__(self, features, output, diag=False):
        super().__init__()
        self.diag = diag
        self.W = nn.Parameter(torch.Tensor(output, features))
        self.b = nn.Parameter(torch.Tensor(output))
        if diag:
            if features != output:
                raise ValueError("diagonal decay requires identical dimensions")
            self.register_buffer("m", torch.eye(features))
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, delta):
        weight = self.W * self.m if self.diag else self.W
        return torch.exp(-F.relu(F.linear(delta, weight, self.b)))


class RITS(nn.Module):
    """Same author allocation order, recurrence and accumulated ORT terms."""
    def __init__(self, features, hidden_size):
        super().__init__()
        self.rnn_cell = nn.LSTMCell(features * 2, hidden_size)
        self.temp_decay_h = TemporalDecay(features, hidden_size)
        self.temp_decay_x = TemporalDecay(features, features, diag=True)
        self.hist_reg = nn.Linear(hidden_size, features)
        self.feat_reg = FeatureRegression(features)
        self.weight_combine = nn.Linear(features * 2, features)
        self.dropout = nn.Dropout(.25)  # Allocated but unused by author forward.
        self.out = nn.Linear(hidden_size, 1)  # Preserve init order, not a RUL head.
        self.out.requires_grad_(False)

    def forward(self, data, *, details=False):
        values, masks, deltas = (data[name] for name in ("values", "masks", "deltas"))
        h = values.new_zeros((len(values), self.rnn_cell.hidden_size))
        c = torch.zeros_like(h)
        x_loss = 0.0
        imputations, estimates = [], []
        for t in range(values.shape[1]):
            x, m, d = values[:, t], masks[:, t], deltas[:, t]
            gamma_h, gamma_x = self.temp_decay_h(d), self.temp_decay_x(d)
            h = h * gamma_h
            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)
            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)
            c_c = m * x + (1 - m) * c_h
            h, c = self.rnn_cell(torch.cat([c_c, m], dim=1), (h, c))
            imputations.append(c_c.unsqueeze(1))
            if details:
                estimates.append(torch.stack((x_h, z_h, c_h), dim=1))
        result = {"imputations": torch.cat(imputations, dim=1), "ort": x_loss}
        if details:
            result["estimates"] = torch.stack(estimates, dim=1)
        return result


class BidirectionalRITS(nn.Module):
    def __init__(self, features=3, hidden_size=64):
        super().__init__()
        self.rits_f = RITS(features, hidden_size)
        self.rits_b = RITS(features, hidden_size)

    def forward(self, inputs, *, details=False):
        first = self.rits_f(inputs["forward"], details=details)
        backward = self.rits_b(inputs["backward"], details=details)
        reverse = backward["imputations"].flip(1)
        consistency = torch.abs(first["imputations"] - reverse).mean() * .1
        ort = first["ort"] + backward["ort"]
        # Keep the author's operation order: 3*ORT_f + 3*ORT_b + consistency.
        loss = first["ort"] * 3. + backward["ort"] * 3. + consistency
        result = {"imputed_data": (first["imputations"] + reverse) / 2,
                  "loss": loss, "reconstruction_loss": ort, "consistency_loss": consistency,
                  "forward_ort": first["ort"], "backward_ort": backward["ort"]}
        if details:
            result.update(forward_details=first, backward_details=backward)
        return result


class BRITSCycleImputer(nn.Module):
    def __init__(self, author_repo: Path, seed=20261005):
        super().__init__()
        self.receipt = author_receipt(author_repo)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.core = BidirectionalRITS(3, 64)

    def forward(self, visible_z, observed, target_z=None, *, stage="test", details=False):
        # target_z is ignored for compatibility with the old completion helper.
        # No hidden clean target or lifetime is put in the training graph.
        inputs = cycle_inputs(visible_z, observed, stage=stage)
        parameter = next(self.core.parameters())
        if parameter.device != visible_z.device or parameter.dtype != visible_z.dtype:
            raise ValueError("model and inputs must have identical dtype/device")
        return self.core(inputs, details=details)

    def parameter_counts(self):
        return {"allocated": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "frozen_classification": sum(p.numel() for p in self.parameters() if not p.requires_grad)}
