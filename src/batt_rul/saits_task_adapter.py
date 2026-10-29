"""Task boundary around the unmodified, externally checked-out author SAITS.

Each sample is one cached battery cycle [128, VIQ], not physical seconds or SOC.
The author core/losses are imported unchanged; no author driver is imported.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

AUTHOR_COMMIT = "660b87f19c1277065f314f24134f646229e89ca9"
AUTHOR_FILES = {
    "modeling/saits.py": "7c52cbcff6d2a223f1e788efabf8d46ebc84f79d9d480d59ae76619b7d31a556",
    "modeling/layers.py": "77bed4cee066db537aa6c738d7030580b561772678af4564cbf2ab77392b78b9",
    "modeling/utils.py": "bca5e2704297dc9a1d7c2ad097511b70e0b5daf6ec5b3c65f379c0316d81981f",
    "LICENSE": "8921a1d45f975221acec71bfd4959549a710c561600c5723274a55d03f059283",
}
MODEL_CONFIG = {
    "n_groups": 1, "n_group_inner_layers": 1, "d_time": 128, "d_feature": 3,
    "d_model": 64, "d_inner": 128, "n_head": 4, "d_k": 16, "d_v": 16,
    "dropout": 0., "input_with_mask": True, "param_sharing_strategy": "inner_group",
    "MIT": True, "diagonal_attention_mask": True,
}


def author_receipt(repo: Path) -> dict:
    repo = Path(repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != AUTHOR_COMMIT:
        raise RuntimeError("author SAITS commit changed")
    hashes = {}
    for relative, expected in AUTHOR_FILES.items():
        digest = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"author SAITS source changed: {relative}")
        hashes[relative] = digest
    return {"repo": str(repo), "commit": commit, "license": "MIT", "sha256": hashes}


def author_class(repo: Path):
    receipt = author_receipt(repo)
    root = Path(receipt["repo"])
    # Author absolute imports use the generic `modeling` namespace. Never reuse
    # a module loaded from a different checkout silently.
    for name, module in tuple(sys.modules.items()):
        if name == "modeling" or name.startswith("modeling."):
            paths = ([module.__file__] if getattr(module, "__file__", None)
                     else list(getattr(module, "__path__", [])))
            if any(not Path(p).resolve().is_relative_to(root) for p in paths):
                raise RuntimeError("another modeling namespace is already loaded")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("modeling.saits").SAITS


def cycle_inputs(visible_z: torch.Tensor, observed: torch.Tensor,
                 target_z: torch.Tensor | None = None, *, stage: str = "test") -> dict:
    """Mask *before* the core; hidden finite values never become observations."""
    if stage not in ("train", "val", "test"):
        raise ValueError("stage must be train, val, or test")
    if visible_z.ndim != 3 or tuple(visible_z.shape[1:]) != (128, 3) or len(visible_z) == 0:
        raise ValueError("expected nonempty [batch,128,3] independent cycles")
    if not visible_z.is_floating_point() or not torch.isfinite(visible_z).all():
        raise ValueError("finite floating-point cycle values required")
    if observed.dtype != torch.bool or observed.shape != visible_z.shape:
        raise ValueError("same-shape boolean observed mask required")
    if observed.device != visible_z.device:
        raise ValueError("mask and values must share device")
    inputs = {"X": torch.where(observed, visible_z, torch.zeros_like(visible_z)),
              "missing_mask": observed.to(visible_z.dtype)}
    if stage != "test":
        if target_z is None or target_z.shape != visible_z.shape:
            raise ValueError("train/val require aligned clean artificial-mask targets")
        if target_z.dtype != visible_z.dtype or target_z.device != visible_z.device:
            raise ValueError("target dtype/device must match values")
        if not torch.isfinite(target_z).all():
            raise ValueError("finite artificial-mask truth required")
        per_sample = observed.reshape(len(observed), -1).sum(1)
        if (per_sample == 0).any() or (per_sample == observed[0].numel()).any():
            raise ValueError("each training/validation cycle needs ORT and MIT positions")
        inputs.update(X_holdout=target_z, indicating_mask=(~observed).to(visible_z.dtype))
    return inputs


class AuthorSAITSAdapter(nn.Module):
    def __init__(self, author_repo: Path, seed: int = 20261005):
        super().__init__()
        cls = author_class(author_repo)
        self.receipt = author_receipt(author_repo)
        # Initialization reproducible without perturbing a caller's sampling RNG.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.core = cls(**MODEL_CONFIG, device="cpu")

    def forward(self, visible_z, observed, target_z=None, *, stage="test"):
        inputs = cycle_inputs(visible_z, observed, target_z, stage=stage)
        if next(self.core.parameters()).device != visible_z.device:
            raise ValueError("move model and input to the same device")
        # Author layers cache this non-buffer attribute for attention eye masks.
        for module in self.core.modules():
            if hasattr(module, "device"):
                module.device = visible_z.device
        result = self.core(inputs, stage)
        if stage != "test":
            result["loss"] = result["reconstruction_loss"] + result["imputation_loss"]
        return result


def restore_observed(completed_z, visible_raw, observed, mean, std):
    """Inverse normalization only on hidden positions; preserve cached observations."""
    if completed_z.shape != visible_raw.shape or observed.shape != visible_raw.shape:
        raise ValueError("completion, raw values, and mask shapes must match")
    if observed.dtype != torch.bool:
        raise ValueError("boolean mask required")
    return torch.where(observed, visible_raw, completed_z.to(visible_raw.dtype) * std + mean)
