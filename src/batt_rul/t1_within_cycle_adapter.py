"""Native-variable T1 adapter: V/I/Q interact within each discharge cycle.

The hash-verified author Model is unchanged. Each 128-point cycle is one batch
member, and the three measured channels are its variables. This is a battery
task adaptation, not a reproduction of the paper's numerical results.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .t1_cycle_adapter import MODEL_CONFIG as CROSS_CYCLE_CONFIG, _check_raw, _load_author


MODEL_CONFIG = copy.deepcopy(CROSS_CYCLE_CONFIG)
MODEL_CONFIG.update(enc_in=3, dec_in=3, c_out=3)
WITHIN_CYCLE_CONFIG = MODEL_CONFIG


class T1WithinCycleAdapter(nn.Module):
    """Map [B,50,128,3] to [50B,128,3], preserving original channel identity.

    ``forward`` returns float32 predictions in external fit-channel-standardized
    units with the original four-axis shape. ``complete_raw`` converts to raw
    float64 units and restores genuinely observed values exactly. External
    statistics are supplied, never fitted or updated by the adapter.

    Width, stages, kernels, dropout, and reconstruction head match the previous
    frozen T1 configuration; only the number and meaning of variables changes.
    Attention heads are convolutional feature channels, not partitions of the
    three raw variables, so n_heads=128 is valid for three variables. The learned
    positional tensor is smaller, and old cross-cycle checkpoints are therefore
    neither shape-compatible nor parameter-matched.
    """

    def __init__(self, author_root: Path, channel_mean: torch.Tensor,
                 channel_std: torch.Tensor, seed: int = 20261008):
        super().__init__()
        for name, value in (("channel_mean", channel_mean), ("channel_std", channel_std)):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise ValueError(f"{name} must be an externally fitted float64 tensor")
            if value.shape != (3,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape [3]")
        if not (channel_std > 0).all():
            raise ValueError("channel_std must be strictly positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        self.register_buffer("channel_mean", channel_mean.detach().cpu().clone())
        self.register_buffer("channel_std", channel_std.detach().cpu().clone())
        cls, source_receipt = _load_author(author_root)
        self.receipt = {
            **source_receipt, "model_config": copy.deepcopy(MODEL_CONFIG), "seed": seed,
            "adapter": "within-cycle VIQ author-core T1; not paper numerical reproduction",
            "axis_contract": "[B,H=50,P=128,VIQ=3] -> [50B,P,VIQ]; row=50*b+h",
            "variables": "V,I,Q, in this order; no inter-cycle attention",
            "normalization": "external fit-only float64 channel statistics plus unchanged author observed per-variable normalization",
            "mask_embedding_difference": "explicit True rather than author run.py/ETTh1 False; unchanged author mask/T scaling",
            "kernel_rule": "floor(71*128/96)=94 -> next odd95; floor(31*128/96)=41",
            "cross_cycle_config_difference": {"enc_in": [50, 3], "dec_in": [50, 3], "c_out": [50, 3]},
            "weight_compatibility": "new initialization required; positional shape differs from cross-cycle T1",
            "core_dtype": "float32", "raw_dtype": "float64", "trainer": "not imported",
        }
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.core = cls(SimpleNamespace(**copy.deepcopy(MODEL_CONFIG))).float()

    def _inputs(self, raw: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_raw(raw, observed)
        if self.channel_mean.dtype != torch.float64 or self.channel_std.dtype != torch.float64:
            raise ValueError("do not downcast the supplied channel normalization buffers")
        if (not torch.isfinite(self.channel_mean).all() or not torch.isfinite(self.channel_std).all()
                or not (self.channel_std > 0).all()):
            raise ValueError("normalization buffers must remain finite with positive std")
        if self.channel_mean.device != raw.device or self.channel_std.device != raw.device:
            raise ValueError("move the adapter and raw input to the same device")
        parameter = next(self.core.parameters())
        if parameter.device != raw.device or parameter.dtype != torch.float32:
            raise ValueError("author core must be float32 on the raw input device")
        safe = torch.where(observed, raw, torch.zeros_like(raw))
        values = (safe - self.channel_mean) / self.channel_std
        values = torch.where(observed, values, torch.zeros_like(values)).float()
        if not torch.isfinite(values).all():
            raise ValueError("observed standardized input overflows float32")
        # No permutation: V/I/Q remain columns, and each physical cell's 50
        # cycles are consecutive independent batch rows.
        return (values.reshape(len(raw) * 50, 128, 3).contiguous(),
                observed.reshape(len(raw) * 50, 128, 3).float().contiguous())

    def forward(self, raw: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        values, mask = self._inputs(raw, observed)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            prediction = self.core(values, mask=mask)
        if prediction.shape != values.shape or prediction.dtype != torch.float32:
            raise RuntimeError("author core returned an unexpected shape/dtype")
        if prediction.device != raw.device or not torch.isfinite(prediction).all():
            raise RuntimeError("author core returned nonfinite or wrong-device predictions")
        return prediction.reshape(len(raw), 50, 128, 3).contiguous()

    def complete_raw(self, raw: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        prediction = self(raw, observed).double() * self.channel_std + self.channel_mean
        completed = torch.where(observed, raw, prediction)
        if not torch.isfinite(completed).all():
            raise RuntimeError("raw-unit completion is nonfinite")
        return completed

    def parameter_counts(self) -> dict:
        return {"allocated": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
