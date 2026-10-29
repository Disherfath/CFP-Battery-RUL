"""Thin cross-cycle imputation boundary around the unchanged author T1 core.

Author: Oppenheimerdinger/T1, commit 823b76aab57a2e60eb5029caab06fdfec8b19256,
MIT License. The externally vendored source and LICENSE are hash-checked before
execution; no author trainer is imported. This is a battery task adaptation,
not an original algorithm or a reproduction of the paper's numerical results.

One input is [physical cells, 50 early cycles, 128 progress points, V/I/Q].
V/I/Q become independent batch members; cycles become the author's variables.
Progress indices are neither physical seconds nor guaranteed common SOC.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

AUTHOR_COMMIT = "823b76aab57a2e60eb5029caab06fdfec8b19256"
AUTHOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "T1"
AUTHOR_FILES = {
    "models/T1.py": "3f87aac66745029a88907b7ce65766d3ce5f785a6d4190f6343f9649d4ad3de8",
    "LICENSE": "f4d5465131eb99397d1ca666023cea25661e5b3d9e386db049af300005ad6986",
}
AUTHOR_URLS = {
    relative: f"https://github.com/Oppenheimerdinger/T1/blob/{AUTHOR_COMMIT}/{relative}"
    for relative in AUTHOR_FILES
}
MODEL_CONFIG = {
    "task_name": "imputation", "seq_len": 128, "pred_len": 128, "label_len": 0,
    "enc_in": 50, "dec_in": 50, "c_out": 50, "n_heads": 128,
    "patch_size": 2, "patch_stride": 1, "n_blocks": [2, 2],
    "kernel_size_large": [95, 41], "kernel_size_small": 5,
    "downsample_ratio": 2, "ffn_ratio": 1.0,
    "positional_encoding": True, "qkv_bias": True,
    "cosine_attention": False, "use_head_reconstruction": True,
    "head_params_shared": True, "imputation_use_mask_embedding": True,
    "drop_attn": 0.0, "drop_ffn": 0.0, "drop_proj": 0.0,
    "drop_path": 0.0, "drop_head": 0.0,
    "dropout": 0.1,  # Author run.py compatibility default; unused by this core.
    "SACA": False, "SACA_first_only": False, "drop_saca": 0.0,
    "saca_hidden_dim_ratio": 8,
    "sig_q_mode": 1, "sig_k_mode": 1, "sig_v_mode": 1,
    "mu_q_mode": 1, "mu_k_mode": 1, "mu_v_mode": 1,
    "use_SAN": False, "use_FAN": False,
}


def _verified_author(author_root: Path) -> tuple[Path, bytes, dict]:
    root = Path(author_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("author_root must be the isolated author directory")
    checked = {}
    source = None
    for relative, expected in AUTHOR_FILES.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"author source escapes its directory: {relative}")
        contents = path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        if digest != expected:
            raise RuntimeError(f"author T1 source changed: {relative}")
        checked[relative] = digest
        if relative == "models/T1.py":
            source = contents
    receipt = {
        "root": str(root), "commit": AUTHOR_COMMIT, "license": "MIT",
        "sha256": checked, "source_urls": dict(AUTHOR_URLS),
        "identity": "fixed source bytes; isolated vendor, not a git checkout",
    }
    return root, source, receipt


def author_receipt(author_root: Path = AUTHOR_ROOT) -> dict:
    """Verify the source and license bytes; the runner enforces its server path."""
    return _verified_author(author_root)[2]


def _load_author(author_root: Path):
    root, source, receipt = _verified_author(author_root)
    path = root / "models/T1.py"
    spec = importlib.util.spec_from_file_location(f"_batt_rul_author_t1_{AUTHOR_COMMIT}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct the isolated author T1 module")
    module = importlib.util.module_from_spec(spec)
    # Execute precisely the bytes checked above, not a second file read by a
    # generic import loader. Do not alter sys.path or reuse a `models` module.
    exec(compile(source, str(path), "exec"), module.__dict__)
    if not isinstance(getattr(module, "Model", None), type):
        raise RuntimeError("verified author source has no Model class")
    return module.Model, receipt


def author_class(author_root: Path = AUTHOR_ROOT):
    return _load_author(author_root)[0]


def _check_raw(raw: torch.Tensor, observed: torch.Tensor) -> None:
    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.float64:
        raise ValueError("raw must be a float64 tensor")
    if raw.ndim != 4 or tuple(raw.shape[1:]) != (50, 128, 3) or len(raw) == 0:
        raise ValueError("nonempty raw [batch,50,128,3] required")
    if not isinstance(observed, torch.Tensor) or observed.dtype != torch.bool:
        raise ValueError("boolean observed mask required")
    if observed.shape != raw.shape or observed.device != raw.device:
        raise ValueError("raw and observed must have identical shape/device")
    if not torch.isfinite(raw[observed]).all():
        raise ValueError("actually observed values must be finite")


class T1CycleAdapter(nn.Module):
    """Frozen-configuration author core; channel statistics are supplied, not fit.

    `forward` predicts every point in the externally supplied fit-channel
    standardized units, float32. `complete_raw` restores only missing positions
    to raw float64 units and retains actually observed values exactly. Neither
    method changes train/eval mode or computes a loss.
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
            "adapter": "cross-cycle author-core baseline; not paper numerical reproduction",
            "axis_contract": "[B,H=50,P=128,VIQ=3] -> [3B,P,H]; row=3*b+viq",
            "normalization": "external fit-only channel float64 stats; author per-cycle observed normalization retained",
            "mask_embedding_difference": "explicit True rather than author run.py/ETTh1 False; author mask/T unchanged",
            "kernel_rule": "floor(71*128/96)=94 -> next odd 95; floor(31*128/96)=41",
            "core_dtype": "float32", "raw_dtype": "float64", "trainer": "not imported",
        }
        # Force CPU allocation and isolate initialization from caller CPU/CUDA
        # sampling streams. The runner moves this adapter to its chosen device.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.core = cls(SimpleNamespace(**copy.deepcopy(MODEL_CONFIG))).float()

    def _inputs(self, raw: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_raw(raw, observed)
        if self.channel_mean.dtype != torch.float64 or self.channel_std.dtype != torch.float64:
            raise ValueError("do not downcast the supplied channel normalization buffers")
        if (not torch.isfinite(self.channel_mean).all()
                or not torch.isfinite(self.channel_std).all() or not (self.channel_std > 0).all()):
            raise ValueError("channel normalization state must remain finite with positive std")
        if self.channel_mean.device != raw.device or self.channel_std.device != raw.device:
            raise ValueError("move the adapter and raw input to the same device")
        parameter = next(self.core.parameters())
        if parameter.device != raw.device or parameter.dtype != torch.float32:
            raise ValueError("author core must be float32 on the raw input device")
        # Never multiply hidden NaN/Inf by zero. Clear before any arithmetic.
        safe_raw = torch.where(observed, raw, torch.zeros_like(raw))
        z = (safe_raw - self.channel_mean) / self.channel_std
        z = torch.where(observed, z, torch.zeros_like(z)).to(torch.float32)
        if not torch.isfinite(z).all():
            raise ValueError("observed standardized input overflows float32")
        batch = len(raw)
        values = z.permute(0, 3, 2, 1).reshape(batch * 3, 128, 50).contiguous()
        mask = observed.permute(0, 3, 2, 1).reshape(batch * 3, 128, 50).contiguous()
        return values, mask.to(torch.float32)

    def forward(self, raw: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        values, mask = self._inputs(raw, observed)
        # This baseline is explicitly float32, even if a caller enables AMP.
        with torch.autocast(device_type=raw.device.type, enabled=False):
            prediction = self.core(values, mask=mask)
        if prediction.shape != values.shape or prediction.dtype != torch.float32:
            raise RuntimeError("author core returned an unexpected shape/dtype")
        if prediction.device != raw.device or not torch.isfinite(prediction).all():
            raise RuntimeError("author core returned nonfinite or wrong-device predictions")
        return prediction.reshape(len(raw), 3, 128, 50).permute(0, 3, 2, 1).contiguous()

    def complete_raw(self, raw: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        prediction_z = self(raw, observed)
        completed = prediction_z.to(torch.float64) * self.channel_std + self.channel_mean
        completed = torch.where(observed, raw, completed)
        if not torch.isfinite(completed).all():
            raise RuntimeError("raw-unit completion is nonfinite")
        return completed

    def parameter_counts(self) -> dict:
        return {"allocated": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
