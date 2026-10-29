"""Leakage-safe cell datasets and input adapters for PyTorch experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from batt_rul.data.hust import HUST_FIT_IDS, HUST_TEST_IDS, HUST_VALID_IDS
from batt_rul.data.xjtu import assign_xjtu_roles
from batt_rul.missingness import apply_history_missingness, interpolate_history
from batt_rul.splits import assert_disjoint_cell_splits

InputMode = Literal["interp", "mask", "mask_delta", "interp_mask_delta"]
TrainingProfile = Literal["clean_only", "mixed"]
TargetSpace = Literal["log_lifetime", "log_rul", "rul"]
INPUT_MODES: tuple[InputMode, ...] = (
    "interp",
    "mask",
    "mask_delta",
    "interp_mask_delta",
)
TRAINING_PROFILES: tuple[TrainingProfile, ...] = ("clean_only", "mixed")


@dataclass(frozen=True)
class CellRecord:
    """One physical cell, held wholly in exactly one role."""

    cell_id: str
    role: str
    curves: NDArray[np.float32]
    rul: float
    lifetime: float
    batch: int | None = None


@dataclass(frozen=True)
class CorruptionSpec:
    """A missingness condition and its fixed evaluation replicate."""

    kind: str
    rate: float
    replicate: int = 0
    dropped_channel: int | None = None
    channel_label: str | None = None

    @property
    def key(self) -> str:
        channel = "none" if self.dropped_channel is None else str(self.dropped_channel)
        return f"{self.kind}:{self.rate:g}:{self.replicate}:channel={channel}"


@dataclass(frozen=True)
class PreparedInput:
    """Uniform-size model input plus pooling weights and mask identity."""

    features: NDArray[np.float32]
    point_weights: NDArray[np.float32]
    mask_hash: str


@dataclass(frozen=True)
class ChannelNormalizer:
    """Per-channel statistics fitted on clean training cells only."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    @classmethod
    def fit(cls, records: list[CellRecord]) -> ChannelNormalizer:
        if not records or any(record.role != "train" for record in records):
            raise ValueError("normalization must use a non-empty train-only record list")
        channel_count = records[0].curves.shape[2]
        total = np.zeros(channel_count, dtype=np.float64)
        squared = np.zeros(channel_count, dtype=np.float64)
        count = 0
        for record in records:
            values = np.asarray(record.curves, dtype=np.float64)
            if values.ndim != 3 or values.shape[2] != channel_count:
                raise ValueError("all train curves must have a common [H, P, C] shape")
            if not np.isfinite(values).all():
                raise ValueError("normalization requires finite clean curves")
            flat = values.reshape(-1, channel_count)
            total += flat.sum(axis=0)
            squared += np.square(flat).sum(axis=0)
            count += flat.shape[0]
        mean = total / count
        variance = np.maximum(squared / count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: NDArray[np.floating]) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 3 or array.shape[2] != self.mean.size:
            raise ValueError("values must have shape [H, P, C] matching the normalizer")
        return (array - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def _explicit_roles(path: Path, cell_ids: set[str]) -> dict[str, str] | None:
    frame = pd.read_csv(path, dtype={"cell_id": str})
    if "role" not in frame.columns:
        return None
    if frame["cell_id"].duplicated().any():
        raise ValueError("role manifest contains duplicate cell_id values")
    roles = dict(zip(frame["cell_id"].astype(str), frame["role"].astype(str), strict=True))
    if set(roles) != cell_ids:
        missing = sorted(cell_ids - set(roles))[:5]
        extra = sorted(set(roles) - cell_ids)[:5]
        raise ValueError(f"role manifest cell mismatch; missing={missing}, extra={extra}")
    if set(roles.values()) != {"train", "val", "test"}:
        raise ValueError("role manifest must use train, val, and test")
    return roles


def _resolve_roles(
    manifest: pd.DataFrame,
    *,
    dataset: str,
    split_manifest: Path | None,
    test_fold: int,
    split_seed: int,
) -> dict[str, str]:
    cell_ids = set(manifest["cell_id"].astype(str))
    if split_manifest is not None:
        explicit = _explicit_roles(split_manifest, cell_ids)
        if explicit is not None:
            return explicit

    if dataset == "hust":
        roles = {}
        for cell_id in cell_ids:
            if cell_id in HUST_FIT_IDS:
                roles[cell_id] = "train"
            elif cell_id in HUST_VALID_IDS:
                roles[cell_id] = "val"
            elif cell_id in HUST_TEST_IDS:
                roles[cell_id] = "test"
            else:
                raise ValueError(f"unknown HUST split for {cell_id}")
        return roles

    if split_manifest is None:
        raise ValueError("XJTU requires a fold manifest or an explicit role manifest")
    split_frame = pd.read_csv(split_manifest, dtype={"cell_id": str})
    required = {"cell_id", "batch", "test_fold"}
    if not required.issubset(split_frame.columns):
        raise ValueError(f"XJTU fold manifest requires columns {sorted(required)}")
    split_ids = set(split_frame["cell_id"].astype(str))
    if split_ids != cell_ids:
        raise ValueError("XJTU fold manifest and compact cache contain different cells")
    return assign_xjtu_roles(
        split_frame.to_dict("records"), test_fold=test_fold, seed=split_seed
    )


def load_cell_records(
    cache_root: Path,
    *,
    dataset: str,
    history: int,
    split_manifest: Path | None = None,
    test_fold: int = 0,
    split_seed: int = 20260815,
    include_roles: tuple[str, ...] | None = None,
) -> list[CellRecord]:
    """Load compact caches and assign roles without deriving any split from cycles.

    ``include_roles`` is a data-access barrier, not a post-load filter. Rows assigned to
    excluded roles are skipped before their cache files are opened. Existing formal
    callers omit it and retain the original train/validation/test behaviour.
    """

    normalized_dataset = dataset.lower()
    if normalized_dataset not in {"hust", "xjtu"}:
        raise ValueError("dataset must be hust or xjtu")
    manifest_path = cache_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"cell_id": str})
    required = {"cell_id", "cache_file", f"rul_h{history}"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"cache manifest requires columns {sorted(required)}")
    if manifest["cell_id"].duplicated().any():
        raise ValueError("cache manifest contains duplicate physical cells")
    roles = _resolve_roles(
        manifest,
        dataset=normalized_dataset,
        split_manifest=split_manifest,
        test_fold=test_fold,
        split_seed=split_seed,
    )
    if include_roles is None:
        requested_roles = ("train", "val", "test")
    else:
        requested_roles = tuple(str(role) for role in include_roles)
        if not requested_roles or len(requested_roles) != len(set(requested_roles)):
            raise ValueError("include_roles must contain unique split roles")
        if not set(requested_roles).issubset({"train", "val", "test"}):
            raise ValueError("include_roles may contain only train, val, and test")
    records: list[CellRecord] = []
    expected_shape: tuple[int, int] | None = None
    for row in manifest.to_dict("records"):
        cell_id = str(row["cell_id"])
        if roles[cell_id] not in requested_roles:
            continue
        with np.load(cache_root / str(row["cache_file"]), allow_pickle=False) as cache:
            curves = np.asarray(cache["curves"][:history], dtype=np.float32)
        if curves.ndim != 3 or curves.shape[0] != history:
            raise ValueError(f"{cell_id} does not contain H={history} curves")
        shape = (curves.shape[1], curves.shape[2])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all compact caches must share point and channel counts")
        if not np.isfinite(curves).all():
            raise ValueError(f"{cell_id} clean cache contains non-finite values")
        rul = float(row[f"rul_h{history}"])
        if not np.isfinite(rul) or rul < 0:
            raise ValueError(f"{cell_id} has an invalid RUL target")
        batch_value = row.get("batch")
        batch = None if batch_value is None or pd.isna(batch_value) else int(batch_value)
        records.append(
            CellRecord(
                cell_id=cell_id,
                role=roles[cell_id],
                curves=curves,
                rul=rul,
                lifetime=rul + history,
                batch=batch,
            )
        )
    split_groups = {
        role: [record.cell_id for record in records if record.role == role]
        for role in requested_roles
    }
    if any(not cells for cells in split_groups.values()):
        raise ValueError("each requested role must contain at least one cell")
    assert_disjoint_cell_splits(split_groups)
    return records


def encode_target(rul: float, *, history: int, target_space: TargetSpace) -> float:
    if target_space == "log_lifetime":
        return float(np.log1p(rul + history))
    if target_space == "log_rul":
        return float(np.log1p(rul))
    if target_space == "rul":
        return float(rul)
    raise ValueError(f"unknown target space: {target_space}")


def decode_target(
    prediction: NDArray[np.floating], *, history: int, target_space: TargetSpace
) -> NDArray[np.float64]:
    raw = np.asarray(prediction, dtype=np.float64)
    if target_space == "log_lifetime":
        result = np.expm1(np.clip(raw, -20.0, 20.0)) - history
    elif target_space == "log_rul":
        result = np.expm1(np.clip(raw, -20.0, 20.0))
    elif target_space == "rul":
        result = raw
    else:
        raise ValueError(f"unknown target space: {target_space}")
    return np.maximum(result, 0.0)


def prepare_input(
    record: CellRecord,
    *,
    normalizer: ChannelNormalizer,
    mode: InputMode,
    condition: CorruptionSpec,
    base_seed: int,
    sensor_channels: tuple[int, ...],
    time_channel: int,
) -> PreparedInput:
    """Create equal-width slots: value, mask, delta; unused slots stay exactly zero."""

    if mode not in INPUT_MODES:
        raise ValueError(f"unknown input mode: {mode}")
    corrupted = apply_history_missingness(
        record.curves,
        condition.kind,
        condition.rate,
        base_seed=base_seed,
        cell_id=record.cell_id,
        replicate=condition.replicate,
        sensor_channels=sensor_channels,
        time_channel=time_channel,
        dropped_channel=condition.dropped_channel,
    )
    sensor_observed = corrupted.observed[:, :, sensor_channels]
    if mode in {"interp", "interp_mask_delta"}:
        filled = interpolate_history(
            corrupted,
            time_channel=time_channel,
            fully_missing_fill=normalizer.mean,
        )
        values = normalizer.transform(filled)
        mask_slots = np.zeros(sensor_observed.shape, dtype=np.float64)
        delta_slots = np.zeros(sensor_observed.shape, dtype=np.float64)
        point_weights = np.ones(sensor_observed.shape[:2], dtype=np.float64)
        if mode == "interp_mask_delta":
            duration = np.ptp(record.curves[:, :, time_channel], axis=1)
            duration = np.maximum(duration, 1e-8)[:, None, None]
            mask_slots = sensor_observed.astype(np.float64)
            delta_slots = corrupted.delta_t[:, :, sensor_channels] / duration
            point_weights = sensor_observed.any(axis=2).astype(np.float64)
    else:
        normalized = normalizer.transform(corrupted.values)
        values = np.where(corrupted.observed, normalized, 0.0)
        mask_slots = sensor_observed.astype(np.float64)
        delta_slots = np.zeros(sensor_observed.shape, dtype=np.float64)
        point_weights = np.ones(sensor_observed.shape[:2], dtype=np.float64)
        if mode == "mask_delta":
            duration = np.ptp(record.curves[:, :, time_channel], axis=1)
            duration = np.maximum(duration, 1e-8)[:, None, None]
            delta_slots = corrupted.delta_t[:, :, sensor_channels] / duration
            point_weights = sensor_observed.any(axis=2).astype(np.float64)
    features = np.concatenate([values, mask_slots, delta_slots], axis=2)
    if not np.isfinite(features).all() or not np.isfinite(point_weights).all():
        raise ValueError("prepared model input contains non-finite values")
    mask_hash = hashlib.sha256(sensor_observed.tobytes()).hexdigest()[:16]
    return PreparedInput(
        features=features.astype(np.float32),
        point_weights=point_weights.astype(np.float32),
        mask_hash=mask_hash,
    )


class CellCurveDataset(Dataset[dict[str, Any]]):
    """Expand physical cells over corruption conditions without changing their role."""

    def __init__(
        self,
        records: list[CellRecord],
        *,
        normalizer: ChannelNormalizer,
        mode: InputMode,
        conditions: list[CorruptionSpec],
        history: int,
        target_space: TargetSpace,
        base_seed: int,
        sensor_channels: tuple[int, ...] = (0, 1, 2),
        time_channel: int = 3,
        resample_each_epoch: bool = False,
    ) -> None:
        if not records or not conditions:
            raise ValueError("dataset requires records and corruption conditions")
        if len({record.role for record in records}) != 1:
            raise ValueError("one dataset instance may contain only one split role")
        self.records = records
        self.normalizer = normalizer
        self.mode = mode
        self.conditions = conditions
        self.history = history
        self.target_space = target_space
        self.base_seed = base_seed
        self.sensor_channels = sensor_channels
        self.time_channel = time_channel
        self.resample_each_epoch = resample_each_epoch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records) * len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, condition_index = divmod(index, len(self.conditions))
        record = self.records[record_index]
        condition = self.conditions[condition_index]
        if self.resample_each_epoch and condition.kind != "clean":
            condition = CorruptionSpec(
                condition.kind,
                condition.rate,
                self.epoch,
                condition.dropped_channel,
                condition.channel_label,
            )
        prepared = prepare_input(
            record,
            normalizer=self.normalizer,
            mode=self.mode,
            condition=condition,
            base_seed=self.base_seed,
            sensor_channels=self.sensor_channels,
            time_channel=self.time_channel,
        )
        target = encode_target(
            record.rul, history=self.history, target_space=self.target_space
        )
        return {
            "features": torch.from_numpy(prepared.features),
            "point_weights": torch.from_numpy(prepared.point_weights),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_rul": torch.tensor(record.rul, dtype=torch.float32),
            "cell_id": record.cell_id,
            "kind": condition.kind,
            "rate": float(condition.rate),
            "replicate": int(condition.replicate),
            "dropped_channel": (
                -1 if condition.dropped_channel is None else int(condition.dropped_channel)
            ),
            "channel_label": condition.channel_label or "none",
            "mask_hash": prepared.mask_hash,
            "sample_key": f"{record.cell_id}|{condition.key}",
        }


def mixed_training_conditions(kinds: list[str], rates: list[float]) -> list[CorruptionSpec]:
    """Return one clean sample plus each configured corrupt kind/rate pair."""

    conditions = [CorruptionSpec("clean", 0.0)]
    conditions.extend(
        CorruptionSpec(kind.lower(), float(rate))
        for kind in kinds
        if kind.lower() != "clean"
        for rate in rates
    )
    return conditions


def training_conditions_for_profile(
    profile: TrainingProfile,
    *,
    mixed_kinds: list[str],
    mixed_rates: list[float],
) -> list[CorruptionSpec]:
    """Return the frozen clean-only or clean-plus-corruption training protocol."""

    if profile == "clean_only":
        return [CorruptionSpec("clean", 0.0)]
    if profile == "mixed":
        return mixed_training_conditions(mixed_kinds, mixed_rates)
    raise ValueError(f"unknown training profile: {profile}")


def fixed_evaluation_conditions(
    kinds: list[str],
    rates: list[float],
    replicates: list[int],
    *,
    channel_dropouts: list[tuple[str, int]] | None = None,
    sensor_channels: tuple[int, ...] = (0, 1, 2),
) -> list[CorruptionSpec]:
    """Create condition order shared verbatim by validation and test datasets."""

    if not replicates:
        raise ValueError("evaluation requires at least one fixed corruption replicate")
    normalized_kinds = [kind.lower() for kind in kinds]
    if len(normalized_kinds) != len(set(normalized_kinds)):
        raise ValueError("evaluation kinds must be unique")
    conditions = [CorruptionSpec("clean", 0.0, 0)]
    conditions.extend(
        CorruptionSpec(kind.lower(), float(rate), int(replicate))
        for kind in normalized_kinds
        if kind not in {"clean", "channel"}
        for rate in rates
        for replicate in replicates
    )
    if "channel" in normalized_kinds:
        if not channel_dropouts:
            raise ValueError("channel evaluation requires explicit channel_dropouts")
        labels = [str(label) for label, _ in channel_dropouts]
        dropped = [int(channel) for _, channel in channel_dropouts]
        if len(labels) != len(set(labels)) or any(not label for label in labels):
            raise ValueError("channel dropout labels must be non-empty and unique")
        if len(dropped) != len(set(dropped)) or set(dropped) != set(sensor_channels):
            raise ValueError("channel_dropouts must name every sensor channel exactly once")
        channel_rate = 1.0 / len(sensor_channels)
        conditions.extend(
            CorruptionSpec(
                "channel",
                channel_rate,
                int(replicate),
                dropped_channel=channel,
                channel_label=label,
            )
            for label, channel in channel_dropouts
            for replicate in replicates
        )
    return conditions
