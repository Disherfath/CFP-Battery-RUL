"""Clean-first reference-learning data helpers; no test-curve access interface.

The existing dataset adapters define lifetime labels. HUST uses effective recorded
cycle count; XJTU uses author-provided cycle_life. Neither is re-estimated here.
Run preprocessing and numerical experiments on the research server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from batt_rul.data.hust import HUST_FIT_IDS, HUST_TEST_IDS, HUST_VALID_IDS
from batt_rul.torch_data import CellRecord, load_cell_records

STATS = ("mean", "std", "min", "max", "last5_minus_first5", "cycle_mean_slope_t01")
FEATURE_NAMES = tuple(f"{channel}_{stat}" for channel in ("V", "I", "Q") for stat in STATS)


def _validate_viq(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 10 or array.shape[1] != 128 or array.shape[2] != 3:
        raise ValueError("expected [H>=10, 128, 3] VIQ curves")
    if not np.isfinite(array).all():
        raise ValueError("VIQ curves must be finite; imputation must be explicit before this API")
    return array


def clean_viq(record: CellRecord) -> np.ndarray:
    """Copy VIQ only; time/temperature, IDs and labels are not model features."""
    return _validate_viq(record.curves[..., :3]).copy()


def load_reference_records(
    cache: Path,
    *,
    dataset: str,
    role: str = "train",
    split_manifest: Path | None = None,
    history: int = 50,
    test_fold: int = 0,
    split_seed: int = 20260815,
) -> list[CellRecord]:
    """Open one explicit non-test role only, before loading any curve arrays.

    XJTU requires the original fold/role manifest. HUST defaults to the existing
    47/8/22 role file and rejects changes to its published ID sets. Loading val is
    a separate explicit call, so training-only callers never open val curves.
    """
    if role not in ("train", "val"):
        raise ValueError("only train or explicitly requested val curves may be loaded")
    if dataset not in ("hust", "xjtu"):
        raise ValueError("dataset must be hust or xjtu")
    if history not in (20, 50):
        raise ValueError("cached task history must be 20 or 50")
    cache = Path(cache)
    if dataset == "hust":
        split_manifest = Path(split_manifest) if split_manifest else cache / "roles_47_8_22.csv"
        roles = pd.read_csv(split_manifest, dtype={"cell_id": str})
        if not {"cell_id", "role"}.issubset(roles.columns) or roles.cell_id.duplicated().any():
            raise ValueError("invalid HUST explicit role file")
        for key, expected in (("train", HUST_FIT_IDS), ("val", HUST_VALID_IDS),
                              ("test", HUST_TEST_IDS)):
            if set(roles.loc[roles.role == key, "cell_id"]) != expected:
                raise ValueError(f"HUST {key} role IDs changed")
    elif split_manifest is None:
        raise ValueError("XJTU requires the original fold or role manifest")
    records = load_cell_records(cache, dataset=dataset, history=history,
        split_manifest=Path(split_manifest), test_fold=test_fold,
        split_seed=split_seed, include_roles=(role,))
    if not records or len({r.cell_id for r in records}) != len(records):
        raise ValueError("empty or duplicated physical cells")
    channels = 4 if dataset == "hust" else 5
    for record in records:
        if record.role != role or record.curves.shape != (history, 128, channels):
            raise ValueError("loaded role or native cache dimensions changed")
        if not np.isclose(record.lifetime - history, record.rul, rtol=0, atol=1e-10):
            raise ValueError("cached lifetime/RUL relationship changed")
        clean_viq(record)
    if dataset == "hust" and {r.cell_id for r in records} != (
        HUST_FIT_IDS if role == "train" else HUST_VALID_IDS
    ):
        raise ValueError("loaded HUST role incomplete")
    return sorted(records, key=lambda r: r.cell_id)


def viq18(values: np.ndarray) -> np.ndarray:
    """Same 18 definitions as diagnose_hust_generalization_20260924.feature_vector.

    Input can be raw or explicitly normalized/imputed VIQ. For clean-first models
    use raw VIQ and fit the feature standardizer on training cells only.
    """
    array = _validate_viq(values)
    t = np.linspace(0., 1., array.shape[0])
    centered_t = t - t.mean()
    result = []
    for channel in range(3):
        curve = array[:, :, channel]
        cycle_means = curve.mean(axis=1)
        slope = np.dot(centered_t, cycle_means) / np.dot(centered_t, centered_t)
        result.extend((curve.mean(), curve.std(ddof=0), curve.min(), curve.max(),
                       curve[-5:].mean() - curve[:5].mean(), slope))
    return np.asarray(result, dtype=np.float64)


def cycle_drift_features(values: np.ndarray) -> dict[str, np.ndarray]:
    """Describe observed early/late curve change, without implying monotone ageing.

    Curves are aligned on their existing normalized sampling grids, not matched
    physical voltage or equal elapsed-time coordinates across different cycles.
    No future cycles beyond the supplied observation history are accessed.
    """
    array = _validate_viq(values)
    early = array[:5].mean(axis=0)
    late = array[-5:].mean(axis=0)
    return {"cycle_means": array.mean(axis=1), "early_mean_curve": early,
            "late_mean_curve": late, "difference_curve": late - early}


def _validate_records(records: list[CellRecord], *, training: bool = False) -> tuple[int, int]:
    if not records or len({r.cell_id for r in records}) != len(records):
        raise ValueError("a non-empty unique physical-cell list is required")
    shape = clean_viq(records[0]).shape
    for record in records:
        if training and record.role != "train":
            raise ValueError("fitting statistics requires train-only records")
        if record.role not in ("train", "val"):
            raise ValueError("test records are not accepted by this development API")
        if clean_viq(record).shape != shape:
            raise ValueError("all records must use the same history and point count")
        if not np.isfinite([record.rul, record.lifetime]).all() or record.rul < 0:
            raise ValueError("invalid target")
        if not np.isclose(record.lifetime - shape[0], record.rul, rtol=0, atol=1e-10):
            raise ValueError("target history does not match input history")
    return shape[:2]


@dataclass(frozen=True)
class ReferenceStandardizer:
    """Train-only curve, feature and log-lifetime statistics with fit-ID receipt."""

    fit_ids: tuple[str, ...]
    history: int
    points: int
    channel_mean: np.ndarray
    channel_std: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, records: list[CellRecord]) -> ReferenceStandardizer:
        history, points = _validate_records(records, training=True)
        curves = np.stack([clean_viq(record) for record in records])
        features = np.stack([viq18(values) for values in curves])
        target = np.log1p([record.lifetime for record in records])
        channel_std = curves.std(axis=(0, 1, 2), ddof=0)
        feature_std = features.std(axis=0, ddof=0)
        channel_std[channel_std < 1e-8] = 1.
        feature_std[feature_std < 1e-8] = 1.
        return cls(tuple(sorted(record.cell_id for record in records)), history, points,
            curves.mean(axis=(0, 1, 2)), channel_std, features.mean(axis=0), feature_std,
            float(target.mean()), max(float(target.std(ddof=0)), 1e-8))

    def transform_viq(self, values: np.ndarray) -> np.ndarray:
        array = _validate_viq(values)
        if array.shape[:2] != (self.history, self.points):
            raise ValueError("normalizer history does not match input")
        return (array - self.channel_mean) / self.channel_std

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        self.transform_viq(values)  # Validate observation shape, without refitting.
        return (viq18(values) - self.feature_mean) / self.feature_std

    def encode_targets(self, records: list[CellRecord]) -> np.ndarray:
        if _validate_records(records) != (self.history, self.points):
            raise ValueError("target history does not match fitted task")
        return (np.log1p([r.lifetime for r in records]) - self.target_mean) / self.target_std

    def decode_rul(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("predictions must be finite")
        return np.maximum(np.expm1(np.clip(array * self.target_std + self.target_mean,
                                          -20., 20.)) - self.history, 0.)

    def to_dict(self) -> dict:
        return {"fit_ids": list(self.fit_ids), "history": self.history, "points": self.points,
                **{key: getattr(self, key).tolist() for key in (
                    "channel_mean", "channel_std", "feature_mean", "feature_std")},
                "target_mean": self.target_mean, "target_std": self.target_std,
                "target_definition": "standardized log1p(lifetime); decoded RUL=lifetime-history"}


def cell_metrics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One model: average individual errors over views, cells and conditions.

    Clean predictions may omit kind/rate/replicate. Repeated predictions cannot
    cancel before the error is calculated; conditions and physical cells receive
    equal weight. All cells must have the same set of observation conditions.
    """
    frame = raw.copy()
    required = {"cell_id", "target", "prediction"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("non-empty cell_id/target/prediction columns required")
    for name, value in (("kind", "clean"), ("rate", 0.), ("replicate", 0)):
        if name not in frame:
            frame[name] = value
    keys = ["cell_id", "kind", "rate", "replicate"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("missing or duplicate view identity")
    if not np.isfinite(frame[["target", "prediction"]]).all().all() or (frame.target < 0).any():
        raise ValueError("invalid target or prediction")
    if not frame.groupby("cell_id").target.nunique().eq(1).all():
        raise ValueError("one physical cell has inconsistent targets")
    condition_sets = [set(zip(group.kind, group.rate, strict=True))
                      for _, group in frame.groupby("cell_id")]
    if any(conditions != condition_sets[0] for conditions in condition_sets):
        raise ValueError("physical cells have unequal condition coverage")
    errors = frame.prediction - frame.target
    frame["mape"] = errors.abs() / frame.target.abs().clip(lower=1.)
    frame["mae"], frame["mse"] = errors.abs(), errors**2
    error_columns = ["mape", "mae", "mse"]
    cells = frame.groupby(["cell_id", "kind", "rate"], as_index=False)[error_columns].mean()
    conditions = cells.groupby(["kind", "rate"], as_index=False)[error_columns].mean()
    cells["rmse"], conditions["rmse"] = np.sqrt(cells.mse), np.sqrt(conditions.mse)
    summary = conditions[error_columns].mean().to_dict()
    summary.update(rmse=float(np.sqrt(summary["mse"])),
                   physical_cells=int(frame.cell_id.nunique()), conditions=len(conditions))
    return cells, conditions, summary
