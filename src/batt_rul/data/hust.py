"""Minimal HUST adapter for fixed-early-history lifetime prediction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Frozen from microsoft/BatteryML at commit 2861ae3b8c79938c7fc8e6fe9986b799ca71c7dd.
HUST_TRAIN_IDS = frozenset(
    {
        "1-3", "1-4", "1-5", "1-6", "1-7", "1-8", "2-2", "2-3",
        "2-4", "2-6", "2-7", "2-8", "3-2", "3-3", "3-4", "3-5",
        "3-6", "3-7", "3-8", "4-1", "4-2", "4-3", "4-4", "4-6",
        "4-7", "4-8", "5-1", "5-2", "5-4", "5-5", "5-6", "5-7",
        "6-3", "6-4", "6-5", "7-1", "7-2", "7-3", "7-4", "7-7",
        "7-8", "8-2", "8-3", "8-4", "8-7", "9-1", "9-2", "9-3",
        "9-5", "9-7", "9-8", "10-2", "10-3", "10-5", "10-8",
    }
)

HUST_TEST_IDS = frozenset(
    {
        "1-1", "1-2", "2-5", "3-1", "4-5", "5-3", "6-1", "6-2",
        "6-6", "6-8", "7-5", "7-6", "8-1", "8-5", "8-6", "8-8",
        "9-4", "9-6", "10-1", "10-4", "10-6", "10-7",
    }
)

# Validation IDs published in the commented 47/8/22 split of BatteryML.
HUST_VALID_IDS = frozenset({"4-3", "5-7", "3-3", "2-3", "9-3", "10-5", "3-2", "3-7"})
HUST_FIT_IDS = HUST_TRAIN_IDS - HUST_VALID_IDS

SKIP_PREFIX_CYCLES = {"7-5": 2}
CURVE_COLUMNS = ("voltage_v", "current_a", "discharge_capacity_ah", "time_s")


def _require_columns(frame: pd.DataFrame) -> None:
    required = {
        "Status",
        "Current (mA)",
        "Voltage (V)",
        "Time (s)",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"HUST cycle is missing columns: {sorted(missing)}")


def resample_discharge_cycle(frame: pd.DataFrame, points: int = 128) -> NDArray[np.float32]:
    """Extract all discharge stages and resample them on normalized elapsed time."""

    if points < 16:
        raise ValueError("points must be at least 16")
    _require_columns(frame)
    is_discharge = frame["Status"].astype(str).str.contains("discharge", case=False, na=False)
    discharge = frame.loc[is_discharge]
    if len(discharge) < 8:
        raise ValueError("fewer than eight discharge observations")

    time_s = discharge["Time (s)"].to_numpy(dtype=np.float64)
    current_a = discharge["Current (mA)"].to_numpy(dtype=np.float64) / 1000.0
    voltage_v = discharge["Voltage (V)"].to_numpy(dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(current_a) & np.isfinite(voltage_v)
    time_s, current_a, voltage_v = time_s[finite], current_a[finite], voltage_v[finite]
    if time_s.size < 8:
        raise ValueError("fewer than eight finite discharge observations")

    order = np.argsort(time_s, kind="stable")
    time_s, current_a, voltage_v = time_s[order], current_a[order], voltage_v[order]
    time_s, unique_indices = np.unique(time_s, return_index=True)
    current_a, voltage_v = current_a[unique_indices], voltage_v[unique_indices]
    if time_s.size < 8 or time_s[-1] <= time_s[0]:
        raise ValueError("discharge timestamps are not usable")

    discharge_rate = np.abs(current_a)
    increments = 0.5 * (discharge_rate[1:] + discharge_rate[:-1]) * np.diff(time_s) / 3600.0
    capacity_ah = np.concatenate(([0.0], np.cumsum(increments)))
    grid = np.linspace(time_s[0], time_s[-1], points, dtype=np.float64)
    values = np.column_stack(
        [
            np.interp(grid, time_s, voltage_v),
            np.interp(grid, time_s, current_a),
            np.interp(grid, time_s, capacity_ah),
            grid - grid[0],
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("resampled HUST curve contains non-finite values")
    return values.astype(np.float32)


def load_hust_entry(path: Path) -> tuple[str, dict[str, Any]]:
    """Load one trusted pickle from the official Mendeley Data archive."""

    cell_id = path.stem
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # nosec: official archive, hash recorded by pipeline
    if cell_id not in payload or "data" not in payload[cell_id]:
        raise ValueError(f"unexpected HUST pickle schema: {path}")
    return cell_id, payload[cell_id]


def build_hust_cell_cache(
    path: Path,
    *,
    max_history: int = 50,
    points: int = 128,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Build a compact first-H-cycle cache and cell-level metadata."""

    cell_id, entry = load_hust_entry(path)
    cycle_data = entry["data"]
    cycle_keys = sorted(cycle_data, key=int)
    skipped = SKIP_PREFIX_CYCLES.get(cell_id, 0)
    usable_keys = cycle_keys[skipped:]
    if len(usable_keys) <= max_history:
        raise ValueError(f"{cell_id} has only {len(usable_keys)} usable cycles")

    selected = usable_keys[:max_history]
    curves = np.stack(
        [resample_discharge_cycle(cycle_data[key], points=points) for key in selected]
    )
    effective_life = len(usable_keys)
    if cell_id in HUST_FIT_IDS:
        split = "train"
    elif cell_id in HUST_VALID_IDS:
        split = "val"
    else:
        split = "test"
    if cell_id not in HUST_TRAIN_IDS | HUST_TEST_IDS:
        raise ValueError(f"{cell_id} is absent from the frozen official split")
    metadata = {
        "dataset": "HUST",
        "cell_id": cell_id,
        "split": split,
        "recorded_cycles": len(cycle_keys),
        "skipped_prefix_cycles": skipped,
        "effective_life": effective_life,
        "max_history": max_history,
        "points": points,
        "rul_h20": effective_life - 20,
        "rul_h50": effective_life - 50,
    }
    arrays = {
        "curves": curves,
        "source_cycle_numbers": np.asarray(selected, dtype=np.int32),
    }
    return metadata, arrays
