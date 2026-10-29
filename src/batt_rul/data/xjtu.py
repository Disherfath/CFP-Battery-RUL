"""Minimal reader for the official 55-cell XJTU battery dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

XJTU_CURVE_COLUMNS = (
    "voltage_v",
    "current_a",
    "capacity_ah",
    "time_min",
    "temperature_c",
)
EXPECTED_BATCH_COUNTS = {1: 8, 2: 15, 3: 8, 4: 8, 5: 8, 6: 8}


def _scalar_text(value: Any) -> str:
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return ""
    return str(array[0])


def parse_xjtu_summary(summary: NDArray[Any]) -> tuple[int, str]:
    """Return author-provided cycle life and experiment description."""

    try:
        record = summary[0][0]
        cycle_life = int(np.asarray(record[8]).reshape(-1)[0])
        description = _scalar_text(record[9])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("unexpected XJTU summary schema") from error
    if cycle_life <= 0:
        raise ValueError("XJTU cycle life must be positive")
    return cycle_life, description


def parse_xjtu_path(path: Path) -> tuple[int, str]:
    try:
        batch = int(path.parent.name.split("-")[-1])
    except ValueError as error:
        raise ValueError(f"cannot infer XJTU batch from {path}") from error
    return batch, f"XJTU_B{batch}_{path.stem}"


def extract_xjtu_stage(
    cycle_record: NDArray[Any], variable: int, *, stage: int = 3
) -> NDArray[np.float64]:
    """Extract one phase using the relative-time resets in the author reader."""

    if variable < 0 or variable > 6:
        raise ValueError("variable must be one of the seven numeric XJTU fields")
    relative_time = np.asarray(cycle_record[1]).reshape(-1).astype(np.float64)
    values = np.asarray(cycle_record[variable]).reshape(-1).astype(np.float64)
    if relative_time.shape != values.shape:
        raise ValueError("XJTU variable length differs from relative time")
    starts = np.flatnonzero(relative_time == 0)
    if starts.size:
        starts = starts[np.r_[True, np.diff(starts) > 1]]
    boundaries = np.r_[starts, len(values)]
    if stage < 1 or stage >= len(boundaries):
        raise ValueError(f"stage {stage} is absent from XJTU cycle")
    result = values[boundaries[stage - 1] : boundaries[stage]]
    if result.size < 8:
        raise ValueError("XJTU discharge stage contains fewer than eight points")
    return result


def resample_xjtu_discharge(
    cycle_record: NDArray[Any], *, points: int = 128
) -> NDArray[np.float32]:
    """Extract stage 3 and linearly resample V/I/Q/t/T by sample position."""

    if points < 16:
        raise ValueError("points must be at least 16")
    variables = [2, 3, 4, 1, 6]
    raw = [extract_xjtu_stage(cycle_record, variable, stage=3) for variable in variables]
    lengths = {values.size for values in raw}
    if len(lengths) != 1:
        raise ValueError("XJTU discharge variables do not share a length")
    source = np.linspace(0.0, 1.0, raw[0].size)
    target = np.linspace(0.0, 1.0, points)
    values = np.column_stack([np.interp(target, source, channel) for channel in raw])
    values[:, 3] -= values[0, 3]
    if not np.isfinite(values).all():
        raise ValueError("resampled XJTU curve contains non-finite values")
    return values.astype(np.float32)


def read_xjtu_manifest_record(path: Path) -> dict[str, Any]:
    mat = loadmat(path, variable_names=["summary"])
    if "summary" not in mat:
        raise ValueError(f"summary missing from {path}")
    cycle_life, description = parse_xjtu_summary(mat["summary"])
    batch, cell_id = parse_xjtu_path(path)
    return {
        "dataset": "XJTU",
        "cell_id": cell_id,
        "batch": batch,
        "relative_path": str(Path(path.parent.name) / path.name),
        "cycle_life": cycle_life,
        "description": description,
        "file_bytes": path.stat().st_size,
    }


def assign_xjtu_test_folds(records: list[dict[str, Any]], *, seed: int) -> dict[str, int]:
    """Assign cells round-robin within each batch after stable SHA-256 ordering."""

    assignments: dict[str, int] = {}
    for batch in sorted(EXPECTED_BATCH_COUNTS):
        cells = [record["cell_id"] for record in records if record["batch"] == batch]
        expected = EXPECTED_BATCH_COUNTS[batch]
        if len(cells) != expected:
            raise ValueError(f"Batch-{batch} expected {expected} cells, found {len(cells)}")
        ordered = sorted(
            cells,
            key=lambda cell: hashlib.sha256(f"{seed}:{cell}".encode()).hexdigest(),
        )
        offset = (batch - 1) % 3
        for index, cell in enumerate(ordered):
            assignments[cell] = (index + offset) % 3
    if len(assignments) != sum(EXPECTED_BATCH_COUNTS.values()):
        raise ValueError("XJTU fold assignment is incomplete")
    return assignments


def assign_xjtu_roles(
    records: list[dict[str, Any]], *, test_fold: int, seed: int
) -> dict[str, str]:
    """Assign one frozen test fold and a small batch-stratified validation set."""

    if test_fold not in (0, 1, 2):
        raise ValueError("test_fold must be 0, 1, or 2")
    roles = {
        record["cell_id"]: "test" if int(record["test_fold"]) == test_fold else "train"
        for record in records
    }
    for batch in sorted(EXPECTED_BATCH_COUNTS):
        candidates = [
            record["cell_id"]
            for record in records
            if int(record["batch"]) == batch and roles[record["cell_id"]] == "train"
        ]
        count = 2 if batch == 2 else 1
        ordered = sorted(
            candidates,
            key=lambda cell: hashlib.sha256(
                f"{seed}:validation:{test_fold}:{cell}".encode()
            ).hexdigest(),
        )
        for cell in ordered[:count]:
            roles[cell] = "val"
    if set(roles.values()) != {"train", "val", "test"}:
        raise ValueError("XJTU role assignment is incomplete")
    return roles


def build_xjtu_cell_cache(
    path: Path,
    *,
    max_history: int = 50,
    points: int = 128,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Load one MAT file and retain only its first H discharge curves."""

    mat = loadmat(path, variable_names=["data", "summary"])
    if "data" not in mat or "summary" not in mat:
        raise ValueError(f"data or summary missing from {path}")
    cycle_life, description = parse_xjtu_summary(mat["summary"])
    batch, cell_id = parse_xjtu_path(path)
    data = mat["data"]
    if cycle_life <= max_history or data.shape[1] < max_history:
        raise ValueError(f"{cell_id} has insufficient early cycles")
    discharge_curves = []
    for cycle in range(max_history):
        try:
            discharge_curves.append(resample_xjtu_discharge(data[0][cycle], points=points))
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"{cell_id} cycle {cycle + 1}: {error}") from error
    curves = np.stack(discharge_curves)
    metadata = {
        "dataset": "XJTU",
        "cell_id": cell_id,
        "batch": batch,
        "cycle_life": cycle_life,
        "description": description,
        "max_history": max_history,
        "points": points,
        "rul_h20": cycle_life - 20,
        "rul_h50": cycle_life - 50,
    }
    arrays = {
        "curves": curves,
        "source_cycle_numbers": np.arange(1, max_history + 1, dtype=np.int32),
    }
    return metadata, arrays
