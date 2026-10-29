"""Deterministic synthetic missingness for within-cycle multivariate curves."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CorruptionResult:
    """Corrupted values and an observed-value mask with the same shape."""

    values: NDArray[np.float64]
    observed: NDArray[np.bool_]
    kind: str
    requested_rate: float

    @property
    def realized_rate(self) -> float:
        return float(1.0 - self.observed.mean())


@dataclass(frozen=True)
class CorruptedHistory:
    """A corrupted [cycle, point, channel] tensor and missingness metadata."""

    values: NDArray[np.float64]
    observed: NDArray[np.bool_]
    delta_t: NDArray[np.float64]
    kind: str
    requested_rate: float
    sensor_channels: tuple[int, ...]

    @property
    def realized_sensor_rate(self) -> float:
        sensor_mask = self.observed[:, :, self.sensor_channels]
        return float(1.0 - sensor_mask.mean())


def _validate(values: NDArray[np.floating], rate: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must have shape [time, channel]")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("values must contain at least two time points and one channel")
    if not np.isfinite(array).all():
        raise ValueError("natural missing values must be handled before synthetic corruption")
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must satisfy 0 <= rate < 1")
    return array


def apply_missingness(
    values: NDArray[np.floating],
    kind: str,
    rate: float,
    rng: np.random.Generator,
    *,
    channel: int | None = None,
) -> CorruptionResult:
    """Apply one corruption after the physical-cell split has been frozen.

    ``mcar_time`` removes random time indices synchronously from every channel.
    ``block`` removes one contiguous time interval from every channel.
    ``window`` retains one contiguous observation window and removes both tails.
    ``channel`` removes a complete channel; its realized rate is therefore 1/C.
    """

    array = _validate(values, rate)
    time_steps, channels = array.shape
    observed = np.ones_like(array, dtype=bool)
    normalized_kind = kind.lower()

    if normalized_kind == "clean" or rate == 0.0:
        pass
    elif normalized_kind == "mcar":
        count = max(1, round(rate * array.size))
        flat = rng.choice(array.size, size=count, replace=False)
        observed.flat[flat] = False
    elif normalized_kind == "mcar_time":
        count = max(1, round(rate * time_steps))
        missing_steps = rng.choice(time_steps, size=count, replace=False)
        observed[missing_steps, :] = False
    elif normalized_kind == "block":
        width = min(time_steps - 1, max(1, round(rate * time_steps)))
        start = int(rng.integers(0, time_steps - width + 1))
        observed[start : start + width, :] = False
    elif normalized_kind == "window":
        keep = max(1, round((1.0 - rate) * time_steps))
        start = int(rng.integers(0, time_steps - keep + 1))
        observed[:, :] = False
        observed[start : start + keep, :] = True
    elif normalized_kind == "channel":
        selected = int(rng.integers(0, channels)) if channel is None else int(channel)
        if not 0 <= selected < channels:
            raise ValueError(f"channel must be in [0, {channels - 1}]")
        observed[:, selected] = False
    else:
        raise ValueError(f"unknown missingness kind: {kind}")

    corrupted = array.copy()
    corrupted[~observed] = np.nan
    return CorruptionResult(corrupted, observed, normalized_kind, float(rate))


def delta_t_from_mask(
    observed: NDArray[np.bool_], coordinates: NDArray[np.floating] | None = None
) -> NDArray[np.float64]:
    """Return elapsed coordinate distance since the last observation per channel."""

    mask = np.asarray(observed, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("observed must have shape [time, channel]")
    if coordinates is None:
        coord = np.arange(mask.shape[0], dtype=np.float64)
    else:
        coord = np.asarray(coordinates, dtype=np.float64)
        if coord.shape != (mask.shape[0],):
            raise ValueError("coordinates must have shape [time]")
        if not np.isfinite(coord).all() or np.any(np.diff(coord) < 0):
            raise ValueError("coordinates must be finite and nondecreasing")

    gaps = np.zeros(mask.shape, dtype=np.float64)
    for feature in range(mask.shape[1]):
        last_observed = coord[0]
        for index in range(mask.shape[0]):
            if mask[index, feature]:
                last_observed = coord[index]
                gaps[index, feature] = 0.0
            else:
                gaps[index, feature] = coord[index] - last_observed
    return gaps


def _stable_seed(*parts: object) -> int:
    payload = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")


def apply_history_missingness(
    curves: NDArray[np.floating],
    kind: str,
    rate: float,
    *,
    base_seed: int,
    cell_id: str,
    replicate: int,
    sensor_channels: tuple[int, ...] = (0, 1, 2),
    time_channel: int = 3,
    dropped_channel: int | None = None,
) -> CorruptedHistory:
    """Corrupt a complete early-history tensor reproducibly and keep time observed."""

    array = np.asarray(curves, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("curves must have shape [cycle, point, channel]")
    if not np.isfinite(array).all():
        raise ValueError("natural missingness must be resolved before synthetic corruption")
    if len(set(sensor_channels)) != len(sensor_channels):
        raise ValueError("sensor_channels must be unique")
    if time_channel in sensor_channels:
        raise ValueError("time_channel must remain outside sensor_channels")
    if any(channel < 0 or channel >= array.shape[2] for channel in sensor_channels):
        raise ValueError("sensor channel is outside the curve tensor")
    if not 0 <= time_channel < array.shape[2]:
        raise ValueError("time_channel is outside the curve tensor")
    if dropped_channel is not None and dropped_channel not in sensor_channels:
        raise ValueError("dropped_channel must name a sensor channel")

    values = array.copy()
    observed = np.ones(array.shape, dtype=bool)
    normalized_kind = kind.lower()
    local_dropout = None
    if normalized_kind == "channel":
        if dropped_channel is None:
            raise ValueError("channel corruption requires an explicit dropped_channel")
        expected_rate = 1.0 / len(sensor_channels)
        if not np.isclose(rate, expected_rate):
            raise ValueError(
                f"channel corruption rate must equal its realized rate {expected_rate:g}"
            )
        selected = dropped_channel
        local_dropout = sensor_channels.index(selected)

    for cycle in range(array.shape[0]):
        rng = np.random.default_rng(
            _stable_seed(base_seed, cell_id, normalized_kind, rate, replicate, cycle)
        )
        result = apply_missingness(
            array[cycle][:, sensor_channels],
            normalized_kind,
            rate,
            rng,
            channel=local_dropout,
        )
        cycle_values = values[cycle]
        cycle_observed = observed[cycle]
        cycle_values[:, sensor_channels] = result.values
        cycle_observed[:, sensor_channels] = result.observed

    gaps = np.zeros_like(values, dtype=np.float64)
    for cycle in range(array.shape[0]):
        gaps[cycle] = delta_t_from_mask(observed[cycle], array[cycle, :, time_channel])
    return CorruptedHistory(
        values=values,
        observed=observed,
        delta_t=gaps,
        kind=normalized_kind,
        requested_rate=float(rate),
        sensor_channels=sensor_channels,
    )


def interpolate_history(
    corrupted: CorruptedHistory,
    *,
    time_channel: int = 3,
    fully_missing_fill: float | NDArray[np.floating] = 0.0,
) -> NDArray[np.float64]:
    """Linearly fill partial gaps; use a fixed value for a wholly missing channel.

    A per-channel training mean can be supplied for wholly missing raw-scale channels.
    """

    filled = corrupted.values.copy()
    fill = np.asarray(fully_missing_fill, dtype=np.float64)
    if fill.ndim == 0:
        fill = np.full(filled.shape[2], float(fill), dtype=np.float64)
    if fill.shape != (filled.shape[2],) or not np.isfinite(fill).all():
        raise ValueError("fully_missing_fill must be finite scalar or one value per channel")
    for cycle in range(filled.shape[0]):
        coordinate = filled[cycle, :, time_channel]
        for channel in corrupted.sensor_channels:
            observed = corrupted.observed[cycle, :, channel]
            if observed.all():
                continue
            if not observed.any():
                filled[cycle, :, channel] = fill[channel]
                continue
            filled[cycle, :, channel] = np.interp(
                coordinate,
                coordinate[observed],
                filled[cycle, observed, channel],
            )
    if not np.isfinite(filled).all():
        raise ValueError("interpolation did not resolve all synthetic missing values")
    return filled
