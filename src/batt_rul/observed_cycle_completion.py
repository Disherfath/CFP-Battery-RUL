"""Fixed visible-only H50 completion controls, with explicit LI fallback.

The cache progress grid is not physical time or matched SOC. Donors may be
before/after the recipient but are all within the supplied first 50 cycles;
these controls are not online per-cycle causal imputers. A True mask means a
cached point still visible after corruption, not a count of raw sensor samples.
No completed point is ever promoted to an observation or used recursively.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("cycle_mean", "cycle_affine", "nearest4", "nearest4_offset")
MAX_DONORS = 4
MIN_COMMON = 8
STATUS = {"observed": 0, "visible_mean": 1, "visible_affine": 2, "visible_donor": 3, "li_fallback": 4}


@dataclass(frozen=True)
class ObservedCycleCompletion:
    mode: str
    values: np.ndarray
    observed_counts: np.ndarray
    used_counts: np.ndarray
    fallback: np.ndarray
    status: np.ndarray
    mean_cycle_distance: np.ndarray
    weighted_offset: np.ndarray
    donor_indices: np.ndarray | None
    donor_weights: np.ndarray | None
    pair_common_counts: np.ndarray | None
    pair_offsets: np.ndarray | None
    affine_slope: np.ndarray | None


def _validate(linear_filled, observed):
    source = np.asarray(linear_filled)
    if source.dtype.kind not in "fiu":
        raise ValueError("finite real linear-filled values required")
    values = np.asarray(source, dtype=np.float64)
    mask = np.asarray(observed)
    if values.ndim < 3 or values.shape[-3:] != (50, 128, 3) or any(n < 1 for n in values.shape[:-3]):
        raise ValueError("expected nonempty [...,50,128,3] H50 VIQ curves")
    if not np.isfinite(values).all():
        raise ValueError("all linear-filled values, including hidden positions, must be finite")
    if mask.dtype.kind != "b" or mask.shape != values.shape:
        raise ValueError("same-shape bool observed mask required")
    return values, mask


def _pair_geometry(values, mask):
    """Whole-curve, channel-specific TRUE support, not a local neighborhood."""
    count = np.empty((50, 50, 3), dtype=np.int16)
    offsets = np.zeros((50, 50, 3), dtype=np.float64)
    for channel in range(3):
        visible = mask[..., channel]
        count[..., channel] = visible.astype(np.int16) @ visible.astype(np.int16).T
        for recipient in range(50):
            common = visible[recipient][None, :] & visible
            # Mask BEFORE subtraction: arbitrary finite hidden LI values are
            # neither offset anchors nor donor observations (no 0*huge values).
            difference = np.where(common, values[recipient, :, channel][None, :], 0.) - np.where(
                common, values[..., channel], 0.)
            offsets[recipient, :, channel] = (difference / np.maximum(count[recipient, :, channel, None], 1)).sum(1)
    return count, offsets


def complete_observed_cycles(linear_filled, observed, mode="nearest4_offset"):
    """Four predeclared rules, no labels/fitted global statistics/hyperparameter search.

Mean/affine use all truly visible cycles at each point/channel. Affine requires
>=2 distinct cycle indices and positive index variance; otherwise visible mean,
then supplied LI if none. Nearest controls use identical donors/weights: target
point truly visible in donor, self excluded, >=8 whole-curve common visible
progress points in that channel, nearest <=4 by distance then smaller index.
Weights are normalized 1/(1+abs(c-d)). Only offset mode adds the weighted mean
recipient-minus-donor difference on those common observed positions.

Returned counts/distances/offsets describe missing points only (zero for original
observations); donor indices are -1 for unused entries. Mean/affine report all
visible support counts and unweighted mean cycle distance, not affine leverage.
Values and diagnostics use float64; support/status never changes after filling.
"""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    values, mask = _validate(linear_filled, observed)
    leading = values.shape[:-3]
    batches = values.reshape(-1, 50, 128, 3)
    masks = mask.reshape(-1, 50, 128, 3)
    output = values.copy().reshape(-1, 50, 128, 3)
    count = mask.sum(-3, dtype=np.int16)
    used = np.zeros_like(output, dtype=np.int16)
    fallback = (~mask).copy().reshape(-1, 50, 128, 3)
    status = np.where(mask, STATUS["observed"], STATUS["li_fallback"]).astype(np.uint8).reshape(-1, 50, 128, 3)
    distance = np.zeros_like(output)
    weighted_offset = np.zeros_like(output)
    nearest = mode in ("nearest4", "nearest4_offset")
    indices = np.full(output.shape + (4,), -1, dtype=np.int16) if nearest else None
    weights = np.zeros(output.shape + (4,), dtype=np.float64) if nearest else None
    common_counts = np.empty((len(batches), 50, 50, 3), dtype=np.int16) if nearest else None
    pair_offsets = np.empty((len(batches), 50, 50, 3), dtype=np.float64) if nearest else None
    slopes = np.zeros((len(batches), 128, 3), dtype=np.float64) if mode == "cycle_affine" else None
    coordinate = np.arange(50, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for b, (source, visible) in enumerate(zip(batches, masks, strict=True)):
                n = visible.sum(0)
                if not nearest:
                    safe = np.where(visible, source, 0.)
                    mean = (safe / np.maximum(n, 1)).sum(0)
                    estimate = np.broadcast_to(mean, source.shape)
                    codes = np.full((128, 3), STATUS["visible_mean"], dtype=np.uint8)
                    if mode == "cycle_affine":
                        tmean = (coordinate[:, None, None] * visible / np.maximum(n, 1)).sum(0)
                        centered = np.where(visible, coordinate[:, None, None] - tmean, 0.)
                        variance_sum = np.square(centered).sum(0)
                        valid = (n >= 2) & (variance_sum > 0.)
                        slope = (centered * safe / np.where(variance_sum > 0., variance_sum, 1.)).sum(0)
                        slope = np.where(valid, slope, 0.)
                        slopes[b] = slope
                        estimate = mean + (coordinate[:, None, None] - tmean) * slope
                        codes = np.where(valid, STATUS["visible_affine"], STATUS["visible_mean"]).astype(np.uint8)
                    supported = (~visible) & (n > 0)
                    output[b] = np.where(supported, estimate, source)
                    used[b] = np.where(supported, n, 0)
                    fallback[b] = (~visible) & ~supported
                    status[b] = np.where(supported, codes, status[b])
                    for recipient in range(50):
                        dmean = (np.abs(coordinate - recipient)[:, None, None] * visible / np.maximum(n, 1)).sum(0)
                        distance[b, recipient] = np.where(supported[recipient], dmean, 0.)
                    continue
                common, offset = _pair_geometry(source, visible)
                common_counts[b], pair_offsets[b] = common, offset
                for recipient in range(50):
                    cycle_distance = np.abs(coordinate - recipient)
                    ordered = np.lexsort((coordinate, cycle_distance))
                    for channel in range(3):
                        eligible_pair = (common[recipient, ordered, channel] >= MIN_COMMON) & (ordered != recipient)
                        eligible = eligible_pair[:, None] & visible[ordered, :, channel]
                        ranks = np.where(eligible, np.arange(50)[:, None], 50)
                        ranked = np.argsort(ranks, axis=0, kind="stable")[:MAX_DONORS]
                        valid = np.take_along_axis(eligible, ranked, axis=0)
                        donors = ordered[ranked]
                        valid &= ~visible[recipient, :, channel][None, :]
                        weight = np.where(valid, 1. / (1. + np.abs(donors - recipient)), 0.)
                        denominator = weight.sum(0)
                        normalized = weight / np.where(denominator > 0., denominator, 1.)
                        donor_values = np.where(valid, source[donors, np.arange(128)[None, :], channel], 0.)
                        correction = np.where(valid, offset[recipient, donors, channel], 0.)
                        mean_offset = (normalized * correction).sum(0)
                        estimate = (normalized * donor_values).sum(0)
                        if mode == "nearest4_offset":
                            estimate = estimate + mean_offset
                        supported = (~visible[recipient, :, channel]) & (denominator > 0.)
                        output[b, recipient, :, channel] = np.where(supported, estimate, source[recipient, :, channel])
                        used[b, recipient, :, channel] = valid.sum(0)
                        fallback[b, recipient, :, channel] = (~visible[recipient, :, channel]) & ~supported
                        status[b, recipient, :, channel] = np.where(supported, STATUS["visible_donor"], status[b, recipient, :, channel])
                        distance[b, recipient, :, channel] = (normalized * np.abs(donors - recipient)).sum(0)
                        weighted_offset[b, recipient, :, channel] = mean_offset
                        indices[b, recipient, :, channel] = np.where(valid, donors, -1).T
                        weights[b, recipient, :, channel] = normalized.T
    except FloatingPointError as exc:
        raise ValueError("nonfinite observed-cycle completion computation") from exc
    if not np.isfinite(output).all() or not np.isfinite(distance).all() or not np.isfinite(weighted_offset).all():
        raise ValueError("nonfinite observed-cycle completion output")
    def history_shape(array):
        return array.reshape(leading + array.shape[1:])
    return ObservedCycleCompletion(mode, history_shape(output), count, history_shape(used),
        history_shape(fallback), history_shape(status), history_shape(distance), history_shape(weighted_offset),
        history_shape(indices) if nearest else None, history_shape(weights) if nearest else None,
        history_shape(common_counts) if nearest else None, history_shape(pair_offsets) if nearest else None,
        history_shape(slopes) if slopes is not None else None)
