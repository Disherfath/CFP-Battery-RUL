"""Physical-cell split helpers and leakage assertions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def assert_disjoint_cell_splits(splits: Mapping[str, Iterable[str]]) -> None:
    """Raise when any physical cell appears in more than one split."""

    normalized = {name: set(map(str, cells)) for name, cells in splits.items()}
    names = list(normalized)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(f"cell leakage between {left} and {right}: {preview}")


def deterministic_cell_split(
    cell_ids: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict[str, list[str]]:
    """Create a deterministic train/validation/test split over unique cell IDs."""

    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("fractions must leave non-empty probability mass for test")
    unique = np.array(sorted(set(map(str, cell_ids))), dtype=object)
    if unique.size < 3:
        raise ValueError("at least three unique cells are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_end = max(1, round(train_fraction * unique.size))
    val_end = max(train_end + 1, round((train_fraction + val_fraction) * unique.size))
    val_end = min(val_end, unique.size - 1)
    result = {
        "train": unique[:train_end].tolist(),
        "val": unique[train_end:val_end].tolist(),
        "test": unique[val_end:].tolist(),
    }
    assert_disjoint_cell_splits(result)
    return result
