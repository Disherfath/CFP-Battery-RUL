"""Frozen paper mask and sampling functions extracted without numerical changes."""
import hashlib
import numpy as np
from .missingness import apply_history_missingness, interpolate_history
CONDITIONS = ("clean", "mcar_time40", "block40", "window40")
VIEW_CONDITION = np.array([0,1,1,1,2,2,2,3,3,3], dtype=np.int64)
VIEW_REPLICATE = np.array([0,0,1,2,0,1,2,0,1,2], dtype=np.int64)
UPDATES, SAMPLING_SEED = 800, 20261017

def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()

def build_views(raw, cell_id):
    """Build masks once per physical cell; the result is shared by all models."""
    views = []
    for condition in CONDITIONS:
        kind = "clean" if condition == "clean" else condition[:-2]
        rate = 0. if kind == "clean" else .4
        for replicate in range(1 if kind == "clean" else 3):
            corrupted = apply_history_missingness(raw, kind, rate, base_seed=20261004,
                cell_id=cell_id, replicate=replicate, sensor_channels=(0, 1, 2), time_channel=3)
            filled = interpolate_history(corrupted, time_channel=3)
            if not corrupted.observed[..., 3:].all():
                raise RuntimeError("time/non-VIQ channels were masked")
            np.testing.assert_array_equal(filled[..., 3:], raw[..., 3:])
            mask = corrupted.observed[..., :3]
            missing = (~mask).sum(axis=1)
            expected_count = 0 if kind == "clean" else 51
            if not np.all(missing == expected_count):
                raise RuntimeError("actual missing amounts are not matched")
            views.append({"condition": condition, "replicate": replicate,
                "mask_sha256": array_hash(mask), "actual_missing_rate": float(1.-mask.mean()),
                "requested_rate": rate, "mask": mask, "filled": filled[..., :3],
                "filled_viq_sha256": array_hash(filled[..., :3])})
    return views

def schedule(n, seed=SAMPLING_SEED):
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("at least two fitting physical cells required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1017, 50]))
    return {"cell": rng.integers(n, size=UPDATES, dtype=np.int64),
            "view": rng.integers(1, 10, size=UPDATES, dtype=np.int64)}

def sampling_schedule(n, updates, seed, batch_size=64):
    """Independent streams, equal-probability physical cells/cycles/conditions/reps."""
    if min(n, updates, batch_size) < 1:
        raise ValueError("positive cells, updates, and batch size required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 104729, 128]))
    shape = (updates, batch_size)
    cell = rng.integers(n, size=shape, dtype=np.int64)
    cycle = rng.integers(50, size=shape, dtype=np.int64)
    condition = rng.integers(3, size=shape, dtype=np.int64)
    replicate = rng.integers(3, size=shape, dtype=np.int64)
    return {"cell": cell, "cycle": cycle, "condition": condition + 1,
            "replicate": replicate, "view": 1 + 3 * condition + replicate}

